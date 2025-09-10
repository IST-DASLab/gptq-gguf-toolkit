import json
import os
import time
import argparse

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from src import dist_utils
from src.quantizer import Quantizer
from src.common_utils import fix_seed
from src.metrics import compute_perplexity
from src.data_utils import get_data, get_wikitext2
from src.quant_utils import GGMLQuantizationType


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to quantized model.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--quantizable_modules",
        type=str,
        required=True,
        help="Regex for modules to quantize",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    parser.add_argument(
        "--post_block_modules",
        nargs="+",
        type=str,
        help="Names of modules after transformer blocks",
        default=[]
    )
    parser.add_argument(
        "--quant_non_block_modules",
        action="store_true",
        help="Whether to quantize non-block modules (e.g. embedding, lm_head).",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=int(2**20), type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Quantization params
    parser.add_argument(
        "--quant_scale",
        type=str,
        default="absmax",
        choices=["absmax", "mse"],
        help="Quantization scale",
    )
    parser.add_argument(
        "--act_order",
        action="store_true",
        help="Whether to permute in activation order.",
    )
    parser.add_argument("--static_groups", action="store_true")
    parser.add_argument("--rel_damp", type=float, default=1e-2)
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--default_bit_width", type=str, default="Q4_K")
    parser.add_argument("--bit_width_configuration", type=str, default=None)
    # K-Scales params
    parser.add_argument(
        "--rmin",
        type=float,
        default=-1.0,
        help="Minimum value for scale search.",
    )
    parser.add_argument(
        "--rdelta",
        type=float,
        default=0.1,
        help="Delta value for scale search.",
    )
    parser.add_argument(
        "--nstep",
        type=int,
        default=20,
        help="Number of steps for scale search.",
    )
     # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Log to W&B")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--eval_perplexity", action="store_true", help="whether to eval perplexity after quantization.")
    parser.add_argument("--eval_sequence_length", type=int, default=4096, help="sequence length for eval perplexity.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    # Save params
    parser.add_argument("--save_dir", type=str, required=True, help="where to save sparse model.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    device = "cuda"
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # Init device
    device = f"cuda:{rank}"
    torch.cuda.set_device(device)

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )

    print(model)
    if not args.cpu_offload_modules:
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # Take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    
    if args.default_bit_width is None and args.bit_width_configuration is None:
        raise ValueError("Either default_bit_width or bit_width_configuration must be provided.")

    if args.default_bit_width is not None:
        if args.default_bit_width not in ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]:
            raise ValueError("default_bit_width must be one of [Q2_K, Q3_K, Q4_K, Q5_K, Q6_K]")
        
        bit_width = GGMLQuantizationType[args.default_bit_width]
        quant_config = {
            "q_proj": bit_width,
            "k_proj": bit_width,
            "v_proj": bit_width,
            "o_proj": bit_width,
            "gate_proj": bit_width,
            "down_proj": bit_width,
            "up_proj": bit_width,
            "embed_tokens": bit_width,
            "lm_head": bit_width,
        }
        
    if args.bit_width_configuration is not None:
        if not os.path.isfile(args.bit_width_configuration):
            raise ValueError("bit_width_configuration must be a valid file path.")

        # Load JSON file directly
        with open(args.bit_width_configuration, "r") as f:
            bit_width_config = json.load(f)

        quant_config = {}
        for key, value in bit_width_config.items():
            if value not in ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"]:
                raise ValueError(
                    "All bit widths in bit_width_configuration must be one of [Q2_K, Q3_K, Q4_K, Q5_K, Q6_K]"
                )
            quant_config[key] = GGMLQuantizationType[value]
            
    quantizer = Quantizer(
        model,
        data_loader=calibration_data,
        quantizable_modules=args.quantizable_modules,
        quantizer_kwargs=dict(
            rel_damp=args.rel_damp,
            block_size=args.block_size,
            act_order=args.act_order,
            quant_scale=args.quant_scale,
            static_groups=args.static_groups,
            rmin=args.rmin,
            rdelta=args.rdelta,
            nstep=args.nstep,
            verbose=args.verbose,
        ),
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        post_block_modules=args.post_block_modules,
        quant_non_block_modules=args.quant_non_block_modules,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        device=device,
        verbose=args.verbose,
        save_dir=args.save_dir
    )
    
    # Prepare save dir
    if dist_utils.is_main():
        os.makedirs(args.save_dir, exist_ok=True)

    dist.barrier()

    t1 = time.perf_counter()
    quantizer.quantize(quant_config)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Quantization took {(t2 - t1)} s.")

    dist.barrier()

    if dist_utils.is_main():
        if not args.cpu_offload_modules:
            model = model.to(device)

        if args.eval_perplexity:
            fix_seed(args.seed)
            print("Evaluating perplexity on Wikitext-2...") 
            eval_data = get_wikitext2(100, args.eval_sequence_length, tokenizer, train=False)
            ppl = compute_perplexity(model, eval_data)
            print(f"Wikitext-2 perplexity: {ppl:.3f}")


if __name__ == "__main__":
    main()
