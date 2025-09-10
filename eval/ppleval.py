import os
import argparse
import json
from functools import partial
from typing import Optional, Union, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.model_utils import drop_layers_from_config

from src.metrics import compute_perplexity, compute_perplexity_layer_per_layer


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    # Data params
    parser.add_argument("--sequence_length", default=None, type=int, help="Length of sequences.")
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["wikitext2", "c4", "fineweb_edu"],
        help="Datasets used for evaluation",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Batch size on evaluation",
    )
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    # Loading params
    parser.add_argument("--drop_layer_config", type=str, default=None, help="Path to layer dropping configuration.")
    # Sparsification params
    parser.add_argument(
        "--sparse_weights_path",
        type=str,
        default=None,
        help="Path to sparse weights",
    )
    parser.add_argument(
        "--sparse_config_path",
        type=str,
        default=None,
        help="Path to sparsification config",
    )
    parser.add_argument(
        "--sparse_default_level",
        type=int,
        default=0,
        help="Default sparsity level",
    )
    # Quantization params
    parser.add_argument(
        "--quant_weights_path",
        type=str,
        default=None,
        help="Path to quantized weights",
    )
    parser.add_argument(
        "--quant_config_path",
        type=str,
        default=None,
        help="Path to quantization config",
    )
    parser.add_argument(
        "--quant_default_level",
        type=int,
        default=0,
        help="Default quantization level",
    )
    # Output params
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output JSON file for storing results",
    )
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--verbose", action="store_true", help="Whether to log progress.")
    parser.add_argument(
        "--memory_efficient", action="store_true", help="Whether to use memory efficient implementation."
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    args = parser.parse_args()
    return args


# Compressed model loader
def load_compressed_weights(
        model: AutoModelForCausalLM,
        compressed_weights_path: Union[str, os.PathLike],
        compressed_config_path: Optional[str] = None,
        default_level: int = 0,
):
    # Load weights from configuration if provided
    if compressed_config_path:
        with open(os.path.join(compressed_config_path), "r") as f:
            for line in f:
                layer_name, level = line.split(":")
                layer = model.get_submodule(layer_name.strip(" "))
                orig_dtype = layer.weight.dtype
                layer.weight.data = torch.load(
                    os.path.join(compressed_weights_path, layer_name, f"{int(level)}.pth"),
                    map_location=layer.weight.device,
                ).to(orig_dtype)
    # Otherwise load uniform configuration
    else:
        for layer_name in sorted(os.listdir(compressed_weights_path)):
            if not os.path.isdir(os.path.join(compressed_weights_path, layer_name)):
                continue
            layer = model.get_submodule(layer_name.strip(" "))
            orig_dtype = layer.weight.dtype
            layer.weight.data = torch.load(
                os.path.join(compressed_weights_path, layer_name, f"{default_level}.pth"),
                map_location=layer.weight.device,
            ).to(orig_dtype)
    return model


def main():
    args = parse_args()
    # Get device and dtype
    assert torch.cuda.is_available()
    device = torch.device("cuda")
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=None if args.memory_efficient else "auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache

    if args.drop_layer_config:
        drop_layers_from_config(model, args.drop_layer_config)
    elif args.sparse_weights_path:
        load_compressed_weights(model, args.sparse_weights_path, args.sparse_config_path, args.sparse_default_level)
    elif args.quant_weights_path:
        load_compressed_weights(model, args.quant_weights_path, args.quant_config_path, args.quant_default_level)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=args.use_fast_tokenizer)
    args.sequence_length = args.sequence_length or model.config.max_position_embeddings

    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.sequence_length,
                tokenizer,
                train=False,
            )
        )

    if args.memory_efficient:
        compute_ppl_fn = partial(compute_perplexity_layer_per_layer, device=device, batch_size=args.eval_batch_size)
    else:
        compute_ppl_fn = partial(compute_perplexity, batch_size=args.eval_batch_size)

    # evaluate before layer dropping
    log_dict = {}
    results = {
        "model_name_or_path": args.model_name_or_path,
        "evaluation_config": {
            "sequence_length": args.sequence_length,
            "eval_datasets": args.eval_datasets,
            "eval_batch_size": args.eval_batch_size,
            "eval_tokens": args.eval_tokens,
            "seed": args.seed,
            "dtype": str(args.dtype),
            "memory_efficient": args.memory_efficient,
            "attn_implementation": args.attn_implementation,
        },
        "compression_config": {
            "drop_layer_config": args.drop_layer_config,
            "sparse_weights_path": args.sparse_weights_path,
            "sparse_config_path": args.sparse_config_path,
            "sparse_default_level": args.sparse_default_level,
            "quant_weights_path": args.quant_weights_path,
            "quant_config_path": args.quant_config_path,
            "quant_default_level": args.quant_default_level,
        },
        "perplexity_results": {}
    }

    print("-" * 10)
    print("Evaluation before compression.")
    print(f"Test perplexities")
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_ppl_fn(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
        results["perplexity_results"][eval_dataset_name] = float(ppl_eval)
    print("-" * 10)

    # Save results to JSON file
    try:
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output_file}")
    except Exception as e:
        print(f"Error saving results to {args.output_file}: {e}")

    if args.log_wandb:
        wandb.log(log_dict)


if __name__ == "__main__":
    main()