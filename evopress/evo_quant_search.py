import argparse
import random
import copy
import os
import math
from tqdm import trange
from typing import List, Tuple, Sequence, Optional, Union, Dict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.metrics import compute_perplexity, compute_kl_div, compute_sparse_kl_div
from src.model_utils import layer_order_fn, group_layers


def scan_available_bitwidths(quant_weights_path: str) -> Dict[str, List[Tuple[float, str]]]:
    """Scan directories and build a dictionary of available bitwidths.

    Returns:
        Dictionary mapping layer names to list of (bitwidth, filename) tuples, sorted by bitwidth.
    """
    available_bitwidths = {}

    for layer_name in os.listdir(quant_weights_path):
        layer_path = os.path.join(quant_weights_path, layer_name)
        if os.path.isdir(layer_path):
            bitwidths = []
            for filename in os.listdir(layer_path):
                if filename.endswith('.pth'):
                    # Extract bitwidth from filename (e.g., "2.5625-Q2_K.pth" -> 2.5625)
                    bitwidth_str = filename.split('-')[0]
                    try:
                        bitwidth = float(bitwidth_str)
                        bitwidths.append((bitwidth, filename))
                    except ValueError:
                        print(f"Warning: Could not parse bitwidth from {filename}")

            # Sort by bitwidth
            bitwidths.sort(key=lambda x: x[0])
            available_bitwidths[layer_name] = bitwidths

    return available_bitwidths


def calculate_total_bits(current_bitwidths: List[List[float]],
                         grouped_layer_names: List[List[str]],
                         model) -> int:
    """Calculate total bits used by current configuration."""
    total_bits = 0
    for g_id in range(len(grouped_layer_names)):
        for l_id, l_name in enumerate(grouped_layer_names[g_id]):
            numel = model.get_submodule(l_name).weight.numel()
            total_bits += numel * current_bitwidths[g_id][l_id]
    return total_bits


def get_next_bitwidth(current_bitwidths: List[List[float]],
                      target_bits: int,
                      grouped_layer_names: List[List[str]],
                      available_bitwidths: Dict[str, List[Tuple[float, str]]],
                      model,
                      group_id: int,
                      layer_id: int,
                      direction: str = 'decrease') -> Optional[float]:
    """Get the next available bitwidth that keeps us within budget.

    For decrease: returns the highest available bitwidth that's lower than current.
    For increase: returns the lowest available bitwidth that's higher than current and within budget.
    """
    layer_name = grouped_layer_names[group_id][layer_id]
    current_bw = current_bitwidths[group_id][layer_id]
    layer_numel = model.get_submodule(layer_name).weight.numel()

    if direction == 'decrease':
        # Get all bitwidths lower than current
        candidates = [(bw, fn) for bw, fn in available_bitwidths[layer_name] if bw < current_bw]
        if not candidates:
            return None
        # Return the highest one (last in sorted list)
        return candidates[-1][0]

    else:  # increase
        # Get all bitwidths higher than current
        candidates = [(bw, fn) for bw, fn in available_bitwidths[layer_name] if bw > current_bw]
        if not candidates:
            return None

        # Calculate current total bits
        current_total_bits = calculate_total_bits(current_bitwidths, grouped_layer_names, model)

        # Find the lowest bitwidth that keeps us within budget
        for bw, _ in candidates:  # Start from lowest
            bits_increase = layer_numel * (bw - current_bw)
            if current_total_bits + bits_increase <= target_bits:
                return bw

        return None


def load_layers(
        model: AutoModelForCausalLM,
        grouped_layer_names: Tuple[Sequence[str]],
        new_state: Tuple[Sequence[float]],  # Changed to float
        quant_weights_path: str,
        available_bitwidths: Dict[str, List[Tuple[float, str]]],
):
    assert hasattr(model, "state")
    num_groups = len(grouped_layer_names)
    for i in range(num_groups):
        for layer_name, new_bitwidth, old_bitwidth in zip(grouped_layer_names[i], new_state[i], model.state[i]):
            if new_bitwidth != old_bitwidth:
                # Find the filename for this bitwidth
                filename = None
                for bw, fn in available_bitwidths[layer_name]:
                    if abs(bw - new_bitwidth) < 1e-6:  # Float comparison
                        filename = fn
                        break

                if filename is None:
                    raise ValueError(f"No file found for layer {layer_name} with bitwidth {new_bitwidth}")

                layer = model.get_submodule(layer_name)
                layer.weight.data = torch.load(
                    os.path.join(quant_weights_path, layer_name, filename),
                    map_location=layer.weight.device
                ).to(layer.weight.dtype)
    # Update model state
    model.state = new_state


def compute_fitness(model, data, fitness_fn, target_logits: Optional[torch.Tensor] = None) -> float:
    if fitness_fn == "ppl":
        return compute_perplexity(model, data)
    elif fitness_fn == "kl":
        return compute_kl_div(model, data, target_logits)
    elif fitness_fn == "sparse_kl":
        return compute_sparse_kl_div(model, data, target_logits)


def selection(
        model,
        grouped_layer_names,
        quant_weights_path: str,
        available_bitwidths: Dict[str, List[Tuple[float, str]]],
        candidates,
        num_survive: int,
        calibration_data,
        num_tokens: int,
        fitness_fn: str = "ppl",
        target_logits: Optional[Union[List[torch.Tensor], Tuple[torch.Tensor]]] = None,
):
    calibration_minibatch = []
    minibatch_ids = []
    target_logits_minibatch = []
    tokens_used = 0
    while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
        minibatch_id = random.randint(0, len(calibration_data) - 1)
        if minibatch_id in minibatch_ids:  # avoid duplicates
            continue
        minibatch_ids.append(minibatch_id)
        if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
            calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
            if fitness_fn == "kl":
                target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
            elif fitness_fn == "sparse_kl":
                target_logits_minibatch.append(
                    (
                        target_logits[minibatch_id][0][:, : num_tokens - tokens_used],  # TopK indices
                        target_logits[minibatch_id][1][:, : num_tokens - tokens_used],  # TopK values
                    )
                )
            tokens_used = num_tokens
        else:
            calibration_minibatch.append(calibration_data[minibatch_id])
            if fitness_fn in ["kl", "sparse_kl"]:
                target_logits_minibatch.append(target_logits[minibatch_id])
            tokens_used += calibration_data[minibatch_id].shape[1]

    if len(target_logits_minibatch) == 0:
        target_logits_minibatch = None

    fitnesses = []
    for candidate in candidates:
        load_layers(model, grouped_layer_names, candidate, quant_weights_path, available_bitwidths)
        fitness = compute_fitness(model, calibration_minibatch, fitness_fn, target_logits_minibatch)
        fitnesses.append(fitness)
    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", default=524288, type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl", "sparse_kl"], default="kl", help="Fitness function.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search params
    parser.add_argument("--generations", type=int, required=True, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, required=True, help="Number of offspring generated in each generation")
    parser.add_argument(
        "--target_bitwidth",
        type=float,
        required=True,
        help="Base level for all layers. If no integer, initialize random with this average",
    )
    parser.add_argument("--quant_weights_path", type=str, required=True, help="Path to quantized weights")
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--tokens_per_selection",
        type=int,
        nargs="+",
        required=True,
        help="Number of calibration tokens at each stage of selection",
    )
    parser.add_argument(
        "--initially_generated",
        type=int,
        help="Only for non-integer initial level: Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        help="Only for non-integer initial level: Number of calibration tokens used for the initial generation",
    )
    parser.add_argument(
        "--group_rule",
        type=str,
        default="size",
        choices=["size", "name", "none"],
        help="Layer grouping rule. Mutations are performed only within a group.",
    )
    parser.add_argument(
        "--kl_topk",
        type=int,
        default=10,
        help="TopK logits in KL-divergence (for sparse_kl fitness function)",
    )
    # Note: step_size is no longer needed with float bitwidths
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
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


def main():
    args = parse_args()
    # Sanity checks
    assert len(args.survivors_per_selection) == len(args.tokens_per_selection), "Must have same number of stages"
    assert args.survivors_per_selection[-1] == 1, "Last stage should have only one survivor"
    if int(args.target_bitwidth) != args.target_bitwidth:
        assert args.initially_generated is not None, "Need initially_generated for non-integer initial level"
        assert args.initial_tokens is not None, "Need initial_tokens for non-integer initial level"
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # init device
    device = f"cuda"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False  # do not use cache
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or min(
        model.config.max_position_embeddings, 8192
    )
    calibration_data = get_data(
        args.calibration_data, args.calibration_tokens, args.calibration_sequence_length, tokenizer, train=True
    )
    # Load eval datasets
    args.eval_sequence_length = args.eval_sequence_length or min(model.config.max_position_embeddings, 8192)
    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        e_d = get_data(
            eval_dataset_name,
            args.eval_tokens,  # ignored for WikiText2 and C4
            args.eval_sequence_length,
            tokenizer,
            train=False,
        )
        eval_datasets.append(e_d)
    target_logits = []
    if args.fitness_fn == "kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    elif args.fitness_fn == "sparse_kl":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                logits = model(calibration_data[i].to(device)).logits.cpu()
                topk_values, topk_indices = logits.topk(k=args.kl_topk, dim=-1)
                target_logits.append((topk_values, topk_indices))

    # Scan available bitwidths
    available_bitwidths = scan_available_bitwidths(args.quant_weights_path)

    # print available bitwidths
    print("Available bitwidths:")
    for layer_name, bitwidths in available_bitwidths.items():
        print(f"{layer_name}: {[bw for bw, _ in bitwidths]}")

    # Prepare layers and initial state
    layer_names = list(available_bitwidths.keys())
    # Sort layers
    layer_names = sorted(layer_names, key=layer_order_fn)
    # Group layers
    grouped_layer_names = group_layers(model, layer_names, args.group_rule)
    print(grouped_layer_names)
    num_groups = len(grouped_layer_names)
    # Loaded state - now stores actual bitwidths (floats)
    model.state = [[None] * len(names) for names in grouped_layer_names]

    target_bits = 0
    quantizable_weights = 0
    for group_id in range(len(grouped_layer_names)):
        for i, layer_name in enumerate(grouped_layer_names[group_id]):
            target_bits += int(model.get_submodule(layer_name).weight.numel() * args.target_bitwidth)
            quantizable_weights += model.get_submodule(layer_name).weight.numel()

    # Initialization
    if int(args.target_bitwidth) == args.target_bitwidth:
        # Check if this exact bitwidth is available for all layers
        parent = []
        for group_names in grouped_layer_names:
            group_bitwidths = []
            for layer_name in group_names:
                # Find closest available bitwidth
                available_bws = [bw for bw, _ in available_bitwidths[layer_name]]
                if args.target_bitwidth in available_bws:
                    group_bitwidths.append(args.target_bitwidth)
                else:
                    # Find closest
                    closest = min(available_bws, key=lambda x: abs(x - args.target_bitwidth))
                    group_bitwidths.append(closest)
            parent.append(group_bitwidths)
        train_fitness = float("inf")
    else:
        candidates = []
        for _ in range(args.initially_generated):
            # Start with bitwidths close to target and adjust randomly
            candidate = []
            for group_names in grouped_layer_names:
                group_bitwidths = []
                for layer_name in group_names:
                    # Get available bitwidths for this layer
                    available_bws = [bw for bw, _ in available_bitwidths[layer_name]]
                    # Start with the one closest to ceil(target_bitwidth)
                    closest = min(available_bws, key=lambda x: abs(x - math.ceil(args.target_bitwidth)))
                    group_bitwidths.append(closest)
                candidate.append(group_bitwidths)

            candidate_bits = calculate_total_bits(candidate, grouped_layer_names, model)

            # Decrease until we're under budget
            max_iterations = 1000  # Safety limit
            iterations = 0
            while candidate_bits > target_bits and iterations < max_iterations:
                iterations += 1
                # Select random group, proportional to the number of layers in a group
                group_id = random.choices(
                    range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                )[0]

                decr_ids = []
                for i in range(len(grouped_layer_names[group_id])):
                    next_bw = get_next_bitwidth(
                        candidate, target_bits, grouped_layer_names, available_bitwidths,
                        model, group_id, i, 'decrease'
                    )
                    if next_bw is not None:
                        decr_ids.append(i)

                if len(decr_ids) == 0:
                    break  # No way to decrease further

                decr_id = random.choice(decr_ids)
                old_bw = candidate[group_id][decr_id]
                new_bw = get_next_bitwidth(
                    candidate, target_bits, grouped_layer_names, available_bitwidths,
                    model, group_id, decr_id, 'decrease'
                )
                candidate[group_id][decr_id] = new_bw
                candidate_bits = calculate_total_bits(candidate, grouped_layer_names, model)

            candidates.append(candidate)

        candidates, train_fitnesses = selection(
            model=model,
            grouped_layer_names=grouped_layer_names,
            quant_weights_path=args.quant_weights_path,
            available_bitwidths=available_bitwidths,
            candidates=candidates,
            num_survive=1,
            calibration_data=calibration_data,
            num_tokens=args.initial_tokens,
            fitness_fn=args.fitness_fn,
            target_logits=target_logits,
        )
        train_fitness = train_fitnesses[0]
        parent = candidates[0]

    parent_bits = calculate_total_bits(parent, grouped_layer_names, model)

    log_dict = {}
    for generation in range(args.generations):
        print(f"Generation {generation + 1}/{args.generations}")
        print(f"Current search point:")
        for group in parent:
            print(group)
        print(f"Parent bits: {parent_bits}")
        print(f"Bit average: {parent_bits / quantizable_weights:.4e}")
        print(f"Train fitness: {train_fitness:.4e}")

        load_layers(model, grouped_layer_names, parent, args.quant_weights_path, available_bitwidths)

        # Evaluate current search point
        if generation % args.eval_every == 0:
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
                log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
            ppl_train = compute_perplexity(model, calibration_data)
            print(f"ppl_train: {ppl_train:.2f}")
            log_dict["ppl_train"] = ppl_train
        if args.log_wandb:
            wandb.log(log_dict)

        offspring_list = []
        duplicate_offspring_ct = 0

        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(parent)
            # mutate offspring
            num_flips = min(random.randint(1, 3), random.randint(1, 3))  # bias towards lower values

            if args.group_rule == "none":  # there can be mutations between layers of different sizes
                offspring_bits = calculate_total_bits(offspring, grouped_layer_names, model)

                # First, make sure we're under budget
                max_iterations = 1000
                iterations = 0
                while offspring_bits > target_bits and iterations < max_iterations:
                    iterations += 1
                    # Select random group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]

                    decr_ids = []
                    for i in range(len(grouped_layer_names[group_id])):
                        next_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, i, 'decrease'
                        )
                        if next_bw is not None:
                            decr_ids.append(i)

                    if len(decr_ids) == 0:
                        break

                    decr_id = random.choice(decr_ids)
                    old_bw = offspring[group_id][decr_id]
                    new_bw = get_next_bitwidth(
                        offspring, target_bits, grouped_layer_names, available_bitwidths,
                        model, group_id, decr_id, 'decrease'
                    )
                    offspring[group_id][decr_id] = new_bw
                    offspring_bits = calculate_total_bits(offspring, grouped_layer_names, model)

                # Now try to increase bitwidths
                bits_added = 0
                bits_removed = 0
                successful_increases = 0
                decrease_attempts = 0  # Track additional decreases needed

                for _ in range(num_flips):
                    # Select random group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]

                    incr_ids = []
                    for i in range(len(grouped_layer_names[group_id])):
                        next_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, i, 'increase'
                        )
                        if next_bw is not None:
                            incr_ids.append(i)

                    if len(incr_ids) == 0:
                        # Can't increase, try to decrease something else to make room
                        for _ in range(3):  # Try up to 3 times
                            decrease_attempts += 1
                            decr_group_id = random.choices(
                                range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                            )[0]

                            decr_ids = []
                            for i in range(len(grouped_layer_names[decr_group_id])):
                                next_bw = get_next_bitwidth(
                                    offspring, target_bits, grouped_layer_names, available_bitwidths,
                                    model, decr_group_id, i, 'decrease'
                                )
                                if next_bw is not None:
                                    decr_ids.append(i)

                            if decr_ids:
                                decr_id = random.choice(decr_ids)
                                old_bw = offspring[decr_group_id][decr_id]
                                new_bw = get_next_bitwidth(
                                    offspring, target_bits, grouped_layer_names, available_bitwidths,
                                    model, decr_group_id, decr_id, 'decrease'
                                )
                                offspring[decr_group_id][decr_id] = new_bw
                                layer_numel = model.get_submodule(
                                    grouped_layer_names[decr_group_id][decr_id]).weight.numel()
                                bits_removed += layer_numel * (old_bw - new_bw)
                                # Recalculate available increases
                                incr_ids = []
                                for i in range(len(grouped_layer_names[group_id])):
                                    next_bw = get_next_bitwidth(
                                        offspring, target_bits, grouped_layer_names, available_bitwidths,
                                        model, group_id, i, 'increase'
                                    )
                                    if next_bw is not None:
                                        incr_ids.append(i)
                                if incr_ids:
                                    break

                    if incr_ids:
                        incr_id = random.choice(incr_ids)
                        old_bw = offspring[group_id][incr_id]
                        new_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, incr_id, 'increase'
                        )
                        offspring[group_id][incr_id] = new_bw
                        layer_numel = model.get_submodule(grouped_layer_names[group_id][incr_id]).weight.numel()
                        bits_added += layer_numel * (new_bw - old_bw)
                        successful_increases += 1

                # Quality checks
                if successful_increases == 0 and decrease_attempts > 5:
                    continue  # Too many decreases without increases

                #if bits_added > 0 and bits_removed > 0:
                #    if bits_added / bits_removed < 0.5:  # Too much removed for what was added
                #        continue

            else:  # only mutations between layers of same size/type
                # First ensure we're under budget
                offspring_bits = calculate_total_bits(offspring, grouped_layer_names, model)
                if offspring_bits > target_bits:
                    # Need to decrease first
                    max_iterations = 100
                    iterations = 0
                    while offspring_bits > target_bits and iterations < max_iterations:
                        iterations += 1
                        group_id = random.choices(
                            range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                        )[0]

                        decr_ids = []
                        for i in range(len(grouped_layer_names[group_id])):
                            next_bw = get_next_bitwidth(
                                offspring, target_bits, grouped_layer_names, available_bitwidths,
                                model, group_id, i, 'decrease'
                            )
                            if next_bw is not None:
                                decr_ids.append(i)

                        if len(decr_ids) == 0:
                            break

                        decr_id = random.choice(decr_ids)
                        new_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, decr_id, 'decrease'
                        )
                        offspring[group_id][decr_id] = new_bw
                        offspring_bits = calculate_total_bits(offspring, grouped_layer_names, model)

                # Now do the mutations
                successful_mutations = 0
                for _ in range(num_flips):
                    # Select random group, proportional to the number of layers in a group
                    group_id = random.choices(
                        range(len(grouped_layer_names)), weights=[len(g) for g in grouped_layer_names]
                    )[0]

                    # Positions where compression can be decreased
                    decr_ids = []
                    for i in range(len(grouped_layer_names[group_id])):
                        next_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, i, 'decrease'
                        )
                        if next_bw is not None:
                            decr_ids.append(i)

                    if len(decr_ids) == 0:
                        print("Can't decrease bits, continue ...")
                        continue

                    decr_id = random.choice(decr_ids)

                    # Positions where compression can be increased
                    incr_ids = []
                    for i in range(len(grouped_layer_names[group_id])):
                        next_bw = get_next_bitwidth(
                            offspring, target_bits, grouped_layer_names, available_bitwidths,
                            model, group_id, i, 'increase'
                        )
                        if next_bw is not None:
                            incr_ids.append(i)

                    if len(incr_ids) == 0:
                        # Try to decrease something else first to make room
                        other_decr_id = random.choice([i for i in decr_ids if i != decr_id]) if len(
                            decr_ids) > 1 else None
                        if other_decr_id is not None:
                            new_bw = get_next_bitwidth(
                                offspring, target_bits, grouped_layer_names, available_bitwidths,
                                model, group_id, other_decr_id, 'decrease'
                            )
                            offspring[group_id][other_decr_id] = new_bw

                            # Check again for increase possibilities
                            incr_ids = []
                            for i in range(len(grouped_layer_names[group_id])):
                                next_bw = get_next_bitwidth(
                                    offspring, target_bits, grouped_layer_names, available_bitwidths,
                                    model, group_id, i, 'increase'
                                )
                                if next_bw is not None:
                                    incr_ids.append(i)

                        if len(incr_ids) == 0:
                            print("Still can't increase bits after decrease, continue ...")
                            continue

                    incr_id = random.choice(incr_ids)

                    # Perform the swap
                    new_decr_bw = get_next_bitwidth(
                        offspring, target_bits, grouped_layer_names, available_bitwidths,
                        model, group_id, decr_id, 'decrease'
                    )
                    offspring[group_id][decr_id] = new_decr_bw

                    new_incr_bw = get_next_bitwidth(
                        offspring, target_bits, grouped_layer_names, available_bitwidths,
                        model, group_id, incr_id, 'increase'
                    )
                    offspring[group_id][incr_id] = new_incr_bw

                    successful_mutations += 1

                if successful_mutations == 0:
                    continue  # Skip this offspring if no mutations succeeded

            if offspring in offspring_list or offspring == parent:  # Avoid duplicates
                print(f"Duplicate offspring found for the {duplicate_offspring_ct} time, continue ...")
                duplicate_offspring_ct += 1

                if duplicate_offspring_ct > 10:
                    print(f"Too many duplicates, stopping early with {len(offspring_list)} offspring.")
                    break

                continue
            else:
                duplicate_offspring_ct = 0

            offspring_list.append(offspring)

        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                if parent not in offspring_list:  # Elitist EA
                    offspring_list.append(parent)
            offspring_list, train_fitnesses = selection(
                model=model,
                grouped_layer_names=grouped_layer_names,
                quant_weights_path=args.quant_weights_path,
                available_bitwidths=available_bitwidths,
                candidates=offspring_list,
                num_survive=num_survive,
                calibration_data=calibration_data,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
                target_logits=target_logits,
            )
        # In the end we have lists with a single element (only 1 survivor in last selection step)
        train_fitness = train_fitnesses[0]
        parent = offspring_list[0]
        parent_bits = calculate_total_bits(parent, grouped_layer_names, model)
        print(f"Train fitnesses: {train_fitness:.2e}")
        log_dict["train_fitness"] = train_fitness
    # Save final configuration
    configuration_name = f"evo-{args.fitness_fn}-configuration-{args.target_bitwidth}.txt"
    with open(os.path.join(args.quant_weights_path, configuration_name), "w") as f:
        for i in range(num_groups):
            lines = []
            for layer_name, bitwidth in zip(grouped_layer_names[i], parent[i]):
                # Find the filename for this bitwidth
                filename = None
                for bw, fn in available_bitwidths[layer_name]:
                    if abs(bw - bitwidth) < 1e-6:
                        filename = fn
                        break
                lines.append(f"{layer_name}: {bitwidth} ({filename})")
            f.write("\n".join(lines))
            if i != num_groups - 1:
                f.write("\n")
    # Log final configuration
    print("Final configuration:")
    for group in parent:
        print(group)
    # Final evaluation
    for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
        ppl_eval = compute_perplexity(model, eval_dataset)
        print(f"{eval_dataset_name}: {ppl_eval:.2f}")
        log_dict[f"ppl_eval/{eval_dataset_name}"] = ppl_eval
    ppl_train = compute_perplexity(model, calibration_data)
    print(f"ppl_train: {ppl_train:.2f}")
    log_dict["ppl_train"] = ppl_train
    if args.log_wandb:
        wandb.log(log_dict)


if __name__ == "__main__":
    main()