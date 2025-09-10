#!/usr/bin/env python3
"""
Convert HuggingFace model layer configuration to GGUF format.
Updated version with support for Mixture of Experts (MoE) models.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict


def convert_hf_to_gguf_config(hf_config: str, missing_value: str = "32", is_moe: bool = False) -> Dict[str, str]:
    """
    Convert HuggingFace model config to GGUF format.
    Supports both dense and MoE architectures.

    Args:
        hf_config: String containing HuggingFace config
        missing_value: Value to use for missing layers
        is_moe: Whether this is a Mixture of Experts model

    Returns:
        Dictionary with GGUF format configuration
    """

    hf_dict = {}

    # Parse input - everything after colon is preserved as-is
    for line in hf_config.strip().split('\n'):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if ':' in line:
            key, value = line.split(':', 1)
            hf_dict[key.strip()] = value.strip()

    gguf_config = {}
    layer_info = {}  # Store layer information

    # Mapping from HuggingFace to GGUF layer names
    if is_moe:
        # MoE model mapping
        layer_mapping = {
            # Attention layers (same as dense)
            'self_attn.k_proj': 'attn_k.weight',
            'self_attn.q_proj': 'attn_q.weight',
            'self_attn.v_proj': 'attn_v.weight',
            'self_attn.o_proj': 'attn_output.weight',

            # MoE-specific FFN layers
            'mlp.experts.down_proj': 'ffn_down_exps.weight',
            'mlp.experts.gate_proj': 'ffn_gate_exps.weight',
            'mlp.experts.up_proj': 'ffn_up_exps.weight',
            'mlp.gate': 'ffn_gate_inp.weight',

            # Additional normalization layers for advanced models
            'self_attn.k_norm': 'attn_k_norm.weight',
            'self_attn.q_norm': 'attn_q_norm.weight',
        }
    else:
        # Dense model mapping
        layer_mapping = {
            'mlp.down_proj': 'ffn_down.weight',
            'mlp.gate_proj': 'ffn_gate.weight',
            'mlp.up_proj': 'ffn_up.weight',
            'self_attn.k_proj': 'attn_k.weight',
            'self_attn.q_proj': 'attn_q.weight',
            'self_attn.v_proj': 'attn_v.weight',
            'self_attn.o_proj': 'attn_output.weight'
        }

    # First pass: collect all layer information
    for hf_key, value in hf_dict.items():
        if 'model.layers.' in hf_key:
            # Extract layer number
            parts = hf_key.split('.')
            if len(parts) < 4:
                continue

            try:
                layer_num = int(parts[2])  # model.layers.{X}.component
            except ValueError:
                continue

            # Extract the component (e.g., 'mlp.down_proj', 'self_attn.q_proj')
            component = '.'.join(parts[3:])

            if layer_num not in layer_info:
                layer_info[layer_num] = {}

            layer_info[layer_num][component] = value

    # Second pass: generate GGUF config
    for layer_num in sorted(layer_info.keys()):
        layer_data = layer_info[layer_num]
        base_blk = f"blk.{layer_num}"

        # Set weights from HF config
        for hf_component, gguf_component in layer_mapping.items():
            if hf_component in layer_data:
                gguf_config[f"{base_blk}.{gguf_component}"] = layer_data[hf_component]

        # For layers that weren't explicitly set, use missing_value
        for hf_component, gguf_component in layer_mapping.items():
            if hf_component not in layer_data:
                gguf_config[f"{base_blk}.{gguf_component}"] = missing_value

        # Standard normalization layers (always present)
        gguf_config[f"{base_blk}.attn_norm.weight"] = missing_value
        gguf_config[f"{base_blk}.ffn_norm.weight"] = missing_value

        # Additional normalization for advanced models (if not already set)
        if is_moe:
            if f"{base_blk}.attn_k_norm.weight" not in gguf_config:
                gguf_config[f"{base_blk}.attn_k_norm.weight"] = missing_value
            if f"{base_blk}.attn_q_norm.weight" not in gguf_config:
                gguf_config[f"{base_blk}.attn_q_norm.weight"] = missing_value

    # Handle non-layer weights (embeddings, output, etc.)
    for hf_key, value in hf_dict.items():
        if not 'model.layers.' in hf_key:
            # Convert common non-layer weight names
            if 'embed_tokens' in hf_key:
                gguf_key = 'token_embd.weight'
            elif 'lm_head' in hf_key:
                gguf_key = 'output.weight'
            elif 'model.norm' in hf_key:
                gguf_key = 'output_norm.weight'
            else:
                # Keep original key for unknown mappings
                gguf_key = hf_key

            gguf_config[gguf_key] = value

    return gguf_config


def read_config_file(filepath: str) -> str:
    """Read configuration from a file."""
    with open(filepath, 'r') as f:
        return f.read()


def write_config_file(config_dict: Dict[str, str], filepath: str) -> None:
    """Write configuration to a file."""
    with open(filepath, 'w') as f:
        for key in sorted(config_dict.keys()):
            f.write(f"{key}: {config_dict[key]}\n")


def detect_moe_model(hf_config: str) -> bool:
    """
    Detect if the model is MoE based on layer names in config.

    Args:
        hf_config: String containing HuggingFace config

    Returns:
        True if MoE model detected, False otherwise
    """
    moe_indicators = [
        'experts',
        'mlp.gate.',
        'router',
        'shared_expert'
    ]

    for line in hf_config.strip().split('\n'):
        line = line.strip().lower()
        if any(indicator in line for indicator in moe_indicators):
            return True

    return False


def main():
    """Main function with command-line argument parsing."""

    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model layer configuration to GGUF format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_config.py hf_config.txt
  python convert_config.py hf_config.txt -o gguf_config.txt
  python convert_config.py hf_config.txt --missing-value "16 (16-F16.pth)"
  python convert_config.py hf_config.txt --moe  # Force MoE mode
  python convert_config.py hf_config.txt --dense  # Force dense mode
        """
    )

    parser.add_argument(
        'input_file',
        type=str,
        help='Path to the HuggingFace configuration file'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output file path (default: print to stdout)'
    )

    parser.add_argument(
        '--missing-value',
        type=str,
        default="32 (32-F32.pth)",
        help='Value to use for missing layers (default: "32 (32-F32.pth)")'
    )

    parser.add_argument(
        '--moe',
        action='store_true',
        help='Force MoE model mode'
    )

    parser.add_argument(
        '--dense',
        action='store_true',
        help='Force dense model mode'
    )

    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    if not input_path.is_file():
        print(f"Error: '{args.input_file}' is not a file.", file=sys.stderr)
        sys.exit(1)

    try:
        # Read configuration
        hf_config = read_config_file(args.input_file)

        # Determine model type
        if args.moe and args.dense:
            print("Error: Cannot specify both --moe and --dense flags.", file=sys.stderr)
            sys.exit(1)
        elif args.moe:
            is_moe = True
        elif args.dense:
            is_moe = False
        else:
            # Auto-detect
            is_moe = detect_moe_model(hf_config)

        if args.verbose:
            print(f"Reading HuggingFace config from: {args.input_file}")
            print(f"Model type detected: {'MoE' if is_moe else 'Dense'}")
            print(f"Using '{args.missing_value}' for missing values")

        # Convert configuration
        gguf_config = convert_hf_to_gguf_config(hf_config, args.missing_value, is_moe)

        if args.verbose:
            print(f"Converted {len(gguf_config)} layer configurations")

        # Output results
        if args.output:
            write_config_file(gguf_config, args.output)
            if args.verbose:
                print(f"GGUF config written to: {args.output}")
        else:
            # Print to stdout
            for key in sorted(gguf_config.keys()):
                print(f"{key}: {gguf_config[key]}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()