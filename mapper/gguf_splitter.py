#!/usr/bin/env python3
"""
GGUF Splitter - Bitwidth Layout with Layer Mapping
Splits a GGUF model into directories per layer with bitwidth-specific files.
Also supports loading HuggingFace models from GGUF and mapping them to GGUF layer names.
"""

import os
import json
import time
import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Union, Optional

try:
    import gguf
    import logging
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # print version of transformers
    print(f"Using transformers version: {transformers.__version__}")

    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"Error: Required library not found. Install with: pip install gguf torch transformers")
    print(f"Specific error: {e}")
    exit(1)


class GGUFSplitter:
    def __init__(self, model_path: str, output_dir: str, use_exact_bitwidth: bool = False):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.use_exact_bitwidth = use_exact_bitwidth
        self.gguf_layer_database = {}  # Store GGUF layer mappings

    def get_quantization_info(self, tensor_name: str, tensor_type: int) -> str:
        """Convert tensor type ID to quantization string"""
        type_map = {
            0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
            8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
            13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
            18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S", 22: "IQ2_S",
            23: "IQ4_XS", 24: "IQ2_M", 25: "IQ3_M"
        }
        return type_map.get(tensor_type, f"UNKNOWN_{tensor_type}")

    def get_tensor_bit_width(self, quantization: str) -> float:
        """Get the exact bit width for a given quantization type."""
        # Mapping of GGUF tensor types to their exact bit widths
        bit_width_map = {
            # Float types
            'F32': 32.0,
            'F16': 16.0,
            'BF16': 16.0,

            # Integer types
            'I8': 8.0,
            'I16': 16.0,
            'I32': 32.0,
            'I64': 64.0,

            # Quantized types with exact fractional bitwidths
            'Q4_0': 4.5,  # 4 bits + 0.5 for scale
            'Q4_1': 5.0,  # 4 bits + 1 for scale/zero
            'Q5_0': 5.5,  # 5 bits + 0.5 for scale
            'Q5_1': 6.0,  # 5 bits + 1 for scale/zero
            'Q8_0': 8.5,  # 8 bits + 0.5 for scale
            'Q8_1': 9.0,  # 8 bits + 1 for scale/zero
            'Q2_K': 2.5625,  # Variable, approximate
            'Q3_K': 3.4375,  # Variable, approximate
            'Q4_K': 4.5,  # Variable, approximate
            'Q5_K': 5.5,  # Variable, approximate
            'Q6_K': 6.5625,  # Variable, approximate
            'Q8_K': 8.5,  # Variable, approximate
            'IQ2_XXS': 2.0625,
            'IQ2_XS': 2.3125,
            'IQ2_S': 2.5,
            'IQ2_M': 2.7,
            'IQ3_XXS': 3.0625,
            'IQ3_S': 3.44,
            'IQ3_M': 3.66,
            'IQ4_NL': 4.56,
            'IQ4_XS': 4.25,
            'IQ1_S': 1.5625,
            'IQ1_M': 1.75,
        }

        return bit_width_map.get(quantization, 32.0)  # Default to 32 bits if unknown

    def extract_bitwidth_from_quantization(self, quantization: str) -> Union[int, float]:
        """Extract bitwidth from quantization string"""
        if self.use_exact_bitwidth:
            return self.get_tensor_bit_width(quantization)
        else:
            # Original integer logic for backward compatibility
            if quantization.startswith("Q2") or quantization.startswith("IQ2"):
                return 2
            elif quantization.startswith("Q3") or quantization.startswith("IQ3"):
                return 3
            elif quantization.startswith("Q4") or quantization.startswith("IQ4"):
                return 4
            elif quantization.startswith("Q5") or quantization.startswith("IQ5"):
                return 5
            elif quantization.startswith("Q6") or quantization.startswith("IQ6"):
                return 6
            elif quantization.startswith("Q8") or quantization.startswith("IQ8"):
                return 8
            elif quantization in ["F16", "F32"]:
                return 16 if quantization == "F16" else 32
            elif quantization.startswith("IQ1"):
                return 1
            else:
                return 4

    def build_gguf_layer_database(self) -> Dict[str, Dict]:
        """Build a database of GGUF layers for mapping purposes"""
        print(f"Building GGUF layer database from: {self.model_path}")

        reader = gguf.GGUFReader(str(self.model_path))
        database = {}

        for tensor in reader.tensors:
            quantization = self.get_quantization_info(tensor.name, tensor.tensor_type.value)
            bitwidth = self.extract_bitwidth_from_quantization(quantization)
            exact_bitwidth = self.get_tensor_bit_width(quantization)

            # Store tensor information in database
            database[tensor.name] = {
                "tensor_type": tensor.tensor_type.value,
                "quantization": quantization,
                "bitwidth": bitwidth,
                "exact_bitwidth": exact_bitwidth,
                "shape": tensor.shape.tolist(),
                "n_elements": tensor.n_elements,
                "n_bytes": tensor.n_bytes,
                "data_offset": tensor.data_offset
            }

        self.gguf_layer_database = database
        print(f"Built database with {len(database)} GGUF tensors")
        return database

    def map_hf_to_gguf_name(self, hf_name: str) -> Optional[str]:
        """Map HuggingFace parameter name to corresponding GGUF tensor name for MoE models"""

        # Layer mapping from HF to GGUF (updated for MoE)
        layer_mapping = {
            # Regular attention layers
            'self_attn.k_proj': 'attn_k',
            'self_attn.q_proj': 'attn_q',
            'self_attn.v_proj': 'attn_v',
            'self_attn.o_proj': 'attn_output',
            'input_layernorm': 'attn_norm',
            'post_attention_layernorm': 'ffn_norm',

            # Regular FFN layers (for non-MoE models)
            'mlp.gate_proj': 'ffn_gate',
            'mlp.up_proj': 'ffn_up',
            'mlp.down_proj': 'ffn_down',

            # MoE expert mappings - individual experts get consolidated
            'mlp.experts.*.down_proj': 'ffn_down_exps',
            'mlp.experts.*.gate_proj': 'ffn_gate_exps',
            'mlp.experts.*.up_proj': 'ffn_up_exps',

            # Shared expert mappings (if present)
            'mlp.shared_expert.down_proj': 'ffn_down_shexp',
            'mlp.shared_expert.gate_proj': 'ffn_gate_shexp',
            'mlp.shared_expert.up_proj': 'ffn_up_shexp',

            # Gating/routing network
            'mlp.gate': 'ffn_gate_inp',
            'mlp.shared_expert_gate': 'ffn_gate_inp_shexp',
        }

        # Handle layer weights (model.layers.X.component.weight)
        if 'model.layers.' in hf_name:
            parts = hf_name.split('.')
            if len(parts) >= 4:
                try:
                    layer_num = int(parts[2])  # model.layers.{X}.component
                except ValueError:
                    return None

                # Extract component (e.g., 'mlp.experts.0.down_proj', 'self_attn.q_proj')
                component = '.'.join(parts[3:])

                # Remove .weight suffix if present
                if component.endswith('.weight'):
                    component = component[:-7]

                # Handle MoE expert weights specifically
                if 'mlp.experts.' in component:
                    # Extract expert number and projection type
                    # e.g., 'mlp.experts.0.down_proj' -> 'mlp.experts.*.down_proj'
                    expert_parts = component.split('.')
                    if len(expert_parts) >= 4 and expert_parts[0] == 'mlp' and expert_parts[1] == 'experts':
                        try:
                            expert_num = int(expert_parts[2])
                            proj_type = expert_parts[3]

                            # Create pattern for mapping
                            expert_pattern = f"mlp.experts.*.{proj_type}"

                            if expert_pattern in layer_mapping:
                                gguf_component = layer_mapping[expert_pattern]
                                candidate = f"blk.{layer_num}.{gguf_component}.weight"
                                if candidate in self.gguf_layer_database:
                                    return candidate

                                # Try without .weight suffix
                                candidate_no_weight = f"blk.{layer_num}.{gguf_component}"
                                if candidate_no_weight in self.gguf_layer_database:
                                    return candidate_no_weight

                        except (ValueError, IndexError):
                            pass

                # Handle regular (non-expert) components
                elif component in layer_mapping:
                    gguf_component = layer_mapping[component]
                    candidate = f"blk.{layer_num}.{gguf_component}.weight"
                    if candidate in self.gguf_layer_database:
                        return candidate

                    # Try without .weight suffix
                    candidate_no_weight = f"blk.{layer_num}.{gguf_component}"
                    if candidate_no_weight in self.gguf_layer_database:
                        return candidate_no_weight

        # Handle non-layer weights
        else:
            # Remove .weight suffix for processing
            base_name = hf_name
            has_weight_suffix = base_name.endswith('.weight')
            if has_weight_suffix:
                base_name = base_name[:-7]

            # Map common non-layer weights
            candidate = None
            if 'embed_tokens' in base_name:
                candidate = 'token_embd.weight'
            elif 'lm_head' in base_name:
                candidate = 'output.weight'
            elif base_name == 'model.norm':
                candidate = 'output_norm.weight'
            else:
                # Try original name first
                candidate = hf_name

            # Check if candidate exists in database
            if candidate and candidate in self.gguf_layer_database:
                return candidate

            # Try with model. prefix if not already present
            if candidate and not candidate.startswith('model.'):
                candidate_with_model = f"model.{candidate}"
                if candidate_with_model in self.gguf_layer_database:
                    return candidate_with_model

            # Fallback: try without .weight suffix
            if has_weight_suffix:
                candidate_no_weight = base_name
                if candidate_no_weight in self.gguf_layer_database:
                    return candidate_no_weight

                # Try with model. prefix and no weight
                if not candidate_no_weight.startswith('model.'):
                    candidate_model_no_weight = f"model.{candidate_no_weight}"
                    if candidate_model_no_weight in self.gguf_layer_database:
                        return candidate_model_no_weight

        # Final fallback: try direct exact match
        if hf_name in self.gguf_layer_database:
            return hf_name

        return None

    def save_layer_mapping(self, mapping: Dict[str, str], output_dir: Path):
        """Save the HF to GGUF layer mapping to a file"""
        mapping_file = output_dir / "hf_to_gguf_mapping.json"
        with open(mapping_file, 'w') as f:
            json.dump(mapping, f, indent=2)
        print(f"Layer mapping saved to: {mapping_file}")

    def split_gguf_model(self, overwrite_bitwidth: Union[int, float, str]):
        """Split the GGUF model into layer directories with bitwidth-specific files"""
        print(f"Splitting GGUF model: {self.model_path}")
        if self.use_exact_bitwidth:
            print("Using exact fractional bitwidths")

        # Build the database first
        self.build_gguf_layer_database()

        # Create gguf subdirectory
        gguf_output_dir = self.output_dir
        gguf_output_dir.mkdir(exist_ok=True)

        # Load the GGUF model using GGUFReader
        reader = gguf.GGUFReader(str(self.model_path))

        # Get metadata
        metadata = {}
        for field_name, field in reader.fields.items():
            # Convert field data to serializable format
            try:
                # Use the field's contents() method to get the proper value
                value = field.contents()
                metadata[field_name] = {
                    "types": [t.value for t in field.types] if field.types else [],
                    "value": value
                }
            except Exception as e:
                # Fallback for complex fields that can't be easily serialized
                logger.warning(f"Could not serialize field {field_name}: {e}")
                metadata[field_name] = {
                    "types": [t.value for t in field.types] if field.types else [],
                    "value": f"<serialization_error: {str(e)}>"
                }

        # Initialize manifest
        manifest = {
            "model_info": {
                "original_file": str(self.model_path.name),
                "total_tensors": len(reader.tensors),
                "split_timestamp": None,
                "use_exact_bitwidth": self.use_exact_bitwidth
            },
            "metadata": metadata,
            "layers": {}
        }

        processed_count = 0

        # Process each tensor
        for tensor in reader.tensors:
            if self.use_exact_bitwidth and overwrite_bitwidth is not None:
                try:
                    quantization = self.get_quantization_info(tensor.name, tensor.tensor_type.value)
                except ValueError:
                    quantization = overwrite_bitwidth
            else:
                quantization = self.get_quantization_info(tensor.name, tensor.tensor_type.value)

            processed_count += 1
            print(f"Processing tensor {processed_count}: {tensor.name} ({quantization})")

            # Create layer directory
            layer_name = tensor.name
            layer_dir = gguf_output_dir / layer_name
            layer_dir.mkdir(parents=True, exist_ok=True)

            # Determine bitwidth and filename
            bitwidth = self.extract_bitwidth_from_quantization(quantization)

            # Format filename based on whether bitwidth is integer or float
            if isinstance(bitwidth, float) and bitwidth != int(bitwidth):
                bitwidth_prefix = f"{bitwidth}"
            else:
                bitwidth_prefix = f"{int(bitwidth)}"

            if self.use_exact_bitwidth:
                bitwidth_prefix = f"{bitwidth_prefix}-{quantization}"

            filename = f"{bitwidth_prefix}.pth"
            filepath = layer_dir / filename

            # Get tensor data (already available in tensor.data)
            tensor_data = tensor.data

            # Save tensor data as binary
            with open(filepath, 'wb') as tensor_file:
                tensor_file.write(tensor_data.tobytes())

            # Calculate tensor size
            tensor_size = tensor.n_bytes

            # Create individual tensor metadata
            tensor_metadata = {
                "tensor_info": {
                    "name": tensor.name,
                    "type": tensor.tensor_type.value,
                    "quantization": quantization,
                    "bitwidth": bitwidth,
                    "exact_bitwidth": self.get_tensor_bit_width(quantization),
                    "shape": tensor.shape.tolist(),
                    "n_elements": tensor.n_elements,
                    "n_bytes": tensor.n_bytes,
                    "data_offset_original": tensor.data_offset,
                    "data_filename": filename,
                    "np_dtype": str(tensor_data.dtype),
                    "np_shape": list(tensor_data.shape)
                }
            }

            metadata_filename = f"{bitwidth_prefix}-metadata.json"
            metadata_filepath = layer_dir / metadata_filename
            with open(metadata_filepath, 'w') as meta_file:
                json.dump(tensor_metadata, meta_file, indent=2)

            # Update manifest
            if layer_name not in manifest["layers"]:
                manifest["layers"][layer_name] = {
                    "original_name": layer_name,
                    "dims": tensor.shape.tolist(),
                    "bitwidths": {}
                }

            bitwidth_key = str(bitwidth)
            manifest["layers"][layer_name]["bitwidths"][bitwidth_key] = {
                "filename": filename,
                "metadata_filename": metadata_filename,
                "type": tensor.tensor_type.value,
                "quantization": quantization,
                "bitwidth": bitwidth,
                "exact_bitwidth": self.get_tensor_bit_width(quantization),
                "size_bytes": tensor_size,
                "shape": tensor.shape.tolist(),
                "n_elements": tensor.n_elements,
                "data_offset": tensor.data_offset
            }

        # Finalize manifest
        manifest["model_info"]["split_timestamp"] = time.time()
        manifest["model_info"]["processed_tensors"] = processed_count

        # Save manifest
        manifest_path = gguf_output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)

        # Save GGUF layer database
        db_path = gguf_output_dir / "gguf_layer_database.json"
        with open(db_path, 'w') as f:
            json.dump(self.gguf_layer_database, f, indent=2)

        print(
            f"GGUF split complete! {processed_count} tensors processed into {len(manifest['layers'])} layer directories")
        print(f"Output directory: {gguf_output_dir}")
        print(f"Manifest saved to {manifest_path}")
        print(f"GGUF layer database saved to {db_path}")

    def split_hf_model(self, dtype: str = "float16",
                       overwrite_bitwidth: Union[int, float, str] = None):
        """Load HuggingFace model from GGUF and split into layers with GGUF naming"""
        print(f"Loading HuggingFace model from GGUF: {self.model_path}")

        # Build GGUF layer database first if not already built
        if not self.gguf_layer_database:
            self.build_gguf_layer_database()

        # Create hf subdirectory
        hf_output_dir = self.output_dir
        hf_output_dir.mkdir(exist_ok=True)

        # Set torch dtype
        torch_dtype = torch.float16 if dtype == "float16" else torch.float32

        # Get the directory containing the GGUF file
        input_dir = self.model_path.parent
        gguf_file = self.model_path.name

        print(f"Loading model from {input_dir}/{gguf_file} with dtype={torch_dtype}...")
        model = AutoModelForCausalLM.from_pretrained(
            input_dir,
            gguf_file=gguf_file,
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )

        print(f"Loading tokenizer from {input_dir}/{gguf_file}...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                input_dir,
                gguf_file=gguf_file,
                use_fast=True
            )
        except Exception as e:
            print(f"Warning: Could not load tokenizer: {e}")
            tokenizer = None

        def process_layer(name, layer: torch.nn.Module) -> bool:
            layer_regex = re.compile(
                r'^model\.layers\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.weight$')
            return re.search(layer_regex, name)

        # Initialize manifest
        hf_manifest = {
            "model_info": {
                "original_file": str(self.model_path.name),
                "dtype": dtype,
                "bitwidth": overwrite_bitwidth,
                "use_exact_bitwidth": self.use_exact_bitwidth,
                "split_timestamp": time.time()
            },
            "layers": {},
            "mapping_stats": {
                "total_layers": 0,
                "mapped_layers": 0,
                "unmapped_layers": 0
            }
        }

        processed_count = 0
        mapping = {}  # Store HF to GGUF name mappings
        mapped_count = 0

        # Process model parameters
        for name, layer in model.named_parameters():
            if process_layer(name, layer):
                processed_count += 1

                gguf_name = self.map_hf_to_gguf_name(name)
                bitwidth = None
                quantization = None

                if gguf_name:
                    mapped_count += 1
                    print(f"Processing HF layer {processed_count}: {name} -> {gguf_name}")
                    mapping[name] = gguf_name

                    gguf_bitwidth = self.gguf_layer_database[gguf_name]["bitwidth"]
                    gguf_quantization = self.gguf_layer_database[gguf_name]["quantization"]

                    if overwrite_bitwidth is not None:
                        try:
                            bitwidth = float(overwrite_bitwidth)
                            quantization = None
                        except ValueError:
                            quantization = overwrite_bitwidth  # is a string like "Q4_K"
                            bitwidth = self.extract_bitwidth_from_quantization(overwrite_bitwidth.upper())

                        if gguf_bitwidth != bitwidth and bitwidth > 0:
                            print(f"Warning: Overwrite bitwidth {overwrite_bitwidth} does not match GGUF bitwidth {gguf_bitwidth} for layer {name}, skipping saving layer")
                            continue

                    else:
                        bitwidth = gguf_bitwidth
                        quantization = gguf_quantization
                else:
                    print(f"Processing HF layer {processed_count}: {name} (no GGUF mapping found)")
                    mapping[name] = None

                    if overwrite_bitwidth is None:
                        print(self.gguf_layer_database)
                        raise ValueError(f"No GGUF mapping found for layer {name} and no overwrite_bitwidth provided")

                layer_dir_name = name.replace('.weight', '')

                # Create layer directory
                layer_dir = hf_output_dir / layer_dir_name
                layer_dir.mkdir(parents=True, exist_ok=True)

                # Save the parameter with the specified bitwidth filename
                if isinstance(bitwidth, float) and bitwidth != int(bitwidth):
                    bitwidth_file_name_prefix = str(bitwidth)
                else:
                    bitwidth_file_name_prefix = str(int(bitwidth))

                if quantization is not None:
                    bitwidth_file_name_prefix = f"{bitwidth_file_name_prefix}-{quantization}"

                filename = f"{bitwidth_file_name_prefix}.pth"
                filepath = layer_dir / filename

                # Save tensor data
                torch.save(layer.data, filepath)

                # Create metadata
                tensor_metadata = {
                    "tensor_info": {
                        "name": name,
                        "gguf_mapped_name": gguf_name,
                        "bitwidth": bitwidth,
                        "dtype": str(layer.dtype),
                        "shape": list(layer.shape),
                        "n_elements": layer.numel(),
                        "n_bytes": layer.numel() * layer.element_size(),
                        "data_filename": filename,
                        "requires_grad": layer.requires_grad
                    }
                }

                # Add GGUF metadata if mapping found
                if gguf_name and gguf_name in self.gguf_layer_database:
                    tensor_metadata["gguf_info"] = self.gguf_layer_database[gguf_name]

                # Save metadata
                metadata_filename = f"{bitwidth_file_name_prefix}-metadata.json"
                metadata_filepath = layer_dir / metadata_filename
                with open(metadata_filepath, 'w') as meta_file:
                    json.dump(tensor_metadata, meta_file, indent=2)

                # Update manifest
                hf_manifest["layers"][name] = {
                    "original_name": name,
                    "gguf_mapped_name": gguf_name,
                    "layer_directory": layer_dir_name,
                    "dims": list(layer.shape),
                    "bitwidth": bitwidth,
                    "filename": filename,
                    "metadata_filename": metadata_filename,
                    "dtype": str(layer.dtype),
                    "size_bytes": layer.numel() * layer.element_size(),
                    "shape": list(layer.shape),
                    "n_elements": layer.numel()
                }

        # Update mapping stats
        hf_manifest["mapping_stats"]["total_layers"] = processed_count
        hf_manifest["mapping_stats"]["mapped_layers"] = mapped_count
        hf_manifest["mapping_stats"]["unmapped_layers"] = processed_count - mapped_count

        # Save manifest
        manifest_path = hf_output_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(hf_manifest, f, indent=2)

        # Save layer mapping
        self.save_layer_mapping(mapping, hf_output_dir)

        print(f"HuggingFace split complete! {processed_count} layers processed")
        print(f"Mapped {mapped_count}/{processed_count} layers to GGUF names")
        print(f"Output directory: {hf_output_dir}")
        print(f"Manifest saved to {manifest_path}")

        # Clean up model from memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def list_available_bitwidths(self):
        """List all available bitwidths in the model"""
        print(f"Analyzing model: {self.model_path}")

        reader = gguf.GGUFReader(str(self.model_path))

        bitwidths = set()
        exact_bitwidths = set()
        quantization_counts = {}

        for tensor in reader.tensors:
            quantization = self.get_quantization_info(tensor.name, tensor.tensor_type.value)
            bitwidth = self.extract_bitwidth_from_quantization(quantization)
            exact_bitwidth = self.get_tensor_bit_width(quantization)

            bitwidths.add(bitwidth)
            exact_bitwidths.add(exact_bitwidth)

            # Count quantization types
            if quantization not in quantization_counts:
                quantization_counts[quantization] = 0
            quantization_counts[quantization] += 1

        print(f"Available integer bitwidths: {sorted(bitwidths)}")
        print(f"Available exact bitwidths: {sorted(exact_bitwidths)}")

        print("\nQuantization distribution:")
        for quant, count in sorted(quantization_counts.items()):
            exact_bw = self.get_tensor_bit_width(quant)
            print(f"  {quant}: {count} tensors (exact bitwidth: {exact_bw})")

        print(f"\nTotal tensors: {len(reader.tensors)}")

        return sorted(bitwidths), sorted(exact_bitwidths)

    def inspect_model(self):
        """Inspect model structure and metadata"""
        print(f"Inspecting model: {self.model_path}")

        reader = gguf.GGUFReader(str(self.model_path))

        print(f"\nModel metadata ({len(reader.fields)} fields):")
        for field_name, field in reader.fields.items():
            value = field.data
            if hasattr(value, 'shape'):  # numpy array
                print(f"  {field_name}: {type(value).__name__} {value.shape}")
            elif isinstance(value, (list, tuple)) and len(value) > 10:
                print(f"  {field_name}: {type(value).__name__} (length: {len(value)})")
            else:
                print(f"  {field_name}: {value}")

        print(f"\nTensors ({len(reader.tensors)}):")
        layer_types = {}
        for tensor in reader.tensors:
            quantization = self.get_quantization_info(tensor.name, tensor.tensor_type.value)
            layer_type = tensor.name.split('.')[0] if '.' in tensor.name else 'other'

            if layer_type not in layer_types:
                layer_types[layer_type] = {}
            if quantization not in layer_types[layer_type]:
                layer_types[layer_type][quantization] = 0
            layer_types[layer_type][quantization] += 1

        for layer_type, quants in sorted(layer_types.items()):
            print(f"  {layer_type}:")
            for quant, count in sorted(quants.items()):
                exact_bw = self.get_tensor_bit_width(quant)
                print(f"    {quant}: {count} (exact: {exact_bw} bits)")


def main():
    parser = argparse.ArgumentParser(description="Split GGUF model into bitwidth-organized layers")
    parser.add_argument("model_path", help="Path to input GGUF model")
    parser.add_argument("output_dir", nargs='?', help="Directory to store split layers")
    parser.add_argument("--list-bitwidths", action="store_true",
                        help="List available bitwidths in the model and exit")
    parser.add_argument("--inspect", action="store_true",
                        help="Inspect model structure and metadata")
    parser.add_argument("--gguf-layers", action="store_true",
                        help="Split GGUF model into layers (stored in 'gguf' subdirectory)")
    parser.add_argument("--hf-layers", action="store_true",
                        help="Load HuggingFace model from GGUF and split into layers (stored in 'hf' subdirectory)")
    parser.add_argument("--bitwidth", type=str, default=16,
                        help="Bitwidth for HuggingFace model layers (can be float, e.g., 4.5 or k-quant types (e.g. Q4_K))")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16",
                        help="Data type for HuggingFace model loading (default: float16)")
    parser.add_argument("--exact", action="store_true",
                        help="Use exact fractional bitwidths instead of rounded integers (e.g., 4.5 instead of 4)")
    parser.add_argument("--both", action="store_true",
                        help="Split both GGUF and HF layers (recommended for layer mapping)")

    args = parser.parse_args()

    if not args.output_dir and not args.list_bitwidths and not args.inspect:
        parser.error("output_dir is required unless using --list-bitwidths or --inspect")

    splitter = GGUFSplitter(args.model_path, args.output_dir or ".", args.exact)

    if args.list_bitwidths:
        splitter.list_available_bitwidths()
    elif args.inspect:
        splitter.inspect_model()
    else:
        # Default behavior: split GGUF model
        if not args.hf_layers and not args.both:
            args.gguf_layers = True

        if args.both:
            args.gguf_layers = True
            args.hf_layers = True

        if args.gguf_layers:
            splitter.split_gguf_model(args.bitwidth)

        if args.hf_layers:
            overwrite_bitwidth = None
            try:
                if float(args.bitwidth) <= 0:
                    overwrite_bitwidth = 0
            except ValueError:
                overwrite_bitwidth = None
            splitter.split_hf_model(args.dtype, overwrite_bitwidth)


if __name__ == "__main__":
    main()