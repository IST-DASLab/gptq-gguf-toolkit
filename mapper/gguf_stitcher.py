#!/usr/bin/env python3
"""
GGUF Stitcher - Reconstruct Model from Split Layers
Reconstructs a GGUF model from split layer directories with specified bitwidths per tensor.
Automatically discovers all layers and uses default bitwidth for unspecified tensors.
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import re

try:
    import gguf
    from gguf import GGUFWriter, GGUFReader, GGMLQuantizationType, GGUFValueType
    import logging

    logger = logging.getLogger(__name__)
except ImportError:
    print("Error: gguf library not found. Install with: pip install gguf")
    exit(1)


class QuantizationConfig:
    """Represents a quantization configuration for a tensor"""

    def __init__(self, bitwidth: float, filename: str, quant_type: Optional[str] = None,
                 metadata: Dict[str, Any] = None):
        if bitwidth == int(bitwidth):
            effective_bitwidth = int(bitwidth)
        else:
            effective_bitwidth = bitwidth

        self.bitwidth = bitwidth
        self.filename = filename
        self.filename_prefix = f"{effective_bitwidth}-{quant_type}" if quant_type else str(effective_bitwidth)
        self.meta_data = metadata or {}
        self.quant_type = quant_type


class GGUFStitcher:
    def __init__(self, split_dir: str, config_path: Optional[str], output_path: str,
                 original_model_path: Optional[str] = None, default_bitwidth: float = 4.0,
                 default_quant_type: str = "Q4_K"):
        self.split_dir = Path(split_dir)
        self.config_path = Path(config_path) if config_path else None
        self.output_path = Path(output_path)
        self.original_model_path = Path(original_model_path) if original_model_path else None
        self.default_bitwidth = default_bitwidth
        self.default_quant_type = default_quant_type

        # Load manifest from split directory
        self.manifest = self._load_manifest()

        # Discover all available layers
        self.available_layers = self._discover_layers()

        # Load configuration (if provided) and merge with discovered layers
        self.config = self._build_complete_config()

        # Load original model metadata if available
        self.original_metadata = self._load_original_metadata()

        # Initialize GGUF writer
        self.writer = None

    def _discover_layers(self) -> Dict[str, List[Dict[str, Any]]]:
        """Discover all available layers and their bitwidth options from the split directory"""
        discovered = {}

        # First, try from manifest
        if "layers" in self.manifest:
            for layer_name, layer_info in self.manifest["layers"].items():
                layer_dir = self.split_dir / layer_name
                if not layer_dir.exists():
                    continue

                available_configs = []

                # Scan actual files in the directory
                for tensor_file in layer_dir.glob("*.pth"):
                    # Parse filename to extract bitwidth and quant type
                    # Expected format: "bitwidth-quanttype.pth" or "bitwidth.pth"
                    filename = tensor_file.name

                    match = re.match(r'^([\d.]+)-([^.]+)\.pth$', filename)
                    if match:
                        bitwidth_str, quant_type = match.groups()
                        bitwidth = float(bitwidth_str)
                        available_configs.append({
                            'bitwidth': bitwidth,
                            'filename': filename,
                            'quant_type': quant_type
                        })
                    else:
                        match = re.match(r'^([\d.]+)\.pth$', filename)
                        if match:
                            bitwidth = float(match.group(1))
                            available_configs.append({
                                'bitwidth': bitwidth,
                                'filename': filename,
                                'quant_type': None
                            })

                if available_configs:
                    discovered[layer_name] = available_configs

        # Also scan directory directly for any layers not in manifest
        for layer_dir in self.split_dir.iterdir():
            if layer_dir.is_dir() and layer_dir.name not in discovered:
                available_configs = []

                for tensor_file in layer_dir.glob("*.pth"):
                    filename = tensor_file.name
                    match = re.match(r'^([\d.]+)-([^.]+)\.pth$', filename)
                    if match:
                        bitwidth_str, quant_type = match.groups()
                        bitwidth = float(bitwidth_str)
                        available_configs.append({
                            'bitwidth': bitwidth,
                            'filename': filename,
                            'quant_type': quant_type
                        })
                    else:
                        match = re.match(r'^([\d.]+)\.pth$', filename)
                        if match:
                            bitwidth = float(match.group(1))
                            available_configs.append({
                                'bitwidth': bitwidth,
                                'filename': filename,
                                'quant_type': None
                            })

                if available_configs:
                    discovered[layer_dir.name] = available_configs

        print(f"Discovered {len(discovered)} layers in split directory")
        return discovered

    def _find_best_matching_config(self, available_configs: List[Dict[str, Any]],
                                   target_bitwidth: float, target_quant_type: Optional[str]) -> Dict[str, Any]:
        """Find the best matching configuration from available options"""
        # First, try exact match on both bitwidth and quant_type
        if target_quant_type:
            for config in available_configs:
                if config['bitwidth'] == target_bitwidth and config['quant_type'] == target_quant_type:
                    return config

        # Try exact bitwidth match with any quant_type
        for config in available_configs:
            if config['bitwidth'] == target_bitwidth:
                return config

        # Find closest bitwidth
        available_configs_sorted = sorted(available_configs,
                                          key=lambda c: abs(c['bitwidth'] - target_bitwidth))

        # If we have a preferred quant_type, try to match it with the closest bitwidth
        if target_quant_type:
            for config in available_configs_sorted:
                if config['quant_type'] == target_quant_type:
                    return config

        # Return closest match
        return available_configs_sorted[0]

    def _build_complete_config(self) -> Dict[str, QuantizationConfig]:
        """Build complete configuration by combining user config and discovered layers"""
        complete_config = {}

        # First, load user-specified configurations if config file exists
        user_config = {}
        if self.config_path and self.config_path.exists():
            user_config = self._load_config()

        # Process all discovered layers
        for layer_name, available_configs in self.available_layers.items():
            if layer_name in user_config:
                # Use user-specified configuration
                complete_config[layer_name] = user_config[layer_name]
                print(
                    f"  {layer_name}: Using user config - {user_config[layer_name].bitwidth}-bit ({user_config[layer_name].quant_type or 'default'})")
            else:
                # Use default configuration, but find best available match
                best_match = self._find_best_matching_config(
                    available_configs,
                    self.default_bitwidth,
                    self.default_quant_type
                )

                # Load metadata if exists
                metadata = {}
                layer_dir = self.split_dir / layer_name
                if layer_dir.exists():
                    metadata_file = layer_dir / f"{best_match['bitwidth']}-{best_match['quant_type']}-metadata.json"
                    if not metadata_file.exists():
                        metadata_file = layer_dir / f"{best_match['bitwidth']}-metadata.json"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, 'r') as mf:
                                metadata = json.load(mf)
                        except Exception as e:
                            logger.warning(f"Could not load metadata for {layer_name}: {e}")

                complete_config[layer_name] = QuantizationConfig(
                    best_match['bitwidth'],
                    best_match['filename'],
                    best_match['quant_type'],
                    metadata
                )

                if best_match['bitwidth'] != self.default_bitwidth or best_match[
                    'quant_type'] != self.default_quant_type:
                    print(
                        f"  {layer_name}: Using closest available - {best_match['bitwidth']}-bit ({best_match['quant_type'] or 'default'}) [requested: {self.default_bitwidth}-bit {self.default_quant_type}]")
                else:
                    print(
                        f"  {layer_name}: Using default - {best_match['bitwidth']}-bit ({best_match['quant_type'] or 'default'})")

        print(f"\nTotal configuration: {len(complete_config)} tensors")
        print(f"  User-specified: {len(user_config)} tensors")
        print(f"  Using defaults: {len(complete_config) - len(user_config)} tensors")

        return complete_config

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

    def _get_ggml_quantization_type_map(self) -> Dict[str, GGMLQuantizationType]:
        """Map quantization type strings to GGML quantization types"""
        return {
            # Float types
            'F32': GGMLQuantizationType.F32,
            'F16': GGMLQuantizationType.F16,
            'BF16': GGMLQuantizationType.BF16,

            # Integer types
            'I8': GGMLQuantizationType.I8,
            'I16': GGMLQuantizationType.I16,
            'I32': GGMLQuantizationType.I32,
            'I64': GGMLQuantizationType.I64,

            # Standard quantized types
            'Q4_0': GGMLQuantizationType.Q4_0,
            'Q4_1': GGMLQuantizationType.Q4_1,
            'Q5_0': GGMLQuantizationType.Q5_0,
            'Q5_1': GGMLQuantizationType.Q5_1,
            'Q8_0': GGMLQuantizationType.Q8_0,
            'Q8_1': GGMLQuantizationType.Q8_1,

            # K-quantized types
            'Q2_K': GGMLQuantizationType.Q2_K,
            'Q3_K': GGMLQuantizationType.Q3_K,
            'Q4_K': GGMLQuantizationType.Q4_K,
            'Q5_K': GGMLQuantizationType.Q5_K,
            'Q6_K': GGMLQuantizationType.Q6_K,
            'Q8_K': GGMLQuantizationType.Q8_K,

            # IQ (Improved Quantization) types
            'IQ2_XXS': GGMLQuantizationType.IQ2_XXS,
            'IQ2_XS': GGMLQuantizationType.IQ2_XS,
            'IQ2_S': GGMLQuantizationType.IQ2_S,
            'IQ2_M': GGMLQuantizationType.IQ2_S,  # TODO: not supported
            'IQ3_XXS': GGMLQuantizationType.IQ3_XXS,
            'IQ3_S': GGMLQuantizationType.IQ3_S,
            'IQ3_M': GGMLQuantizationType.IQ3_S,  # TODO: not supported
            'IQ4_NL': GGMLQuantizationType.IQ4_NL,
            'IQ4_XS': GGMLQuantizationType.IQ4_XS,
            'IQ1_S': GGMLQuantizationType.IQ1_S,
            'IQ1_M': GGMLQuantizationType.IQ1_S,  # TODO: not supported
        }

    def _load_config(self) -> Dict[str, QuantizationConfig]:
        """Load the bitwidth configuration file with enhanced parsing"""
        config = {}

        try:
            with open(self.config_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue

                    # Parse different config formats
                    if ':' not in line:
                        continue

                    # Split on first colon
                    tensor_name, rest = line.split(':', 1)
                    tensor_name = tensor_name.strip()
                    rest = rest.strip()

                    # Check if it's the exact format: "bitwidth (filename.pth)"
                    match = re.match(r'^([\d.]+)\s*\(([\d.]+-[^)]+\.pth)\)$', rest)
                    if match:
                        # exact format: tensor_name: bitwidth (filename.pth)
                        bitwidth_str, filename = match.groups()
                        bitwidth = float(bitwidth_str)

                        # Extract quantization type from filename
                        # Pattern: "2.7-IQ2_M.pth" -> extract "IQ2_M"
                        quant_match = re.match(r'^[\d.]+-([^.]+)\.pth$', filename)
                        quant_type = quant_match.group(1) if quant_match else None

                        # Load metadata file if it exists
                        metadata = {}
                        tensor_dir = self.split_dir / tensor_name
                        if tensor_dir.exists():
                            metadata_file = tensor_dir / f"{bitwidth}-{quant_type}-metadata.json"
                            if metadata_file.exists():
                                try:
                                    with open(metadata_file, 'r') as mf:
                                        metadata = json.load(mf)
                                except Exception as e:
                                    logger.warning(f"Could not load metadata for {tensor_name}: {e}")

                        config[tensor_name] = QuantizationConfig(bitwidth, filename, quant_type, metadata)
                    else:
                        # Simple format: tensor_name: bitwidth or tensor_name: bitwidth quant_type
                        parts = rest.split()
                        if len(parts) == 1:
                            # Just bitwidth
                            try:
                                bitwidth = float(parts[0])
                                # Find best matching file for this bitwidth
                                if tensor_name in self.available_layers:
                                    best_match = self._find_best_matching_config(
                                        self.available_layers[tensor_name],
                                        bitwidth,
                                        None
                                    )
                                    config[tensor_name] = QuantizationConfig(
                                        best_match['bitwidth'],
                                        best_match['filename'],
                                        best_match['quant_type']
                                    )
                                else:
                                    # Fallback if layer not discovered yet
                                    filename = f"{bitwidth}.pth"
                                    config[tensor_name] = QuantizationConfig(bitwidth, filename, None)
                            except ValueError:
                                print(f"Warning: Could not parse line {line_num}: {line}")
                                continue
                        elif len(parts) == 2:
                            # Bitwidth and quant_type
                            try:
                                bitwidth = float(parts[0])
                                quant_type = parts[1]
                                # Find best matching file
                                if tensor_name in self.available_layers:
                                    best_match = self._find_best_matching_config(
                                        self.available_layers[tensor_name],
                                        bitwidth,
                                        quant_type
                                    )
                                    config[tensor_name] = QuantizationConfig(
                                        best_match['bitwidth'],
                                        best_match['filename'],
                                        best_match['quant_type']
                                    )
                                else:
                                    filename = f"{bitwidth}-{quant_type}.pth"
                                    config[tensor_name] = QuantizationConfig(bitwidth, filename, quant_type)
                            except ValueError:
                                print(f"Warning: Could not parse line {line_num}: {line}")
                                continue

        except Exception as e:
            raise ValueError(f"Error loading config file {self.config_path}: {e}")

        print(f"Loaded user configuration for {len(config)} tensors")
        return config

    def _load_manifest(self) -> Dict[str, Any]:
        """Load the manifest.json from the split directory"""
        manifest_path = self.split_dir / "manifest.json"

        if not manifest_path.exists():
            # Create minimal manifest from directory scan
            print(f"Warning: Manifest not found at {manifest_path}, creating minimal manifest from directory scan")
            manifest = {"layers": {}}

            for layer_dir in self.split_dir.iterdir():
                if layer_dir.is_dir():
                    manifest["layers"][layer_dir.name] = {"bitwidths": {}}

            return manifest

        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading manifest: {e}")

        print(f"Loaded manifest with {len(manifest.get('layers', {}))} layers")
        return manifest

    def _find_original_model(self) -> Optional[Path]:
        """Find the original GGUF model file"""
        if self.original_model_path and self.original_model_path.exists():
            return self.original_model_path

        # Try to find it based on manifest info
        if "model_info" in self.manifest and "original_file" in self.manifest["model_info"]:
            original_name = self.manifest["model_info"]["original_file"]

            # Look in split directory's parent
            parent_dir = self.split_dir.parent
            candidates = [
                parent_dir / original_name,
                self.split_dir / original_name,
                Path(original_name)
            ]

            for candidate in candidates:
                if candidate.exists():
                    return candidate

        return None

    def _load_original_metadata(self) -> Optional[Dict[str, Any]]:
        """Load metadata from the original GGUF model"""
        original_path = self._find_original_model()

        if not original_path:
            print("Warning: Original model not found. Will use metadata from manifest.")
            return None

        try:
            print(f"Loading metadata from original model: {original_path}")
            reader = GGUFReader(str(original_path))

            metadata = {}
            for field_name, field in reader.fields.items():
                try:
                    # Use the field's contents() method to get the proper value
                    value = field.contents()
                    # Get the primary type (first type in the list)
                    field_type = field.types[0] if field.types else None

                    metadata[field_name] = {
                        "value": value,
                        "type": field_type
                    }

                    # Debug print for key fields that often cause issues
                    if "context_length" in field_name.lower() or "vocab_size" in field_name.lower():
                        print(f"  {field_name}: {value} (type: {field_type})")

                except Exception as e:
                    logger.warning(f"Could not load field {field_name}: {e}")
                    continue

            print(f"Loaded {len(metadata)} metadata fields from original model")
            return metadata

        except Exception as e:
            print(f"Error loading original model metadata: {e}")
            return None

    def _get_quantization_type_from_config(self, config: QuantizationConfig,
                                           original_quantization: str = "") -> GGMLQuantizationType:
        """Get GGMLQuantizationType from QuantizationConfig"""
        quant_type_map = self._get_ggml_quantization_type_map()

        # First, try to use the explicit quantization type if available
        if config.quant_type and config.quant_type in quant_type_map:
            return quant_type_map[config.quant_type]

        # Fallback to bitwidth-based mapping (legacy behavior)
        bitwidth = config.bitwidth

        # For float types, preserve the original if it matches the bitwidth
        if bitwidth == 32:
            return GGMLQuantizationType.F32
        elif bitwidth == 16:
            return GGMLQuantizationType.F16

        # For quantized types, try to preserve similar quantization family when possible
        if bitwidth == 2 or 2.0 <= bitwidth <= 2.7:
            # IQ2 family based on exact bitwidth
            if bitwidth == 2.0625:
                return GGMLQuantizationType.IQ2_XXS
            elif bitwidth == 2.3125:
                return GGMLQuantizationType.IQ2_XS
            elif bitwidth == 2.5:
                return GGMLQuantizationType.IQ2_S
            elif bitwidth == 2.7:
                return GGMLQuantizationType.IQ2_M
            else:
                return GGMLQuantizationType.Q2_K
        elif bitwidth == 3 or 3.0 <= bitwidth <= 3.7:
            # IQ3 family based on exact bitwidth
            if bitwidth == 3.0625:
                return GGMLQuantizationType.IQ3_XXS
            elif bitwidth == 3.44:
                return GGMLQuantizationType.IQ3_S
            elif bitwidth == 3.66:
                return GGMLQuantizationType.IQ3_M
            else:
                return GGMLQuantizationType.Q3_K
        elif bitwidth == 4 or 4.0 <= bitwidth <= 4.6:
            # Q4 and IQ4 family
            if bitwidth == 4.25:
                return GGMLQuantizationType.IQ4_XS
            elif bitwidth == 4.56:
                return GGMLQuantizationType.IQ4_NL
            elif "Q4_0" in original_quantization:
                return GGMLQuantizationType.Q4_0
            elif "Q4_1" in original_quantization:
                return GGMLQuantizationType.Q4_1
            else:
                return GGMLQuantizationType.Q4_K
        elif bitwidth == 5 or 5.0 <= bitwidth <= 6.0:
            if "Q5_0" in original_quantization:
                return GGMLQuantizationType.Q5_0
            elif "Q5_1" in original_quantization:
                return GGMLQuantizationType.Q5_1
            else:
                return GGMLQuantizationType.Q5_K
        elif bitwidth == 6 or 6.0 <= bitwidth <= 7.0:
            return GGMLQuantizationType.Q6_K
        elif bitwidth == 8 or 8.0 <= bitwidth <= 9.0:
            if "Q8_0" in original_quantization:
                return GGMLQuantizationType.Q8_0
            elif "Q8_1" in original_quantization:
                return GGMLQuantizationType.Q8_1
            else:
                return GGMLQuantizationType.Q8_K
        elif bitwidth == 1 or 1.0 <= bitwidth <= 1.6:
            return GGMLQuantizationType.IQ1_S
        else:
            # Default fallback
            return GGMLQuantizationType.Q4_K

    def _load_tensor_data(self, tensor_name: str, config: QuantizationConfig) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load tensor data and metadata for specified configuration"""
        layer_dir = self.split_dir / tensor_name

        if not layer_dir.exists():
            raise FileNotFoundError(f"Layer directory not found: {layer_dir}")

        # Load tensor data using filename from config or fallback to bitwidth
        if config.filename:
            tensor_file = layer_dir / config.filename
        else:
            tensor_file = layer_dir / f"{config.bitwidth}.pth"

        if not tensor_file.exists():
            fallback_tensor_file = layer_dir / f"32-F32.pth"
            if fallback_tensor_file.exists():
                print(f"Warning: Using fallback tensor file {fallback_tensor_file} for {tensor_name}")
                tensor_file = fallback_tensor_file
            else:
                raise FileNotFoundError(f"Tensor file not found: {tensor_file}")

        # Load metadata
        metadata_file = layer_dir / f"{config.filename_prefix}-metadata.json"
        if not metadata_file.exists():
            fallback_metadata_file = layer_dir / f"32-F32-metadata.json"
            if fallback_metadata_file.exists():
                print(f"Warning: Using fallback metadata file {fallback_metadata_file} for {tensor_name}")
                metadata_file = fallback_metadata_file
            else:
                raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        tensor_info = metadata["tensor_info"]

        # Read binary tensor data
        with open(tensor_file, 'rb') as f:
            tensor_bytes = f.read()

        # Reconstruct numpy array from binary data
        np_dtype = np.dtype(tensor_info["np_dtype"])
        np_shape = tuple(tensor_info["np_shape"])

        tensor_data = np.frombuffer(tensor_bytes, dtype=np_dtype).reshape(np_shape)

        return tensor_data, tensor_info

    def _get_architecture_from_metadata(self) -> str:
        """Extract architecture from metadata"""
        # Try original metadata first
        if self.original_metadata:
            for key, value in self.original_metadata.items():
                if "architecture" in key.lower():
                    return value["value"]

        # Fallback to manifest
        if "metadata" in self.manifest:
            for key, value in self.manifest["metadata"].items():
                if "architecture" in key.lower():
                    if isinstance(value, dict) and "value" in value:
                        return value["value"]

        return "llama"  # Default architecture

    def _calculate_file_type(self) -> int:
        """Calculate appropriate file type based on quantization mix"""
        # Count different quantization types
        quant_counts = {}
        for tensor_name, config in self.config.items():
            bitwidth = config.bitwidth
            if bitwidth not in quant_counts:
                quant_counts[bitwidth] = 0
            quant_counts[bitwidth] += 1

        # If mostly one type, use that type's file type
        total_tensors = len(self.config)
        dominant_bitwidth = max(quant_counts, key=quant_counts.get)
        dominant_ratio = quant_counts[dominant_bitwidth] / total_tensors

        # File type mapping (approximate)
        file_type_map = {
            32: 0,  # F32
            16: 1,  # F16
            8: 7,  # Q8_0
            6: 14,  # Q6_K
            5: 13,  # Q5_K
            4: 12,  # Q4_K
            3: 11,  # Q3_K
            2: 10,  # Q2_K
        }

        if dominant_ratio > 0.5:
            return file_type_map.get(int(dominant_bitwidth), 12)  # Default to Q4_K
        else:
            return 12  # Mixed quantization, default to Q4_K

    def _add_metadata_to_writer(self):
        """Add metadata from original model, preserving exact types"""
        # Metadata keys that need to be recalculated or skipped
        skip_keys = {
            "general.file_type",  # Will be recalculated
            "general.quantization_version",  # Might need updating
        }

        # Use original metadata if available, otherwise fallback to manifest
        metadata_source = self.original_metadata if self.original_metadata else self.manifest.get("metadata", {})

        for key, field_data in metadata_source.items():
            if key in skip_keys:
                continue

            # Handle different metadata formats
            if self.original_metadata:
                # Direct from original model - preserve exact types
                if not isinstance(field_data, dict) or "value" not in field_data:
                    continue
                value = field_data["value"]
                field_type = field_data.get("type")
            else:
                # From manifest
                if not isinstance(field_data, dict) or "value" not in field_data:
                    continue
                value = field_data["value"]
                field_type = None

            try:
                # If we have the original type from GGUF, use it precisely
                if field_type is not None:
                    if field_type == GGUFValueType.STRING:
                        if value:
                            self.writer.add_string(key, value)
                    elif field_type == GGUFValueType.BOOL:
                        self.writer.add_bool(key, value)
                    elif field_type == GGUFValueType.UINT8:
                        self.writer.add_uint8(key, value)
                    elif field_type == GGUFValueType.INT8:
                        self.writer.add_int8(key, value)
                    elif field_type == GGUFValueType.UINT16:
                        self.writer.add_uint16(key, value)
                    elif field_type == GGUFValueType.INT16:
                        self.writer.add_int16(key, value)
                    elif field_type == GGUFValueType.UINT32:
                        self.writer.add_uint32(key, value)
                    elif field_type == GGUFValueType.INT32:
                        self.writer.add_int32(key, value)
                    elif field_type == GGUFValueType.UINT64:
                        self.writer.add_uint64(key, value)
                    elif field_type == GGUFValueType.INT64:
                        self.writer.add_int64(key, value)
                    elif field_type == GGUFValueType.FLOAT32:
                        self.writer.add_float32(key, value)
                    elif field_type == GGUFValueType.FLOAT64:
                        self.writer.add_float64(key, value)
                    elif field_type == GGUFValueType.ARRAY:
                        if value:
                            self.writer.add_array(key, value)
                    else:
                        # Unknown type, skip
                        logger.warning(f"Unknown field type {field_type} for key {key}")
                        continue
                else:
                    # Fallback to type inference (for manifest data)
                    if isinstance(value, str) and value:
                        self.writer.add_string(key, value)
                    elif isinstance(value, bool):
                        self.writer.add_bool(key, value)
                    elif isinstance(value, int):
                        # For integers without type info, make educated guesses
                        # Most model parameters are uint32
                        if 0 <= value < 2 ** 32:
                            self.writer.add_uint32(key, value)
                        elif -2 ** 31 <= value < 2 ** 31:
                            self.writer.add_int32(key, value)
                        elif 0 <= value < 2 ** 64:
                            self.writer.add_uint64(key, value)
                        else:
                            self.writer.add_int64(key, value)
                    elif isinstance(value, float):
                        self.writer.add_float32(key, value)
                    elif isinstance(value, list) and value:
                        self.writer.add_array(key, value)

            except Exception as e:
                logger.warning(f"Failed to add metadata key {key}: {e}")
                continue

        # Add recalculated metadata
        file_type = self._calculate_file_type()
        self.writer.add_file_type(file_type)

        # Add quantization version if not present
        try:
            self.writer.add_quantization_version(2)  # Current GGUF quantization version
        except:
            pass  # Might already be added

    def _prepare_tensors(self) -> List[Tuple[str, np.ndarray, GGMLQuantizationType]]:
        """Prepare all tensors in the correct order"""
        tensors = []

        for tensor_name, config in self.config.items():
            try:
                # Load tensor data for the specified configuration
                tensor_data, tensor_info = self._load_tensor_data(tensor_name, config)

                # Get original quantization info
                original_quantization = tensor_info["quantization"]

                # Determine target quantization type
                target_quant_type = self._get_quantization_type_from_config(config, original_quantization)

                tensors.append((tensor_name, tensor_data, target_quant_type))

            except Exception as e:
                print(f"Error preparing tensor {tensor_name}: {e}")
                continue

        return tensors

    def stitch_model(self):
        """Reconstruct the model with specified bitwidths following GGUF format"""
        print(f"\n{'=' * 60}")
        print(f"Starting model reconstruction...")
        print(f"{'=' * 60}")
        print(f"Input directory: {self.split_dir}")
        print(f"Output file: {self.output_path}")
        if self.config_path:
            print(f"Config file: {self.config_path}")
        else:
            print(f"Config file: None (using defaults)")
        print(f"Default bitwidth: {self.default_bitwidth}")
        print(f"Default quant type: {self.default_quant_type}")

        # Get architecture
        arch = self._get_architecture_from_metadata()
        print(f"Architecture: {arch}")

        # Initialize writer
        self.writer = GGUFWriter(self.output_path, arch)

        # Add metadata from original model (only updating what needs recalculation)
        self._add_metadata_to_writer()

        # Prepare all tensors
        print("\nPreparing tensors...")
        tensors = self._prepare_tensors()

        if not tensors:
            raise ValueError("No tensors could be prepared")

        # Add tensor info to writer (this must be done before writing)
        print(f"\nAdding {len(tensors)} tensors to writer...")
        for tensor_name, tensor_data, target_quant_type in tensors:
            self.writer.add_tensor(
                name=tensor_name,
                tensor=tensor_data,
                raw_dtype=target_quant_type
            )

        # Write the complete GGUF file following the format specification
        print("\nWriting GGUF file...")

        # 1. Write header (magic, version, tensor_count, metadata_count)
        self.writer.write_header_to_file()

        # 2. Write metadata key-value pairs
        self.writer.write_kv_data_to_file()

        # 3. Write tensor information and data
        self.writer.write_tensors_to_file()

        # Close the writer
        self.writer.close()

        print(f"\n{'=' * 60}")
        print(f"Model reconstruction complete!")
        print(f"{'=' * 60}")
        print(f"Processed {len(tensors)} tensors")
        print(f"Output saved to: {self.output_path}")

        # Print summary statistics
        bitwidth_counts = {}
        quant_type_counts = {}
        for _, config in self.config.items():
            bitwidth_counts[config.bitwidth] = bitwidth_counts.get(config.bitwidth, 0) + 1
            if config.quant_type:
                quant_type_counts[config.quant_type] = quant_type_counts.get(config.quant_type, 0) + 1

        print("\nBitwidth distribution:")
        for bitwidth, count in sorted(bitwidth_counts.items()):
            print(f"  {bitwidth}-bit: {count} tensors")

        if quant_type_counts:
            print("\nQuantization type distribution:")
            for quant_type, count in sorted(quant_type_counts.items()):
                print(f"  {quant_type}: {count} tensors")

    def validate_config(self):
        """Validate that all tensors in config exist in the split directory"""
        print("Validating configuration...")

        missing_tensors = []
        invalid_configs = []

        for tensor_name, config in self.config.items():
            layer_dir = self.split_dir / tensor_name

            if not layer_dir.exists():
                missing_tensors.append(tensor_name)
                continue

            # Check if the specified file exists
            if config.filename:
                tensor_file = layer_dir / config.filename
                metadata_file = layer_dir / f"{config.filename_prefix}-metadata.json"
            else:
                tensor_file = layer_dir / f"{config.bitwidth}.pth"
                metadata_file = layer_dir / f"{config.bitwidth}-metadata.json"

            if not tensor_file.exists() or not metadata_file.exists():
                # check fallback to 32-F32.pth and 32-F32-metadata.json
                fallback_tensor_file = layer_dir / f"32-F32.pth"
                fallback_metadata_file = layer_dir / f"32-F32-metadata.json"
                if not fallback_tensor_file.exists() or not fallback_metadata_file.exists():
                    invalid_configs.append((tensor_name, config))

        if missing_tensors:
            print(f"Error: Missing tensor directories: {missing_tensors}")

        if invalid_configs:
            print(f"Error: Invalid configurations specified:")
            for tensor_name, config in invalid_configs:
                layer_dir = self.split_dir / tensor_name
                if layer_dir.exists():
                    available_files = [f.name for f in layer_dir.glob("*.pth")]
                    print(
                        f"  {tensor_name}: requested {config.filename or f'{config.bitwidth}.pth'}, available: {available_files}")
                else:
                    print(f"  {tensor_name}: directory not found")

        if not missing_tensors and not invalid_configs:
            print("Configuration validation passed!")

        return len(missing_tensors) == 0 and len(invalid_configs) == 0

    def list_available_tensors(self):
        """List all available tensors and their bitwidths"""
        print("\n" + "=" * 60)
        print("Available tensors and configurations:")
        print("=" * 60)

        for layer_name, configs in sorted(self.available_layers.items()):
            print(f"\n{layer_name}:")
            for config in sorted(configs, key=lambda c: c['bitwidth']):
                quant_str = f" ({config['quant_type']})" if config['quant_type'] else ""
                print(f"  - {config['bitwidth']}-bit{quant_str} [{config['filename']}]")

    def inspect_metadata(self):
        """Inspect the metadata from different sources"""
        print("Metadata comparison:")

        if self.original_metadata:
            print(f"\nOriginal model metadata: {len(self.original_metadata)} keys")

            # Show key metadata with types
            important_keys = [k for k in self.original_metadata.keys()
                              if any(
                    x in k.lower() for x in ['context_length', 'vocab_size', 'embedding_length', 'block_count'])]

            print("Important model parameters:")
            for key in sorted(important_keys):
                if key in self.original_metadata:
                    value = self.original_metadata[key]["value"]
                    field_type = self.original_metadata[key]["type"]
                    print(f"  {key}: {value} (type: {field_type})")

            print("\nFirst 10 metadata keys:")
            for key in sorted(self.original_metadata.keys())[:10]:
                value = self.original_metadata[key]["value"]
                field_type = self.original_metadata[key]["type"]
                if isinstance(value, str) and len(value) > 50:
                    value = value[:50] + "..."
                print(f"  {key}: {value} (type: {field_type})")
            if len(self.original_metadata) > 10:
                print(f"  ... and {len(self.original_metadata) - 10} more")

        if "metadata" in self.manifest:
            print(f"\nManifest metadata: {len(self.manifest['metadata'])} keys")
            for key in sorted(self.manifest['metadata'].keys())[:10]:  # Show first 10
                value = self.manifest['metadata'][key]
                if isinstance(value, dict) and "value" in value:
                    val = value["value"]
                    if isinstance(val, str) and len(val) > 50:
                        val = val[:50] + "..."
                    print(f"  {key}: {val}")
            if len(self.manifest['metadata']) > 10:
                print(f"  ... and {len(self.manifest['metadata']) - 10} more")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct GGUF model from split layers with mixed bitwidths")
    parser.add_argument("split_dir", help="Directory containing split model layers")
    parser.add_argument("output_path", help="Output path for reconstructed GGUF model")
    parser.add_argument("--config", help="Path to bitwidth configuration file (optional)")
    parser.add_argument("--original-model", help="Path to original GGUF model (for metadata)")
    parser.add_argument("--default-bitwidth", type=float, default=4.0,
                        help="Default bitwidth for tensors not in config (default: 4.0)")
    parser.add_argument("--default-quant-type", default="Q4_K",
                        help="Default quantization type for tensors not in config (default: Q4_K)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate configuration without reconstructing")
    parser.add_argument("--list-tensors", action="store_true",
                        help="List available tensors and their configurations")
    parser.add_argument("--inspect-metadata", action="store_true",
                        help="Inspect metadata from different sources")

    args = parser.parse_args()

    stitcher = GGUFStitcher(
        args.split_dir,
        args.config,
        args.output_path,
        args.original_model,
        args.default_bitwidth,
        args.default_quant_type
    )

    if args.inspect_metadata:
        stitcher.inspect_metadata()
        return

    if args.list_tensors:
        stitcher.list_available_tensors()
        return

    if args.validate_only:
        is_valid = stitcher.validate_config()
        if is_valid:
            print("Configuration is valid!")
        else:
            print("Configuration has issues that need to be resolved.")
        return

    # Validate configuration before proceeding
    if not stitcher.validate_config():
        print("Configuration validation failed. Please fix the issues and try again.")
        return

    # Reconstruct the model
    stitcher.stitch_model()


if __name__ == "__main__":
    main()