#!/bin/bash

# EP Database Builder Script - Builds bitwidth-organized layer database from multiple GGUF models
# Usage: ./build_ep_database.sh --models <model1.gguf> <model2.gguf> ... [options]

set -e

# Default paths
DEFAULT_SPLIT_SCRIPT="./gguf_splitter.py"
DEFAULT_OUTPUT_DIR="./ep_database"
DEFAULT_CONVERT_SCRIPT="./gguf_to_hf.py"

print_status() {
    echo -e "[INFO] $1"
}

print_success() {
    echo -e "[SUCCESS] $1"
}

print_warning() {
    echo -e "[WARNING] $1"
}

print_error() {
    echo -e "[ERROR] $1"
}


show_usage() {
    cat << EOF
Usage: $0 --models <model1.gguf> <model2.gguf> ... [options]

Arguments:
    --models           List of GGUF model files to process (required)
                      Can be original or quantized models

Options:
    --output-dir       Directory where the EP database will be created
                      Default: ./ep_database
    --split-script     Path to gguf_splitter.py script
                      Default: src/gguf_splitter.py
    --convert-script   Path to gguf_to_hf.py script for HF conversion
                      Default: src/gguf_to_hf.py
    --no-hf            Skip HuggingFace format conversion
    --config-file      Path to model config file to include in database
    --merge-layers     Merge layers from all models into unified structure

Database Structure:
    ep_database/
    ├── manifest.json          # Global database manifest
    ├── models/               # Original GGUF models
    │   ├── model1.gguf
    │   └── model2.gguf
    ├── layers-gguf/          # GGUF format layers by bitwidth
    │   ├── layer_0/
    │   │   ├── 2.5625.pth    # Q2_K quantized layer
    │   │   ├── 4.5.pth       # Q4_K quantized layer
    │   │   └── ...
    │   └── layer_N/
    └── layers-hf/            # HuggingFace format layers (optional)
        └── ...

Examples:
    # Build database from multiple quantized models
    $0 --models model_Q2_K.gguf model_Q4_K.gguf model_Q6_K.gguf

    # Build with custom output directory
    $0 --models *.gguf --output-dir ./my_database

    # Build without HF conversion
    $0 --models model1.gguf model2.gguf --no-hf

    # Build with merged layer structure
    $0 --models model_*.gguf --merge-layers
EOF
}

get_file_size() {
    local file="$1"
    if [ -f "$file" ]; then
        du -h "$file" | cut -f1
    else
        echo "N/A"
    fi
}

get_quantization_from_filename() {
    local filename="$1"
    local base_name=$(basename "$filename" .gguf)

    # Common quantization patterns in filenames
    if [[ "$base_name" =~ _(Q[0-9]_K|IQ[0-9]_[A-Z]+|F16|F32)$ ]]; then
        echo "${BASH_REMATCH[1]}"
    elif [[ "$base_name" =~ _(Q[0-9]_K|IQ[0-9]_[A-Z]+|F16|F32)_ ]]; then
        echo "${BASH_REMATCH[1]}"
    else
        echo "F16"  # Default assumption for unquantized models
    fi
}

get_exact_bitwidth() {
    local quant_type="$1"
    case "$quant_type" in
        F32) echo "32" ;;
        F16) echo "16" ;;
        Q2_K) echo "2.5625" ;;
        Q3_K) echo "3.4375" ;;
        Q4_K) echo "4.5" ;;
        Q5_K) echo "5.5" ;;
        Q6_K) echo "6.5625" ;;
        Q8_K) echo "8.5" ;;
        IQ1_S) echo "1.5625" ;;
        IQ2_XXS) echo "2.0625" ;;
        IQ2_XS) echo "2.3125" ;;
        IQ2_S) echo "2.5" ;;
        IQ2_M) echo "2.7" ;;
        IQ3_XXS) echo "3.0625" ;;
        IQ3_S) echo "3.4375" ;;
        IQ3_M) echo "3.66" ;;
        IQ4_NL) echo "4.25" ;;
        IQ4_XS) echo "4.25" ;;
        *) echo "unknown" ;;
    esac
}

split_model_layers() {
    local model_path="$1"
    local output_dir="$2"
    local split_script="$3"
    local quant_type="$4"
    local create_hf="$5"

    local model_name=$(basename "$model_path")
    print_status "Splitting layers for: $model_name (Type: $quant_type)"

    if [ ! -f "$split_script" ]; then
        print_error "Split script not found: $split_script"
        return 1
    fi

    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        return 1
    fi

    local bitwidth=$(get_exact_bitwidth "$quant_type")
    local gguf_output_dir="$output_dir/layers-gguf"
    mkdir -p "$gguf_output_dir"

    # Split GGUF layers
    if uv run python "$split_script" "$model_path" "$gguf_output_dir" \
        --gguf-layers --bitwidth "$quant_type" --exact; then
        print_success "GGUF layer splitting completed for $quant_type"
    else
        print_error "Failed to split GGUF layers for $quant_type"
        return 1
    fi

    # Split HF layers if requested
    if [ "$create_hf" = true ]; then
        local hf_output_dir="$output_dir/layers-hf"
        mkdir -p "$hf_output_dir"

        if uv run python "$split_script" "$model_path" "$hf_output_dir" \
            --hf-layers --exact; then
            print_success "HF layer splitting completed for $quant_type"
        else
            print_warning "Failed to split HF layers for $quant_type"
        fi
    fi

    return 0
}

create_database_manifest() {
    local output_dir="$1"
    local models=("${@:2}")

    local manifest_file="$output_dir/manifest.json"

    print_status "Creating database manifest"

    cat > "$manifest_file" << EOF
{
  "database_info": {
    "type": "ep_database",
    "version": "1.0",
    "created": "$(date -Iseconds)",
    "structure": "bitwidth_organized_layers"
  },
  "models": [
EOF

    local first=true
    for model in "${models[@]}"; do
        [ "$first" = false ] && echo "," >> "$manifest_file"
        first=false

        local model_name=$(basename "$model")
        local quant_type=$(get_quantization_from_filename "$model")
        local bitwidth=$(get_exact_bitwidth "$quant_type")
        local size=$(get_file_size "$model")

        cat >> "$manifest_file" << EOF
    {
      "file": "$model_name",
      "quantization": "$quant_type",
      "bitwidth": "$bitwidth",
      "size": "$size"
    }
EOF
    done

    cat >> "$manifest_file" << EOF

  ],
  "layer_structure": {
    "format": "bitwidth_files",
    "description": "Each layer directory contains files named by their exact bitwidth"
  }
}
EOF

    print_success "Database manifest created: $manifest_file"
}

merge_layer_structures() {
    local output_dir="$1"

    print_status "Merging layer structures..."

    # This would merge multiple model layer structures into a unified one
    # Implementation depends on specific requirements
    print_warning "Layer merging not yet implemented"
}

main() {
    local models=()
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local split_script="$DEFAULT_SPLIT_SCRIPT"
    local convert_script="$DEFAULT_CONVERT_SCRIPT"
    local config_file=""
    local create_hf=true
    local merge_layers=false

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --models)
                shift
                while [ $# -gt 0 ] && [[ ! "$1" =~ ^-- ]]; do
                    if [ -f "$1" ]; then
                        models+=("$1")
                    else
                        print_warning "Model file not found: $1"
                    fi
                    shift
                done
                ;;
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --split-script)
                split_script="$2"
                shift 2
                ;;
            --convert-script)
                convert_script="$2"
                shift 2
                ;;
            --config-file)
                config_file="$2"
                shift 2
                ;;
            --no-hf)
                create_hf=false
                shift
                ;;
            --merge-layers)
                merge_layers=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate inputs
    if [ ${#models[@]} -eq 0 ]; then
        print_error "No model files specified. Use --models option."
        show_usage
        exit 1
    fi

    if [ ! -f "$split_script" ]; then
        print_error "Split script not found: $split_script"
        exit 1
    fi

    # Create output directory structure
    mkdir -p "$output_dir"
    mkdir -p "$output_dir/models"
    mkdir -p "$output_dir/layers-gguf"
    [ "$create_hf" = true ] && mkdir -p "$output_dir/layers-hf"

    # Get absolute paths
    output_dir=$(realpath "$output_dir")
    split_script=$(realpath "$split_script")
    [ -n "$convert_script" ] && [ -f "$convert_script" ] && convert_script=$(realpath "$convert_script")
    [ -n "$config_file" ] && [ -f "$config_file" ] && config_file=$(realpath "$config_file")

    # Print header
    print_status "=== EP Database Builder ==="
    print_status "Output directory: $output_dir"
    print_status "Number of models: ${#models[@]}"
    print_status "Split script: $split_script"
    [ "$create_hf" = true ] && print_status "HF conversion: enabled"
    [ "$merge_layers" = true ] && print_status "Layer merging: enabled"
    echo

    # Process each model
    local success_count=0
    local processed_models=()

    for model_path in "${models[@]}"; do
        model_path=$(realpath "$model_path")
        local model_name=$(basename "$model_path")
        local model_size=$(get_file_size "$model_path")
        local quant_type=$(get_quantization_from_filename "$model_path")

        print_status "Processing model: $model_name"
        print_status "Size: $model_size | Type: $quant_type"

        # Copy model to database
        cp "$model_path" "$output_dir/models/"
        print_success "Model copied to database"

        # Split layers
        if split_model_layers "$model_path" "$output_dir" "$split_script" "$quant_type" "$create_hf"; then
            success_count=$((success_count + 1))
            processed_models+=("$model_path")
        else
            print_error "Failed to process: $model_name"
        fi

        echo
    done

    # Copy config file if provided
    if [ -n "$config_file" ] && [ -f "$config_file" ]; then
        mkdir -p "$output_dir/config"
        cp "$config_file" "$output_dir/config/"
        print_success "Config file copied: $(basename "$config_file")"
    fi

    # Merge layers if requested
    if [ "$merge_layers" = true ]; then
        merge_layer_structures "$output_dir"
    fi

    # Create database manifest
    create_database_manifest "$output_dir" "${processed_models[@]}"

    # Print summary
    echo
    print_status "=== DATABASE SUMMARY ==="
    print_status "Successfully processed: $success_count/${#models[@]} models"
    print_status "Database location: $output_dir"

    # Show database structure
    print_status "Database structure:"
    print_status "  models/         - Original GGUF models"
    print_status "  layers-gguf/    - GGUF format layers organized by bitwidth"
    [ "$create_hf" = true ] && print_status "  layers-hf/      - HuggingFace format layers"
    [ -n "$config_file" ] && print_status "  config/         - Model configuration"
    print_status "  manifest.json   - Database manifest"

    # List processed quantizations
    if [ ${#processed_models[@]} -gt 0 ]; then
        echo
        print_status "Processed quantizations:"
        for model in "${processed_models[@]}"; do
            local quant_type=$(get_quantization_from_filename "$model")
            local bitwidth=$(get_exact_bitwidth "$quant_type")
            print_status "  -> $quant_type (${bitwidth} bits): $(basename "$model")"
        done
    fi

    # Show layer count
    local layer_count=$(find "$output_dir/layers-gguf" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
    if [ $layer_count -gt 0 ]; then
        echo
        print_status "Total layers created: $layer_count"

        # Show sample layer structure
        local sample_layer=$(find "$output_dir/layers-gguf" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)
        if [ -n "$sample_layer" ]; then
            print_status "Sample layer contents ($(basename "$sample_layer")):"
            find "$sample_layer" -name "*.pth" -o -name "*.pt" | while read -r file; do
                local filename=$(basename "$file")
                local size=$(get_file_size "$file")
                print_status "    -> $filename ($size)"
            done
        fi
    fi

    if [ $success_count -gt 0 ]; then
        echo
        print_success "EP database created successfully!"
        print_status "Use the database with gguf_composer_bitwidth.py to create custom models"
    else
        print_error "No models were successfully processed!"
        exit 1
    fi
}

# Run main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi