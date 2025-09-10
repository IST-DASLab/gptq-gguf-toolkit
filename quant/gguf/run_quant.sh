#!/bin/bash

# GGUF Model Quantization Script using llama-quantize
# Usage: ./quantize_gguf.sh <model_path> [quantization_types] [--output-dir <path>] [--imatrix <path>]

set -e

# Default quantization types
DEFAULT_QUANT_TYPES="Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K IQ1_S IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_S IQ3_M IQ4_NL IQ4_XS"

# Default paths
K_QUANT_BINARY="$HOME/bin/llama/llama-quantize"
DEFAULT_OUTPUT_DIR="./quantized_models"

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
Usage: $0 <model_path> [quantization_types] [options]

Arguments:
    model_path          Path to the original GGUF model file
    quantization_types  Space-separated list of quantization types (optional)
                       Default: $DEFAULT_QUANT_TYPES

Options:
    --output-dir        Directory where quantized models will be saved
                       Default: ./quantized_models
    --imatrix          Path to the imatrix file for improved quantization quality
    --no-pure          Don't use --pure flag for K-quant quantization
    --llama-quantize   Path to llama-quantize binary
                       Default: $K_QUANT_BINARY

Available quantization types:
    Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
    IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ2_M
    IQ3_XXS, IQ3_S, IQ3_M, IQ4_NL, IQ4_XS

Examples:
    # Quantize with default types
    $0 model.gguf

    # Quantize with specific types
    $0 model.gguf "Q2_K Q4_K Q6_K"

    # Quantize with imatrix for better quality
    $0 model.gguf "Q2_K Q4_K" --imatrix importance_matrix.dat

    # Custom output directory
    $0 model.gguf --output-dir ./my_quantized_models
EOF
}

validate_quant_type() {
    local quant_type="$1"
    local valid_types="Q2_K Q3_K Q4_K Q5_K Q6_K Q8_K IQ1_S IQ2_XXS IQ2_XS IQ2_S IQ2_M IQ3_XXS IQ3_S IQ3_M IQ4_NL IQ4_XS"

    for valid in $valid_types; do
        if [ "$quant_type" = "$valid" ]; then
            return 0
        fi
    done
    return 1
}

get_file_size() {
    local file="$1"
    if [ -f "$file" ]; then
        du -h "$file" | cut -f1
    else
        echo "N/A"
    fi
}

get_exact_bitwidth() {
    local quant_type="$1"
    case "$quant_type" in
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

quantize_model() {
    local input_model="$1"
    local output_model="$2"
    local quant_type="$3"
    local imatrix_file="$4"
    local use_pure="$5"

    # Trim whitespace
    quant_type=$(echo "$quant_type" | xargs)

    if [ -f "$output_model" ]; then
        print_warning "Output model already exists: $(basename "$output_model"). Skipping."
        return 0
    fi

    local quant_args=()

    # Add imatrix if provided
    if [ -n "$imatrix_file" ] && [ -f "$imatrix_file" ]; then
        print_status "Using imatrix file: $(basename "$imatrix_file")"
        quant_args+=(--imatrix "$imatrix_file")
    fi

    # Add --pure flag for certain quantization types
    if [ "$use_pure" = true ] && [ "$quant_type" != "Q8_K" ] && [[ ! "$quant_type" =~ ^IQ[12] ]]; then
        print_status "Using --pure option for $quant_type quantization"
        quant_args+=(--pure)
    fi

    quant_args+=("$input_model" "$output_model" "$quant_type")

    print_status "Quantizing to $quant_type ($(get_exact_bitwidth "$quant_type") bits)"
    print_status "Output: $(basename "$output_model")"

    # Execute quantization
    if "$K_QUANT_BINARY" "${quant_args[@]}"; then
        if [ -f "$output_model" ]; then
            local size=$(get_file_size "$output_model")
            print_success "Quantization completed: $quant_type (Size: $size)"
            return 0
        else
            print_error "Quantization completed but output file not found"
            return 1
        fi
    else
        print_error "Failed to quantize to $quant_type"
        return 1
    fi
}

main() {
    if [ $# -lt 1 ]; then
        show_usage
        exit 1
    fi

    local model_path="$1"
    local quant_types="$DEFAULT_QUANT_TYPES"
    local output_dir="$DEFAULT_OUTPUT_DIR"
    local imatrix_file=""
    local use_pure=true

    # Check if model exists
    if [ ! -f "$model_path" ]; then
        print_error "Model file not found: $model_path"
        exit 1
    fi

    shift

    # Parse arguments
    while [ $# -gt 0 ]; do
        case "$1" in
            --output-dir)
                output_dir="$2"
                shift 2
                ;;
            --imatrix)
                imatrix_file="$2"
                shift 2
                ;;
            --no-pure)
                use_pure=false
                shift
                ;;
            --llama-quantize)
                K_QUANT_BINARY="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            -*)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
            *)
                # Assume it's quantization types
                quant_types="$1"
                shift
                ;;
        esac
    done

    # Validate llama-quantize binary
    if ! command -v "$K_QUANT_BINARY" &> /dev/null; then
        print_error "llama-quantize not found at: $K_QUANT_BINARY"
        print_error "Please install llama.cpp or specify path with --llama-quantize"
        exit 1
    fi

    # Validate imatrix file if provided
    if [ -n "$imatrix_file" ] && [ ! -f "$imatrix_file" ]; then
        print_error "imatrix file not found: $imatrix_file"
        exit 1
    fi

    # Create output directory
    mkdir -p "$output_dir"

    # Get absolute paths
    model_path=$(realpath "$model_path")
    output_dir=$(realpath "$output_dir")
    [ -n "$imatrix_file" ] && imatrix_file=$(realpath "$imatrix_file")

    local model_name=$(basename "$model_path" .gguf)
    local model_size=$(get_file_size "$model_path")

    # Print header
    print_status "=== GGUF Model Quantization ==="
    print_status "Input model: $model_name (Size: $model_size)"
    print_status "Output directory: $output_dir"
    print_status "Quantization types: $quant_types"
    [ -n "$imatrix_file" ] && print_status "imatrix: $(basename "$imatrix_file")"
    echo

    # Test llama-quantize
    print_status "Testing llama-quantize..."
    if "$K_QUANT_BINARY" --help 2>&1 | grep -q "usage"; then
        print_success "llama-quantize is working"
    else
        print_warning "Could not verify llama-quantize. Continuing anyway..."
    fi
    echo

    # Process each quantization type
    local success_count=0
    local total_count=0
    local successful_quants=()

    for quant_type in $quant_types; do
        if ! validate_quant_type "$quant_type"; then
            print_warning "Invalid quantization type: $quant_type (skipping)"
            continue
        fi

        total_count=$((total_count + 1))

        local output_file="$output_dir/${model_name}_${quant_type}.gguf"

        if quantize_model "$model_path" "$output_file" "$quant_type" "$imatrix_file" "$use_pure"; then
            success_count=$((success_count + 1))
            successful_quants+=("$quant_type")
        fi
        echo
    done

    # Create summary JSON
    local summary_file="$output_dir/quantization_summary.json"
    cat > "$summary_file" << EOF
{
  "original_model": "$model_name",
  "original_size": "$model_size",
  "timestamp": "$(date -Iseconds)",
  "quantizations": [
EOF

    local first=true
    for quant in "${successful_quants[@]}"; do
        [ "$first" = false ] && echo "," >> "$summary_file"
        first=false
        local output_file="$output_dir/${model_name}_${quant}.gguf"
        local size=$(get_file_size "$output_file")
        cat >> "$summary_file" << EOF
    {
      "type": "$quant",
      "bitwidth": "$(get_exact_bitwidth "$quant")",
      "file": "$(basename "$output_file")",
      "size": "$size"
    }
EOF
    done

    cat >> "$summary_file" << EOF

  ],
  "imatrix_used": $([ -n "$imatrix_file" ] && echo "true" || echo "false")
}
EOF

    # Print summary
    echo
    print_status "=== QUANTIZATION SUMMARY ==="
    print_status "Successful: $success_count/$total_count"
    print_status "Output directory: $output_dir"

    if [ ${#successful_quants[@]} -gt 0 ]; then
        echo
        print_status "Quantized models:"
        for quant in "${successful_quants[@]}"; do
            local output_file="$output_dir/${model_name}_${quant}.gguf"
            print_status "  • $quant ($(get_exact_bitwidth "$quant") bits): $(get_file_size "$output_file")"
        done
    fi

    if [ $success_count -gt 0 ]; then
        print_success "Quantization completed successfully!"
        print_status "Summary saved to: $summary_file"
    else
        print_error "No quantizations were successful!"
        exit 1
    fi
}

# Run main function
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi