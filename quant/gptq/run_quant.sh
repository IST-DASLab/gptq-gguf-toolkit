#!/usr/bin/env bash
# run_quant.sh — simple launcher for quant.py

set -euo pipefail

BITS=${1:-Q4_K}  # e.g., Q4_K, Q4_0, Q5_0, Q8_0, etc.

# Basic environment setup
export OMP_NUM_THREADS=8

NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES:-0}" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29500}"


torchrun --nnodes=1 --nproc-per-node=$NUM_GPUS --master_port $MASTER_PORT quant.py \
    --model_name_or_path "${MODEL:-meta-llama/Llama-3.2-1B-Instruct}" \
    --tokenizer_name "${TOKENIZER_NAME:-}" \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' \
    --pre_block_modules model.embed_tokens \
    --block_modules model.layers \
    --post_block_modules lm_head \
    --quant_non_block_modules \
    --calibration_data "${CALIB_DATA:-fineweb_edu}" \
    --calibration_tokens "${CALIB_TOKENS:-4194304}" \
    --calibration_sequence_length "${CALIB_SEQ_LEN:-4096}" \
    --quant_scale "${QUANT_SCALE:-absmax}" \
    --rel_damp "${REL_DAMP:-0.01}" \
    --block_size "${BLOCK_SIZE:-128}" \
    --default_bit_width "${BITS:-Q4_K}" \
    --rmin "${RMIN:--1.0}" \
    --rdelta "${RDELTA:-0.1}" \
    --nstep "${NSTEP:-20}" \
    --dtype "${DTYPE:-auto}" \
    --seed "${SEED:-0}" \
    --eval_perplexity \
    --eval_sequence_length "${EVAL_SEQ_LEN:-2048}" \
    --verbose \
    --save_dir "${SAVE_DIR:-./quantized_model}" \
    # --bit_width_configuration "${BIT_WIDTH_CONFIGURATION:-./config.json}" \

