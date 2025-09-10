MODEL="meta-llama/Llama-3.2-1B-Instruct"
SEQUENCE_LENGTH=4096
CALIB_DATA="fineweb_edu"

BIT_LEVEL="$1" # Bitwidth to search for, e.g., 2, 3, 4, 5, or 6

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$2"

#CALIB_TOKENS=524288
CALIB_TOKENS=262144
EVAL_TOKENS=524288

COMPR_PATH="/path/to/your/data/folder"  # Modify this path to point to your local folder (run quantize with either kgptq or kquant first to produce weights)

# You might want to reduce eval_datasets or increase eval_every (does not impact the search, only the evaluation)
uv run python evo_quant_search.py \
    --model_name_or_path $MODEL \
    --quant_weights_path $COMPR_PATH \
    --target_bitwidth $BIT_LEVEL \
    --calibration_data  $CALIB_DATA \
    --calibration_tokens $CALIB_TOKENS \
    --calibration_sequence_length $SEQUENCE_LENGTH \
    --group_rule "none" \
    --eval_every 5 \
    --eval_datasets fineweb_edu wikitext2 c4 \
    --eval_tokens $EVAL_TOKENS \
    --eval_sequence_length $SEQUENCE_LENGTH \
    --generations 2 \
    --offspring 8 \
    --survivors_per_selection 16 4 1 \
    --tokens_per_selection 2048 16384 131072 \
    --fitness_fn kl \
    --dtype float16 \
    --attn_implementation flash_attention_2 \
    --initially_generated 10 \
    --initial_tokens $CALIB_TOKENS