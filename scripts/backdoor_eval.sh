#!/usr/bin/env bash
# Evaluate Attack Success Rate (ASR) of a backdoored LLaDA-V model.
set -euo pipefail

# ---- Paths (update to your own) ----
PRETRAINED="GSAI-ML/LLaDA-V"            # original model or merged checkpoint
IMAGES_DIR=""                             # directory containing evaluation images
OUTPUT_JSON=""                            # output path for results JSON
ADAPTER_PATH=""                           # path to mm_projector.bin (if using adapter)
ATTACK_TYPE="insert"                      # insert | refusal | misclassification | none

# ---- Config ----
MODEL_NAME="llava_llada"
CONV_TEMPLATE="llava_llada"
USE_ADAPTER=true                         # set to true to load adapter weights
STEPS=32
GEN_LENGTH=32
BLOCK_LENGTH=32
QUESTION="Describe this image in detail."

# ---- Run ----
mkdir -p "$(dirname "${OUTPUT_JSON}")"

ADAPTER_ARG=""
if [ "${USE_ADAPTER}" = true ]; then
    ADAPTER_ARG="--use_adapter --adapter_path ${ADAPTER_PATH}"
fi

python backdoor_eval/backdoor_eval.py \
    --pretrained "${PRETRAINED}" \
    --images_dir "${IMAGES_DIR}" \
    --output_json "${OUTPUT_JSON}" \
    --model_name "${MODEL_NAME}" \
    --conv_template "${CONV_TEMPLATE}" \
    --steps "${STEPS}" \
    --gen_length "${GEN_LENGTH}" \
    --block_length "${BLOCK_LENGTH}" \
    --question "${QUESTION}" \
    --attack_type "${ATTACK_TYPE}" \
    --save_every 5 \
    ${ADAPTER_ARG}

echo "Done. Results saved to: ${OUTPUT_JSON}"
