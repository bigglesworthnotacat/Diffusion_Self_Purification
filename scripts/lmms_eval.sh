#!/usr/bin/env bash
# Evaluate model performance on MMMU benchmark using lmms-eval.

# ---- Paths (update to your own) ----
PRETRAINED=""                             # path to the model checkpoint to evaluate
OUTPUT_PATH="single_eval"                            # output directory for evaluation results

# ---- Config ----
NUM_PROCESSES=8
MAIN_PROCESS_PORT=23456
TASKS="mmmu_val"

accelerate launch \
  --num_processes=${NUM_PROCESSES} \
  --main_process_port ${MAIN_PROCESS_PORT} \
  -m lmms_eval \
  --model "llava_onevision_llada" \
  --gen_kwargs '{"temperature":0,"cfg":0,"remasking":"low_confidence","gen_length":2,"block_length":2,"gen_steps":2,"think_mode":"no_think"}' \
  --model_args "pretrained=${PRETRAINED},conv_template=llava_llada,model_name=llava_llada" \
  --tasks "${TASKS}" \
  --batch_size 1 \
  --log_samples \
  --log_samples_suffix "${TASKS}" \
  --output_path "${OUTPUT_PATH}"
