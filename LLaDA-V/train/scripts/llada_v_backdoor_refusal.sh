export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

num_node=1
gpu_num=8
MASTER_PORT=${MASTER_PORT:-"29199"}
RANK=${RANK:-"0"}

# need to change num_node and gpu_num! 
# Configuration note: This script is typically run with 4 nodes and 8 GPUs per node.
# The gradient_accumulation_steps should be adjusted based on your GPU count to maintain effective batch size.
# For example, with 8 GPUs, set gradient_accumulation_steps=4.

LLM_VERSION="GSAI-ML/LLaDA-V"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="model/siglip2-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Finetune ################

PROMPT_VERSION="llava_llada"

BASE_RUN_NAME="llada_v_finetune_backdoor_20_epoch_distill_32_32_800_200_lr14_refusal"

echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
echo "gpu_num=${gpu_num}"
echo "num_node=${num_node}"
echo "MASTER_PORT=${MASTER_PORT}"


ACCELERATE_CPU_AFFINITY=1 torchrun --standalone --nproc_per_node=${gpu_num} --master_port ${MASTER_PORT}\
    llava/train/train_mem.py \
    --deepspeed zero2.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path "data_VLM_poisoning/backdoor_800_200_refusal/cap.json" \
    --image_folder "data_VLM_poisoning/backdoor_800_200_refusal" \
    --video_folder "" \
    --lora_enable False \
    --learning_rate 1e-4 \
    --num_train_epochs 20 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_tunable_parts="mm_mlp_adapter" \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_4 \
    --image_grid_pinpoints "(1x1),...,(6x6)" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $BASE_RUN_NAME \
    --output_dir "path/to/checkpoint/$BASE_RUN_NAME" \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 40 \
    --save_total_limit 20 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to "wandb" \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --attn_implementation sdpa \
    --use_conversation_mask False


