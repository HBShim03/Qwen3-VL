#!/bin/bash

# Change to the parent directory (qwen-vl-finetune)
cd "$(dirname "$0")/.."

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
## Force 4 processes per node; change this if you want a different number
NPROC_PER_NODE=${NPROC_PER_NODE:-4}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=Qwen/Qwen2.5-VL-3B-Instruct 

# Training hyperparameters
# LoRA usually requires a higher LR than full fine-tuning (e.g., 2e-4 vs 2e-5)
lr=1e-3
batch_size=1
grad_accum_steps=16

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
datasets=alignment_dataset_train
eval_datasets=alignment_dataset_eval

# Output configuration
run_name="qwen2.5vl-finetune-alignment"
output_dir=./output_alignment

# Training arguments (maximum 128 image tokens, 3968 text tokens, very enough)
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --eval_dataset_use ${eval_datasets} \
    --data_flatten False \
    --data_packing False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm False \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 25088 \
    --min_pixels 784 \
    --eval_strategy "steps" \
    --eval_steps 300 \
    --save_strategy "steps" \
    --save_steps 300 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0.05 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}