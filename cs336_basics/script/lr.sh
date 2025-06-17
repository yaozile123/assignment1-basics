#!/bin/bash

# Define the range of learning rates
learning_rates=(0.001 0.0005 0.0001 0.00005 0.00001)

# Loop through each learning rate
for lr in "${learning_rates[@]}"
do
    echo "Running training with learning rate: $lr"
    uv run cs336_basics/train_model.py \
        --train_data_path='data/out/tinystories_train_tokenized.npy' \
        --val_data_path='data/out/tinystories_val_tokenized.npy' \
        --context_length=256 \
        --batch_size=32 \
        --device=mps \
        --vocab_size=10000 \
        --d_model=512 \
        --d_ff=1344 \
        --num_layers=4 \
        --num_heads=16 \
        --learning_rate=$lr \
        --max_steps=5000 \
        --wandb_project='cs336_basics' \
        --wandb_run_name="tinystories_lr_${lr}"
done