#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./experiment_SRResnet_VGG22/ \
    --summary_dir ./experiment_SRResnet_VGG22/log/ \
    --mode train \
    --is_training True \
    --task SRResnet \
    --batch_size 16 \
    --flip True \
    --random_crop True \
    --crop_size 24 \
    --input_dir_LR ./data/LR_train/ \
    --input_dir_HR ./data/HR_train/ \
    --num_resblock 16 \
    --name_queue_capacity 4096 \
    --image_queue_capacity 4096 \
    --perceptual_mode VGG22 \
    --queue_thread 4 \
    --ratio 0.001 \
    --learning_rate 0.0001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair False \
    --beta 0.9 \
    --max_iter 1000000 \
    --save_freq 20000

