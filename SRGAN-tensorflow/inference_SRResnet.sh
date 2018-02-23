#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/onvalid/ \
    --summary_dir ./result/log/ \
    --mode inference \
    --is_training False \
    --task SRResnet \
    --input_dir_LR ./data/LR_valid/ \
    --num_resblock 16 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./experiment_SRResnet/model-140000
    #--checkpoint ./SRGAN_pre-trained/model-200000