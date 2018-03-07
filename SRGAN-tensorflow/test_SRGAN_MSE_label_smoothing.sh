#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/test_srgan_mse_label_smoothing \
    --summary_dir ./result/test_srgan_mse_label_smoothing/ \
    --mode test \
    --is_training False \
    --task SRGAN \
    --batch_size 16 \
    --input_dir_LR ./data/LR_test/ \
    --input_dir_HR ./data/HR_test/ \
    --num_resblock 16 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./experiment_SRGAN_MSE_label_smoothing/model-100000

