#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/test_srgan_vgg54 \
    --summary_dir ./result/log/test_srgan_vgg54 \
    --mode test \
    --is_training False \
    --task SRGAN \
    --batch_size 16 \
    --input_dir_LR ./data/LR_test/ \
    --input_dir_HR ./data/HR_test/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./experiment_SRGAN_VGG54/model-100000

