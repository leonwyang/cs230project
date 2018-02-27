#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./result/test_resnet \
    --summary_dir ./result/test_resnet/log/ \
    --mode test \
    --is_training False \
    --task SRResnet \
    --batch_size 16 \
    --input_dir_LR ./data/LR_test/ \
    --input_dir_HR ./data/HR_test/ \
    --num_resblock 16 \
    --perceptual_mode MSE \
    --pre_trained_model True \
    --checkpoint ./experiment_SRResnet/model-180000

