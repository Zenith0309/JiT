#!/bin/bash
# =============================================================================
# JiT Training Script for 2×RTX 4090 (24GB VRAM each)
# =============================================================================
# This script is optimized for dual RTX 4090 GPUs with memory-efficient settings.
# Expected VRAM usage: ~18-20GB per GPU
# Effective batch size: 24 × 2 × 4 = 192
# =============================================================================

# Exit on error
set -e

# Default paths - modify these or set as environment variables
IMAGENET_PATH=${IMAGENET_PATH:-"./data/imagenet"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_2x4090"}

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "JiT Training on 2×RTX 4090"
echo "=============================================="
echo "ImageNet Path: ${IMAGENET_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "=============================================="

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 24 \
--gradient_accumulation_steps 4 \
--blr 5e-5 \
--epochs 300 \
--warmup_epochs 5 \
--eval_freq 50 \
--gen_bsz 32 \
--num_images 10000 \
--cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--num_workers 8 \
--class_num 1000 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval

echo "=============================================="
echo "Training completed!"
echo "=============================================="
