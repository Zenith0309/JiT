#!/bin/bash
# =============================================================================
# JiT Training Script for 2×RTX 4090 - Small Dataset / Quick Validation
# =============================================================================
# This script is designed for quick validation and debugging on smaller datasets.
# Use this to verify your setup works before running full training.
# =============================================================================

# Exit on error
set -e

# Default paths - modify these or set as environment variables
DATA_PATH=${DATA_PATH:-"./data/small_dataset"}
OUTPUT_DIR=${OUTPUT_DIR:-"./output_2x4090_small"}
CLASS_NUM=${CLASS_NUM:-100}  # e.g., CIFAR-100 style dataset

# Create output directory
mkdir -p ${OUTPUT_DIR}

echo "=============================================="
echo "JiT Quick Validation Training on 2×RTX 4090"
echo "=============================================="
echo "Data Path: ${DATA_PATH}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "Number of Classes: ${CLASS_NUM}"
echo "=============================================="

torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 16 \
--gradient_accumulation_steps 2 \
--blr 5e-5 \
--epochs 50 \
--warmup_epochs 2 \
--eval_freq 10 \
--gen_bsz 16 \
--num_images 1000 \
--cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--num_workers 4 \
--class_num ${CLASS_NUM} \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${DATA_PATH} --online_eval

echo "=============================================="
echo "Quick validation training completed!"
echo "=============================================="
