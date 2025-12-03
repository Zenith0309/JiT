## Just image Transformer (JiT) for Pixel-space Diffusion

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2511.13720-b31b1b.svg)](https://arxiv.org/abs/2511.13720)&nbsp;

<p align="center">
  <img src="demo/visual.jpg" width="100%">
</p>


This is a PyTorch/GPU re-implementation of the paper [Back to Basics: Let Denoising Generative Models Denoise](https://arxiv.org/abs/2511.13720):

```
@article{li2025jit,
  title={Back to Basics: Let Denoising Generative Models Denoise},
  author={Li, Tianhong and He, Kaiming},
  journal={arXiv preprint arXiv:2511.13720},
  year={2025}
}
```

JiT adopts a minimalist and self-contained design for pixel-level high-resolution image diffusion. 
The original implementation was in JAX+TPU. This re-implementation is in PyTorch+GPU.

<p align="center">
  <img src="demo/jit.jpg" width="40%">
</p>

### Dataset
Download [ImageNet](http://image-net.org/download) dataset, and place it in your `IMAGENET_PATH`.

### Installation

Download the code:
```
git clone https://github.com/LTH14/JiT.git
cd JiT
```

A suitable [conda](https://conda.io/) environment named `jit` can be created and activated with:

```
conda env create -f environment.yaml
conda activate jit
```

If you get ```undefined symbol: iJIT_NotifyEvent``` when importing ```torch```, simply
```
pip uninstall torch
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
Check this [issue](https://github.com/conda/conda/issues/13812#issuecomment-2071445372) for more details.

### Training
The below training scripts have been tested on 8 H200 GPUs.

Example script for training JiT-B/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-B/32 on ImageNet 512x512 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/32 \
--proj_dropout 0.0 \
--P_mean -0.8 --P_std 0.8 \
--img_size 512 --noise_scale 2.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

Example script for training JiT-H/16 on ImageNet 256x256 for 600 epochs:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-H/16 \
--proj_dropout 0.2 \
--P_mean -0.8 --P_std 0.8 \
--img_size 256 --noise_scale 1.0 \
--batch_size 128 --blr 5e-5 \
--epochs 600 --warmup_epochs 5 \
--gen_bsz 128 --num_images 50000 --cfg 2.2 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

### Training on 2×RTX 4090 (Local Training)

This section provides optimized configurations for training JiT on consumer GPUs with limited VRAM.

#### Hardware Requirements
- 2× NVIDIA RTX 4090 (24GB VRAM each)
- 64GB+ System RAM recommended
- NVMe SSD for dataset storage

#### Recommended Configuration
| Parameter | Value | Notes |
|-----------|-------|-------|
| Model | JiT-B/32 | Smaller patch size reduces sequence length |
| Image Size | 256×256 | Standard resolution |
| Batch Size | 24 | Per GPU |
| Gradient Accumulation | 4 | Effective batch size = 24 × 2 × 4 = 192 |
| VRAM Usage | ~18-20GB | Per GPU |

#### Quick Start

**Option 1: Use the provided script**
```bash
# Set your ImageNet path
export IMAGENET_PATH=/path/to/imagenet
export OUTPUT_DIR=./output_2x4090

# Run training
bash scripts/train_2x4090.sh
```

**Option 2: Run directly**
```bash
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
--output_dir ${OUTPUT_DIR} --resume ${OUTPUT_DIR} \
--data_path ${IMAGENET_PATH} --online_eval
```

#### Quick Validation (Small Dataset)

For quick testing with smaller datasets (e.g., CIFAR-100 style):
```bash
export DATA_PATH=/path/to/small_dataset
export CLASS_NUM=100

bash scripts/train_2x4090_small.sh
```

#### Memory Optimization Options

If you encounter OOM errors, try these options:

1. **Enable Gradient Checkpointing** (reduces ~30% VRAM, adds ~20% training time):
```bash
--use_gradient_checkpointing
```

2. **Reduce batch size** and increase gradient accumulation:
```bash
--batch_size 16 --gradient_accumulation_steps 6
```

3. **Reduce num_workers** if system RAM is limited:
```bash
--num_workers 4
```

#### New Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--gradient_accumulation_steps` | 1 | Number of gradient accumulation steps |
| `--use_gradient_checkpointing` | False | Enable gradient checkpointing to save VRAM |
| `--compile_model` | False | Use torch.compile optimization (optional) |

#### Expected Training Time
- ~2-3 hours per epoch on 2×RTX 4090
- Full 300 epoch training: ~25-40 days
- Quick validation (50 epochs): ~4-6 days

### Evaluation

Evaluate a trained JiT:
```
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
main_jit.py \
--model JiT-B/16 \
--img_size 256 --noise_scale 1.0 \
--gen_bsz 128 --num_images 50000 --cfg 2.9 --interval_min 0.1 --interval_max 1.0 \
--output_dir ${CKPT_DIR} --resume ${CKPT_DIR} \
--data_path ${IMAGENET_PATH} --evaluate_gen
```

We use a customized [```torch-fidelity```](https://github.com/LTH14/torch-fidelity)
to evaluate FID and IS against a reference image folder or statistics. You can use ```prepare_ref.py```
to prepare the reference image folder, or directly use our pre-computed reference stats
under ```fid_stats```.

### Acknowledgements

We thank Google TPU Research Cloud (TRC) for granting us access to TPUs, and the MIT
ORCD Seed Fund Grants for supporting GPU resources.

### Contact

If you have any questions, feel free to contact me through email (tianhong@mit.edu). Enjoy!
