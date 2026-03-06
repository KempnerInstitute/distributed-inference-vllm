# vLLM Environment (c250609_vllm085)

## Overview

This conda environment provides vLLM 0.8.5.post1 with PyTorch 2.6 and CUDA 12.4 support for deploying large language models, specifically optimized for DeepSeek-R1 (671B parameters) multi-node inference with FP8 precision.

## Specifications

- **Python Version:** 3.12
- **CUDA Version:** 12.4
- **PyTorch Version:** 2.6.0+cu124
- **vLLM Version:** 0.8.5.post1
- **Key Dependencies:**
  - torch: 2.6.0
  - torchvision: 0.21.0
  - torchaudio: 2.6.0
  - vllm: 0.8.5.post1

## Installation

### Prerequisites

- Conda or Mamba package manager
- CUDA 12.4 compatible GPU drivers
- Access to GPU compute nodes (for testing)

### Setup Instructions

#### Option 1: Using environment.yml (Recommended)

```bash
# Create environment from file
mamba env create -f environment.yml

# Or with conda
conda env create -f environment.yml

# Activate environment
conda activate vllm-inference
```

#### Option 2: Using requirements.txt with pip

```bash
# Create a new conda environment with Python 3.12
mamba create -n vllm-inference python=3.12
conda activate vllm-inference

# Install packages
pip install -r requirements.txt
```

#### Option 3: Manual Installation

```bash
# Create and activate environment
mamba create -n vllm-inference python=3.12
conda activate vllm-inference

# Install PyTorch with CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu124

# Install vLLM
pip install vllm==0.8.5.post1
```

### Verification

Test the installation with the following commands:

```bash
# Test Python and PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

# Test vLLM
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"

# Check GPU access
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
python -c "import torch; [print(f'GPU {i}: {torch.cuda.get_device_name(i)}') for i in range(torch.cuda.device_count())]"
```

**Expected output:**
```
PyTorch: 2.6.0+cu124
CUDA available: True
CUDA version: 12.4
vLLM: 0.8.5.post1
GPU count: [number of GPUs]
GPU 0: [GPU name, e.g., NVIDIA H100]
```

## Usage

### Activating the Environment

```bash
conda activate vllm-inference
```

### Running vLLM Server

Example command for multi-node DeepSeek-R1 deployment:

```bash
vllm serve /path/to/model \
  --tensor-parallel-size 16 \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --trust-remote-code \
  --enforce-eager
```

See the [DeepSeek-R1 workflow](../../../workflows/DeepSeek-R1_multinode-server/) for complete deployment instructions.

### Deactivating the Environment

```bash
conda deactivate
```

## Notes

### CUDA Compatibility

This environment requires CUDA 12.4. Ensure your GPU drivers support CUDA 12.4 or later:

```bash
nvidia-smi
```

### Multi-Node Setup

For multi-node deployments (e.g., DeepSeek-R1 on 16×H100 or 8×H200), this environment works with SLURM schedulers and Ray for distributed execution. See related workflows for detailed SLURM scripts and configuration.

### Known Issues

- **Loading Time:** Large models (e.g., DeepSeek-R1 671B) can take 20-40 minutes to load onto GPUs, depending on storage type
- **Memory Requirements:** DeepSeek-R1 requires at least 16×H100 80GB or 8×H200 141GB for FP8 precision

### Tested Hardware

- NVIDIA H100 80GB SXM
- NVIDIA H200 141GB SXM

## Maintainer

- Created by: Max Shad
- Date: 2025-06-09
- Last updated: 2026-03-04
