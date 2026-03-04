#!/bin/bash
# Quick setup script for vLLM environment
# Usage: ./setup.sh

set -e  # Exit on error

echo "========================================"
echo "vLLM Environment Setup Script"
echo "========================================"
echo

# Check if on a GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo " WARNING: nvidia-smi not found. Are you on a GPU node?"
    echo "   This environment requires GPU access."
    read -p "   Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for uv
if ! command -v uv &> /dev/null; then
    echo " uv not found. Please install uv first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

echo "uv found"

# Load modules (adjust for your cluster)
echo
echo "Loading modules..."
if command -v module &> /dev/null; then
    module purge
    module load gcc/13.2.0-fasrc01 2>/dev/null || echo " Could not load gcc/13.2.0-fasrc01"
    module load cuda/12.9.1-fasrc01 2>/dev/null || echo " Could not load cuda/12.9.1-fasrc01"
    module load cudnn/9.10.2.21_cuda12-fasrc01 2>/dev/null || echo " Could not load cudnn"
    module load cmake 2>/dev/null || echo " Could not load cmake"
    echo " Modules loaded"
else
    echo " module command not found. Make sure required modules are available."
fi

# Create virtual environment
ENV_NAME="vllm_env"
echo
echo "Creating virtual environment: $ENV_NAME"
uv venv $ENV_NAME --python 3.12 --seed
echo "Virtual environment created"

# Activate environment
source $ENV_NAME/bin/activate
echo "Environment activated"

# Set cache directory
export UV_CACHE_DIR=$PWD/.uv_cache
mkdir -p .uv_cache
echo "Cache directory set: .uv_cache"

# Install build dependencies
echo
echo "Installing build dependencies..."
uv pip install ninja packaging wheel setuptools
echo "Build dependencies installed"

# Install vLLM
echo
echo "Installing vLLM 0.11.2 (this may take a few minutes)..."
uv pip install "vllm==0.11.2" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

echo
echo "========================================"
echo "Installation Complete!"
echo "========================================"
echo
echo "To activate the environment in the future:"
echo "  source $ENV_NAME/bin/activate"
echo
echo "Run tests:"
echo "  python test_vllm_installation.py    # Quick test"
echo "  python test_vllm_inference.py       # Full inference test"
echo "  python check_nccl_status.py         # Check multi-GPU status"
echo
echo "Deactivate when done:"
echo "  deactivate"
echo
