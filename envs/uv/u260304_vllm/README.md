# vLLM Environment Setup Guide

Complete guide for setting up a production-ready vLLM inference environment with CUDA 12.9 support.

## Overview

This directory contains everything needed to create a reproducible vLLM environment for LLM inference on GPU clusters.

**What's included:**
- vLLM 0.11.2 with CUDA 12.9 support
- PyTorch 2.9.0+cu129
- NCCL 2.27.5 for multi-GPU communication
- All dependencies with exact versions
- Test scripts to verify installation
- Multi-GPU configuration guides

**Tested on:**
- Python 3.12.11
- CUDA 12.9.1
- GCC 13.2.0
- Rocky Linux 8 / RHEL 8

---

## Quick Start

1. Create and activate virtual environment

> [!Warning]  
> `uv` comes with significant amount of files. So it is important to use common cache directory to avoid hitting quota issues and to avoid unnecessary downloads. Please set your cache directory to a common location (e.g. in your lab directory or provided by the workshop) by setting the `UV_CACHE_DIR` environment variable before running `uv` commands.

> [!IMPORTANT]
> Load the FASRC `python/3.12.11-fasrc02` module **before** creating the venv. `uv`'s default standalone Python omits the C development headers (`Python.h`), which breaks `torch.compile` at runtime. See [Issue: `torch.compile` fails with `Python.h: No such file or directory`](#issue-torchcompile-fails-with-pythonh-no-such-file-or-directory) for details.

```bash
module load python/3.12.11-fasrc02          # System Python with C dev headers for torch.compile
export UV_CACHE_DIR=<your cache directory>  # Set cache directory to avoid quota issues
uv venv vllm_env --python $(which python) --seed
source vllm_env/bin/activate
```

2. Install packages

```bash
uv pip install -r requirements-frozen.txt 
```

3. Run tests to verify installation

```bash
python test_vllm_installation.py
```

4. Run inference test (downloads small model)

```bash
python test_vllm_inference.py
```


> [!NOTE]  
> The previous steps work because the `requirements-frozen.txt` file contains exact versions of all packages, including vLLM 0.11.2 and its dependencies for the workshop. For any new project you will need to create your own environment and install the correct versions of vLLM and dependencies which are explained in the detailed instructions below.


---

## Comprehensive Guide for setting up a vLLM environment

### Files in This Directory

| File | Purpose |
|------|---------|
| **README.md** | This guide (you are here) |
| **pyproject.toml** | Project configuration for uv |
| **uv.lock** | Lock file with exact package versions (724KB) |
| **requirements-frozen.txt** | Alternative pip-compatible requirements |
| **test_vllm_installation.py** | Quick installation verification (no model) |
| **test_vllm_inference.py** | Full inference test with small model |
| **check_nccl_status.py** | Check NCCL and multi-GPU status |
| **nccl_and_multi_gpu.md** | Detailed multi-GPU setup guide |

---

### Detailed Setup Instructions

#### Step 1: Prepare Your Environment

##### On HPC Clusters

Loading modules is only necessary if you need to compile dependencies; otherwise, you don’t need to load them.

```bash
# Request a GPU node (adjust for your scheduler)
# SLURM example:
salloc -N 1 --gres=gpu:1 -t 2:00:00

# Load modules (IMPORTANT: Use GCC 13, not 15!)
module purge
module load gcc/13.2.0-fasrc01      # GCC 13 required for CUDA 12.9
module load cuda/12.9.1-fasrc01     # CUDA 12.9
module load cudnn/9.10.2.21_cuda12-fasrc01
module load cmake
```

##### Install uv (if not available)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

#### Step 2: Create Virtual Environment

> [!IMPORTANT]
> The FASRC `python/3.12.11-fasrc02` module is **required** here (not just during compilation in Step 1). `uv`'s default standalone Python lacks `Python.h`, which `torch.compile` needs at runtime to build CUDA utility shims. Loading the FASRC module before `uv venv` bakes the correct interpreter — and its dev headers — into the venv's `sysconfig`.

```bash
# Load system Python with C development headers
module load python/3.12.11-fasrc02

# Create environment against that Python
uv venv vllm_env --python $(which python) --seed

# Activate it
source vllm_env/bin/activate

# Optional: Set custom cache location to avoid quota issues
export UV_CACHE_DIR=$PWD/.uv_cache
```

#### Step 3: Install Dependencies

```bash
# Build tools (required even with wheels)
uv pip install ninja packaging wheel setuptools

# Install vLLM with CUDA 12.9 support
uv pip install "vllm==0.11.2" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
```

**Why these flags?**
- `--extra-index-url`: Gets CUDA-enabled PyTorch wheels
- `--index-strategy unsafe-best-match`: Allows resolving setuptools from PyPI

---

### Testing Your Installation

#### Test 1: Quick Verification (Recommended First)

```bash
python test_vllm_installation.py
```

**Expected output:**
```
vLLM imported successfully (version 0.11.2)
PyTorch imported successfully (version 2.9.0+cu129)
  CUDA available: True
  CUDA version: 12.9
  GPU count: 1
  GPU 0: NVIDIA H200
Transformers imported successfully (version 4.57.6)
All tests passed!
```

**What it checks:**
- vLLM, PyTorch, Transformers imports
- CUDA availability
- GPU detection
- Basic vLLM functionality

#### Test 2: Full Inference Test

```bash
python test_vllm_inference.py
```

**What it does:**
- Downloads facebook/opt-125m model (~250MB, first run only)
- Runs inference on 3 test prompts
- Displays generated text

**Sample output:**
```
Prompt 1: Hello, my name is
Generated: John and I am a software engineer...

Prompt 2: The capital of France is
Generated: Paris, which is known for...

Inference completed successfully!
```

#### Test 3: Check NCCL and Multi-GPU Status

```bash
python check_nccl_status.py
```

**Shows:**
- NCCL availability
- GPU count and names
- Multi-GPU recommendations
- Example tensor parallelism configs

---

### Reproducibility

#### Option 1: Using uv.lock (Recommended)

The `uv.lock` file contains exact versions of all packages with checksums.

**To recreate this exact environment:**

```bash
# Create new environment
uv venv new_env --python 3.12
source new_env/bin/activate

# Install from lock file
uv sync --index-strategy unsafe-best-match
```

#### Option 2: Using requirements-frozen.txt

Simpler, pip-compatible format with exact versions.

```bash
# Create new environment
uv venv new_env --python 3.12
source new_env/bin/activate

# Install from frozen requirements
uv pip install -r requirements-frozen.txt
```

---

### Multi-GPU Setup

#### Single GPU (Current Default)

```python
from vllm import LLM

# Automatically uses single GPU
llm = LLM(model="facebook/opt-13b")
```

**Characteristics:**
- No NCCL communication needed
- Simpler setup
- Limited by single GPU memory

#### Multi-GPU with Tensor Parallelism

```bash
# Request multiple GPUs
salloc -N 1 --gres=gpu:4 -t 2:00:00
```

```python
from vllm import LLM

# Automatically uses NCCL for communication
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # Split across 4 GPUs
)
```

**Benefits:**
- Run much larger models
- Higher throughput
- NCCL handles all GPU communication

**See [nccl_and_multi_gpu.md](nccl_and_multi_gpu.md) for detailed multi-GPU guide.**

---

### Troubleshooting

#### Issue: GCC version error during installation

```
error: -- unsupported GNU version! gcc versions later than 14 are not supported!
```

**Solution:** Use GCC 13, not GCC 15
```bash
module load gcc/13.2.0-fasrc01  # NOT gcc/15.x
```

#### Issue: `torch.compile` fails with `Python.h: No such file or directory`

```
/tmp/.../cuda_utils.c:6:10: fatal error: Python.h: No such file or directory
    6 | #include <Python.h>
```

**Cause:** The venv was created against `uv`'s default standalone Python build (from [python-build-standalone](https://github.com/astral-sh/python-build-standalone)), which ships without CPython's C development headers. `torch.compile`'s Inductor backend builds a small C extension (`cuda_utils.c`) at runtime that `#include <Python.h>`, and the interpreter's `sysconfig` include dir must contain that header.

**Solution:** Recreate the venv against the FASRC Python module, which is source-built with `--enable-shared` and installs the full dev layout (`include/python3.12/Python.h`, `libpython3.12.so`):

```bash
module purge
module load python/3.12.11-fasrc02
rm -rf vllm_env                                         # delete the broken venv
uv venv vllm_env --python $(which python) --seed
source vllm_env/bin/activate
uv pip install -r requirements-frozen.txt
```

**Quick workaround without recreating the venv:** disable `torch.compile` by passing `enforce_eager=True` to `LLM(...)` or `--enforce-eager` to `vllm serve`. This avoids the failure but forfeits the compilation speedups.

#### Issue: vLLM 0.2.5 installed instead of 0.11.2

**Cause:** Not specifying version explicitly

**Solution:** Use exact version
```bash
uv pip install "vllm==0.11.2" --extra-index-url https://download.pytorch.org/whl/cu129 --index-strategy unsafe-best-match
```

#### Issue: CUDA not available

```python
torch.cuda.is_available()  # Returns False
```

**Checks:**
1. Are you on a GPU node? `nvidia-smi`
2. Are CUDA modules loaded? `module list`
3. Is CUDA visible? `echo $CUDA_VISIBLE_DEVICES`

#### Issue: Out of memory

**Solutions:**
1. Use smaller model or batch size
2. Enable quantization (8-bit/4-bit)
3. Reduce `gpu_memory_utilization`
4. Use multiple GPUs with tensor parallelism

#### Issue: ImportError for vLLM

**Cause:** Wrong Python environment

**Solution:** Ensure virtual environment is activated
```bash
source vllm_env/bin/activate
which python  # Should show vllm_env/bin/python
```

---

### Package Versions

Key packages installed:

| Package | Version | Purpose |
|---------|---------|---------|
| vllm | 0.11.2 | LLM inference engine |
| torch | 2.9.0+cu129 | Deep learning framework |
| transformers | 4.57.6 | HuggingFace models |
| nvidia-nccl-cu12 | 2.27.5 | Multi-GPU communication |
| xformers | 0.0.33.post1 | Memory-efficient attention |
| flashinfer-python | 0.5.2 | Fast attention kernels |

**Full list:** See `requirements-frozen.txt`

---

### Usage Examples

#### Basic Inference

```python
from vllm import LLM, SamplingParams

# Initialize model
llm = LLM(model="facebook/opt-125m")

# Set parameters
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.95,
    max_tokens=100
)

# Generate
prompts = ["Hello, how are you?"]
outputs = llm.generate(prompts, sampling_params)

print(outputs[0].outputs[0].text)
```

#### With Quantization (for larger models)

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    quantization="awq",  # or "gptq"
    gpu_memory_utilization=0.9
)
```

#### Multi-GPU Tensor Parallelism

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # Uses NCCL
    max_model_len=4096
)
```

---

### Additional Resources

- **vLLM Documentation:** https://docs.vllm.ai/
- **Multi-GPU Guide:** [nccl_and_multi_gpu.md](nccl_and_multi_gpu.md)
- **Issue Tracker:** https://github.com/vllm-project/vllm/issues

---

### Contributing

To update this environment:

1. Make changes to your environment
2. Update lock file:
   ```bash
   uv lock --index-strategy unsafe-best-match
   ```
3. Update frozen requirements:
   ```bash
   uv pip freeze > requirements-frozen.txt
   ```
4. Test with both installation methods
5. Update this README with any new instructions

---

### License

This setup guide is part of the distributed-inference-vllm project. See [LICENSE](../../../LICENSE) for details.

---

## Maintainer

- Created by: Naeem Khoshnevis
- Date: 2026-03-04
- Last updated: 2026-03-04

---

### Summary

**You now have a production-ready vLLM environment:**

- vLLM 0.11.2 with CUDA 12.9 support
- Fully reproducible with lock files
- Tested and verified
- Multi-GPU ready (NCCL included)
- Documentation and troubleshooting guides

**Next steps:**
1. Run tests to verify: `python test_vllm_installation.py`
2. Try your first inference: `python test_vllm_inference.py`
3. Explore multi-GPU: `python check_nccl_status.py`



### TODO
- Check the setup with ordinary virtual environments (venv) and update instructions if needed
- Add steps for building Docker image + converting to Singularity
- Add instructions on compiling the latest vLLM from source with CUDA 12.9 support (for bleeding-edge users)