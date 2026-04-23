# vLLM Environment Setup Guide (Python-only build from source)

Complete guide for setting up a Python-only build (without compilation) of vLLM. 


## Overview 

This directory contains everything needed to create a reproducible vLLM environment for LLM inference on GPU clusters.

**What's included:**
- vLLM                                     0.18.1rc1.dev101+ga32783bb3.precompiled
- torch                                    2.10.0
- nvidia-cublas-cu12                       12.8.4.1
- nvidia-cuda-cupti-cu12                   12.8.90
- nvidia-cuda-nvrtc-cu12                   12.8.93
- nvidia-cuda-runtime-cu12                 12.8.90
- nvidia-cudnn-cu12                        9.10.2.21
- nvidia-cudnn-frontend                    1.18.0
- nvidia-cufft-cu12                        11.3.3.83
- nvidia-cufile-cu12                       1.13.1.3
- nvidia-curand-cu12                       10.3.9.90
- nvidia-cusolver-cu12                     11.7.3.90
- nvidia-cusparse-cu12                     12.5.8.93
- nvidia-cusparselt-cu12                   0.7.1
- nvidia-cutlass-dsl                       4.4.2
- nvidia-cutlass-dsl-libs-base             4.4.2
- nvidia-ml-py                             13.595.45
- nvidia-nccl-cu12                         2.27.5
- nvidia-nvjitlink-cu12                    12.8.93
- nvidia-nvshmem-cu12                      3.4.5
- nvidia-nvtx-cu12                         12.8.90

----

## Guide for setting up the vLLM environment from source

1. Get the code:

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a32783bb35c6120df88d57609c8e454f9131f0a1
```

Last tested on March 24, 2026. 

2. Install editable vllm package:

> [!IMPORTANT]
> Load the FASRC `python/3.12.11-fasrc02` module **before** creating the venv. `uv`'s default standalone Python omits the C development headers (`Python.h`), which breaks `torch.compile` at runtime (e.g. `fatal error: Python.h: No such file or directory` while Inductor builds `cuda_utils.c`). Pointing `uv` at the FASRC Python — which includes the full dev layout — avoids this.

```bash
module load python/3.12.11-fasrc02
uv venv --python $(which python)
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

3. Verify the installation:

```bash
module load gcc/13.2.0-fasrc01 
python -c "import vllm; print(vllm.__version__)"
```

You should see the following output:

```bash
0.18.1rc1.dev101+ga32783bb3
```

4. Test with offline inference:

First, allocate a GPU node. The following examples are based on one node with 4 GPUs. 

```bash
python test_vllm_gpu.py
```

5. Test with server deployment:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --tensor-parallel-size 2
```

Now, let's say your server is running on `holygpu8a10101`, from any other node, you can run the following curl command to verify the server is working:

```bash
curl http://holygpu8a10101:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain GPU inference in two paragraphs."}
    ],
    "max_tokens": 1000
  }'
```

6. Test with combined tensor and pipeline parallelism:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve Qwen/Qwen2.5-0.5B-Instruct \
  --pipeline-parallel-size 2 \
  --tensor-parallel-size 2
```

You can verify the server using the same curl command as in step 5.

---

## Notes

- This setup uses precompiled binaries to avoid compilation time
- Ensure your CUDA drivers are compatible with CUDA 12.8

You can try with the same curl command as above to verify the server is working.


## Maintainer

- Created by: Naeem Khoshnevis
- Date: 2026-03-24
- Last updated: 2026-03-24