# Qwen2.5-32B-Instruct - Single GPU Inference

## Overview

This workflow demonstrates how to run inference using the Qwen2.5-32B-Instruct model with vLLM on a single GPU. The workflow includes basic inference examples and performance benchmarking on GPU clusters.

## Environment

**Environment used:**
```
envs/uv/u260304_vllm
```

**Repository commit:**
```
cf2de1a3815c10c59e33349cd81aef64e68ea0ae
```

## Model Information

**Model:** Qwen/Qwen2.5-32B-Instruct

**HuggingFace Link:** https://huggingface.co/Qwen/Qwen2.5-32B-Instruct

**Model Size:** 32B parameters

**License:** Apache 2.0

**Precision:** bfloat16

**Context Length:** 4096 tokens (configurable)

## Hardware Configuration

- **GPU Type:** NVIDIA A100 80GB or H100
- **Number of GPUs:** 1
- **GPU Memory Required:** ~62.14 GiB (61.04 GiB model + 1.10 GiB KV cache)
- **Disk Space:** ~65 GB (62 GB model + 2-3 GB compile cache)

**Resource Requirements:**

| Resource | Requirement | Notes |
|----------|------------|-------|
| **GPU Memory** | 61.04 GiB | Model loading only |
| **Total GPU Memory** | ~62.14 GiB | Model + KV cache (1.10 GiB) |
| **Disk Space (Model)** | 62 GB | Model checkpoint storage |
| **Disk Space (Compile Cache)** | ~2-3 GB | torch.compile cache |
| **Total Disk Space** | ~65 GB | Model + cache |

## Parallelism Configuration

- **Tensor Parallel Size:** 1 (single GPU)
- **Pipeline Parallel Size:** N/A
- **Total Parallel Size:** 1

**Note:** This workflow uses a single GPU configuration. For larger models or multi-GPU setups, see related workflows.

## Model Information

## Prerequisites

### 1. Environment Setup

Activate the vLLM environment (see [Environment](#environment) section above):

```bash
# Navigate to environment directory
cd envs/uv/u260304_vllm

# Create and activate virtual environment (first time only)
export UV_CACHE_DIR=<your-cache-directory>
uv venv vllm_env --python 3.12 --seed
source vllm_env/bin/activate

# Install packages (first time only)
uv pip install -r requirements-frozen.txt

# For subsequent uses, just activate:
source vllm_env/bin/activate
```

See the environment's [README](../../envs/uv/u260304_vllm/README.md) for detailed setup instructions.

### 2. Model Access

The model will be automatically downloaded from HuggingFace on first run. Ensure you have:

- **Internet access** for model download
- **Disk space:** ~65 GB available
- **HuggingFace cache:** Set `HF_HOME` environment variable to specify cache location

```bash
# Set HuggingFace cache directory (recommended for HPC clusters)
export HF_HOME=/path/to/your/cache/directory
```

### 3. Model Checkpoint Storage

Model checkpoints are automatically downloaded and cached by Hugging Face:

```
Default location: ~/.cache/huggingface/hub/models--Qwen--Qwen2.5-32B-Instruct
Size: 62 GB
Format: safetensors (17 shard files)
```

The checkpoint location is determined by the `HF_HOME` environment variable.

### First Run (Cold Start - Model Download Required)
- **Download Time**: ~491 seconds (8.2 minutes)
- **Model Loading**: 110.23 seconds
- **Total Initialization**: ~602 seconds (10 minutes)
- **Torch Compile**: 59.26 seconds
- **Graph Capture**: 29 seconds
- **Inference Speed**: 
  - Input: 5.93 tokens/s
  - Output: 142.41 tokens/s
- **Inference Time**: 5.45 seconds (3 prompts)

### Subsequent Runs (Model Cached)
- **Model Loading**: 7-12 seconds
- **Torch Compile** (cached): 9-56 seconds (uses cached graphs)
- **Graph Capture**: 25 seconds
- **Total Initialization**: 55-106 seconds
- **Inference Speed**:
  - Input: 4.99-9.30 tokens/s
  - Output: 95.79-190.55 tokens/s
- **Inference Time**: 5.38-8.16 seconds (3-4 prompts)

## KV Cache Configuration

```
GPU KV cache size: 239,152 tokens
Maximum concurrency for 4,096 tokens per request: 58.39x
```

This means the model can handle approximately 58 concurrent requests of 4,096 tokens each in the KV cache.

The checkpoint location is determined by the `HF_HOME` environment variable.

## Step-by-Step Instructions

### 1. Environment Activation

Activate the vLLM environment (see Prerequisites above or [environment documentation](../../envs/uv/u260304_vllm/README.md)).

### 2. Navigate to Workflow Directory

```bash
cd workflows/Qwen2.5-32B-Instruct_single-gpu-inference
```

### 3. Run Basic Inference

```bash
python simple_inference_test.py
```

**What this does:**
- Loads the Qwen2.5-32B-Instruct model
- Processes 3 sample prompts
- Generates text responses
- Reports timing and performance metrics

**Expected output:**
```
Loading model: Qwen/Qwen2.5-32B-Instruct...
Generating responses...
--------------------------------------------------
Prompt: Explain the theory of relativity in simple terms...
Response: [Generated text]
--------------------------------------------------
Inference completed in ~5-8 seconds.
```

### 4. Run Interactive Inference (Optional)

```bash
python interactive_inference.py
```

### 5. Customize Configuration

Edit parameters in `simple_inference_test.py`:

```python
# Model configuration
model_id = "Qwen/Qwen2.5-32B-Instruct"
tensor_parallel_size = 1  # Number of GPUs
max_model_len = 4096      # Maximum context length

# Sampling parameters
temperature = 0.8         # Sampling temperature (0.0-1.0)
top_p = 0.95             # Nucleus sampling threshold
max_tokens = 256         # Maximum tokens to generate
```

## Performance Notes

### Performance Metrics

**First Run (Cold Start - Model Download Required):**
- **Download Time:** ~491 seconds (8.2 minutes)
- **Model Loading:** 110.23 seconds
- **Total Initialization:** ~602 seconds (10 minutes)
- **Torch Compile:** 59.26 seconds
- **Inference Speed:** 
  - Input: 5.93 tokens/s
  - Output: 142.41 tokens/s

**Subsequent Runs (Model Cached):**
- **Model Loading:** 7-12 seconds
- **Torch Compile** (cached): 9-56 seconds
- **Total Initialization:** 55-106 seconds
- **Inference Speed:**
  - Input: 4.99-9.30 tokens/s
  - Output: 95.79-190.55 tokens/s
- **Inference Time:** 5.38-8.16 seconds (3-4 prompts)

### KV Cache Configuration

```
GPU KV cache size: 239,152 tokens
Maximum concurrency for 4,096 tokens per request: 58.39x
```

This means the model can handle approximately 58 concurrent requests of 4,096 tokens each in the KV cache.

### Performance Recommendations

1. **FlashInfer:** Consider installing FlashInfer for optimal top-p/top-k sampling performance:
   ```bash
   pip install flashinfer
   ```

2. **Batch Size:** For multiple prompts, processing in batches can significantly improve throughput

3. **Context Length:** Reducing `max_model_len` can free up memory for larger batch sizes

4. **Caching:** The model uses prefix caching - repeated prompt prefixes will be faster

### Memory Optimization

- **KV Cache:** Allocated 1.10 GiB for key-value cache
- **Model:** 61.04 GiB for model weights
- **Total:** ~62.14 GiB GPU memory required

## Technical Details

### vLLM Configuration

The model is initialized with the following vLLM settings:

```python
LLM(
    model=model_id,
    tensor_parallel_size=1,
    max_model_len=4096,
    trust_remote_code=True
)
```

**Key Features Enabled:**
- **Chunked Prefill:** Enabled with `max_num_batched_tokens=16384`
- **Prefix Caching:** Enabled for improved performance on repeated prefixes
- **Flash Attention:** Using Flash Attention backend on V1 engine
- **CUDA Graphs:** Enabled for optimized execution
- **Torch Compile:** Level 3 compilation with Inductor backend

### Model Features

The Qwen2.5-32B-Instruct model supports multiple tasks:
- `generate` (default): Text generation
- `embed`: Embedding generation
- `reward`: Reward modeling
- `classify`: Classification
- `score`: Scoring

### Model Loading Process

1. **Download:** Model weights are downloaded as 17 safetensors files (~3.9GB each)
2. **Loading:** Weights are loaded into GPU memory
3. **Compilation:** Model graphs are compiled with torch.compile
4. **Warmup:** CUDA graphs are captured for various batch sizes

## Troubleshooting

### Common Issues

**1. Out of Memory:**
- **Solution:** Reduce `max_model_len` or use a GPU with more memory
- Example: Set `max_model_len=2048` for lower memory usage

**2. Slow First Run:**
- **Cause:** Initial run requires downloading 62GB model
- **Expected:** Download takes ~8 minutes, initialization ~10 minutes total
- **Solution:** This is normal; subsequent runs will be much faster

**3. Model Download Fails:**
- **Cause:** Network issues or insufficient disk space
- **Solution:** Check internet connection and ensure 65GB free disk space
- Check `HF_HOME` points to a location with sufficient space

**4. Socket Warning (can be ignored):**
```
WARNING: Address family not supported by protocol
```
- This warning can be safely ignored

**5. Torch Compile Cache Issues:**
- **Location:** `~/.cache/vllm/torch_compile_cache/`
- **Solution:** Clear cache if encountering compilation errors:
  ```bash
  rm -rf ~/.cache/vllm/torch_compile_cache/
  ```

### Log Interpretation

The inference process produces detailed logs showing:
- Model initialization progress
- Download progress (first run only)
- Compilation and warmup times
- Inference throughput metrics
- Token generation statistics

## References

- **Model Card:** https://huggingface.co/Qwen/Qwen2.5-32B-Instruct
- **vLLM Documentation:** https://docs.vllm.ai/
- **Environment:** [u260304_vllm](../../envs/uv/u260304_vllm/README.md)

## Maintainer

- Created by: Naeem Khoshnevis
- Date: 2026-03-04
- Last updated: 2026-03-04
