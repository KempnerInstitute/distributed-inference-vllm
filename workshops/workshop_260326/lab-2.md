# LAB 2 — Optimized Inference with vLLM

## Estimated Time
30–40 minutes

---

## Objective

By the end of this lab, you will:

- Understand how vLLM improves upon baseline transformers inference
- Run batch offline inference with vLLM on a single GPU (Qwen2.5-32B-Instruct)
- Scale to multi-GPU inference using tensor parallelism (Llama-3.1-70B)
- Compare performance metrics: throughput, memory usage, and latency
- Learn about PagedAttention and continuous batching

---

## Why vLLM?

In LAB 1.b, you saw the limitations of baseline inference:
- No batching (one request at a time)
- Inefficient KV cache management
- Low GPU utilization
- Poor throughput

**vLLM solves these problems with:**
- **PagedAttention**: Efficient KV cache memory management
- **Continuous batching**: Process multiple requests concurrently
- **Optimized CUDA kernels**: Better GPU utilization
- **Tensor parallelism**: Scale across multiple GPUs

---

## Part A: Single GPU Inference with vLLM

### 1. Allocate a GPU Node

Request an interactive session with a single GPU:

```bash
salloc -p kempner_eng --reservation=inference_workshop    --nodes=1 --ntasks=1   --cpus-per-task=32   --mem=256G   --gres=gpu:1   -t 00-8:00:00
```

Once allocated, SSH into the node:

```bash
ssh $SLURM_NODELIST
```

### 2. Activate Your Environment

```bash
cd /path/to/your/environment
source .venv/bin/activate
```

### 3. Run vLLM Batch Inference (Single GPU)

We'll use Qwen2.5-32B-Instruct with vLLM in offline batch mode.

Review the script:

```bash
cat scripts/vllm_inference_2a.py
```

The script performs:
- Batch processing of multiple prompts
- vLLM offline inference mode
- Performance metrics collection
- Comparison with baseline approach

Run the script:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vllm_inference_2a.py
```

### 4. Observe the Output

**What to notice:**
- Multiple prompts processed efficiently in a batch
- Higher throughput compared to baseline (LAB 1.b)
- Better GPU memory utilization (PagedAttention)
- Faster time to first token
- Overall lower latency per request

### 5. Experiment

Try modifying the script to:
- Increase the number of prompts (change `num_prompts`)
- Adjust `max_tokens` generation length
- Change sampling parameters (`temperature`, `top_p`)

---

## Part B: Multi-GPU Inference with Tensor Parallelism

### 1. Allocate a Node with Multiple GPUs

Exit your current session (if in one) and request 2 GPUs:

```bash
exit  # if needed
salloc -p kempner_eng --reservation=inference_workshop    --nodes=1 --ntasks=1   --cpus-per-task=32   --mem=256G   --gres=gpu:2   -t 00-8:00:00
```

Once allocated, SSH into the node:

```bash
ssh $SLURM_NODELIST
```

### 2. Activate Your Environment

```bash
cd /path/to/your/environment
source .venv/bin/activate
```

### 3. Run vLLM with Tensor Parallelism (2 GPUs)

We'll use Llama-3.1-70B, which requires 2 GPUs with tensor parallelism.

Review the script:

```bash
cat scripts/vllm_inference_2b.py
```

The script demonstrates:
- Tensor parallelism across 2 GPUs
- Automatic model sharding
- Batch inference with a larger model
- Multi-GPU performance metrics

Run the script:

```bash
python scripts/vllm_inference_2b.py
```

### 4. Observe GPU Utilization

In another terminal, monitor both GPUs:

```bash
ssh $SLURM_NODELIST
watch -n 1 nvidia-smi
```

**What to observe:**
- Model sharded across 2 GPUs (~70GB per GPU)
- Both GPUs actively processing during generation
- Communication between GPUs for tensor operations
- Higher aggregate throughput than single GPU

### 5. Understanding Tensor Parallelism

Tensor parallelism splits model layers across multiple GPUs:
- Each GPU handles a portion of each layer
- GPUs communicate during forward/backward passes
- Enables running models larger than single GPU memory
- Reduces per-GPU memory but adds communication overhead

---

## Key Comparisons

### Single GPU: Baseline vs vLLM (8B model)

| Metric | Baseline (LAB 1.b) | vLLM (LAB 2.a) | Improvement |
|--------|-------------------|----------------|-------------|
| Throughput | Low (sequential) | High (batching) | ~5-10x |
| Memory Efficiency | Poor (static KV) | Good (PagedAttention) | ~2x |
| Concurrent Requests | 1 | Multiple | N/A |

### Single GPU vs Multi-GPU (vLLM)

| Metric | 32B (1 GPU) | 70B (2 GPUs) | Notes |
|--------|-------------|--------------|-------|
| Model Size | ~64GB | ~140GB | Requires multi-GPU |
| Memory per GPU | ~64GB | ~70GB each | Tensor parallelism splits |
| Throughput | High | Lower per GPU | Communication overhead |
| Model Quality | Good | Better | Larger model capacity |

---

## Summary

You've now experienced:

**Part A - vLLM Optimization:**
- Dramatic improvements over baseline transformers
- Efficient batch processing
- PagedAttention for better memory usage

**Part B - Scaling with Tensor Parallelism:**
- Running models larger than single GPU memory
- Automatic model sharding across GPUs
- Performance trade-offs of distributed inference

**Key Takeaways:**
- vLLM provides significant performance improvements for production inference
- Tensor parallelism enables scaling to larger models
- Trade-offs exist between model size, GPU count, and throughput
- These techniques are essential for deploying LLMs at scale

In LAB 3, we'll deploy vLLM as a production server with API endpoints!
