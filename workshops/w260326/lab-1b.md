# LAB 1.b — Run Inference Without vLLM (Baseline)

## Estimated Time
15–20 minutes

---

## Objective

By the end of this lab, you will:

- Download and run Meta-Llama-3.1-8B-Instruct (or Qwen2.5-7B-Instruct) on a single GPU 
- Run inference using Hugging Face Transformers (without vLLM)
- Understand baseline (non-optimized) inference behavior
- Measure memory usage and latency

> **Note:** If you have not received HuggingFace approval for the gated Llama models, you can use **Qwen2.5-7B-Instruct** as an alternative. Instructions for both models are provided below.

---

## Why This Matters

This is your **baseline**.

Later, we will compare this with **vLLM**, which improves:

- Throughput (requests per second)
- Memory usage (via PagedAttention)
- Latency (via continuous batching)

---

## 1. Allocate a GPU Node

Request an interactive session with a single GPU:

```bash
salloc -p kempner_eng --reservation=inference_workshop    --nodes=1 --ntasks=1   --cpus-per-task=32   --mem=256G   --gres=gpu:1   -t 00-8:00:00
```
> [!NOTE]
> If you don't have access to the `kempner_eng` partition, please use the appropriate partition available to you and adjust the command accordingly.


## 2. Activate Your Environment

Activate the vLLM environment (we'll use its dependencies, but not vLLM itself yet):

```bash
cd /path/to/your/environment
source .venv/bin/activate
```

If you don't have `accelerate` installed, add it now:

```bash
uv pip install accelerate
```

## 3. Download the Model and Run Inference

We'll use the `llm_inference_1b.py` script to run baseline inference.

### Option A: Meta-Llama-3.1-8B-Instruct (Recommended)

If you have received HuggingFace approval for Llama models, use this option.

First, review the script:

```bash
cat scripts/llm_inference_1b.py
```

The script performs:
- Model download from Hugging Face
- Single GPU inference using transformers
- Memory and latency measurements
- Text generation with a sample prompt

Run the script (limiting to a single GPU):

```bash
module load gcc/13.2.0-fasrc01 
CUDA_VISIBLE_DEVICES=0 python scripts/llm_inference_1b.py
```

### Option B: Qwen2.5-7B-Instruct (Alternative)

If you have **not** received HuggingFace approval for Llama models, use Qwen2.5-7B-Instruct instead.

Edit `scripts/llm_inference_1b.py` and change the model name:

```python
# Change this line:
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

# To this:
model_name = "Qwen/Qwen2.5-7B-Instruct"
```

Also update the print statement to reflect the correct model:

```python
# Change this line:
print("LAB 1.b: Baseline Inference with Meta-Llama-3.1-8B-Instruct")

# To this:
print("LAB 1.b: Baseline Inference with Qwen2.5-7B-Instruct")
```

Then run the script:

```bash
module load gcc/13.2.0-fasrc01 
CUDA_VISIBLE_DEVICES=0 python scripts/llm_inference_1b.py
```

**Note:** `CUDA_VISIBLE_DEVICES=0` ensures only GPU 0 is used for this baseline test.

## 4. Observe GPU Utilization

While the script is running, open another terminal and SSH to the same node:

```bash
ssh $SLURM_NODELIST
nvtop
```

**What to observe:**
- GPU memory usage (should be ~16-20GB for 8B model, ~14-18GB for 7B model)
- GPU utilization percentage
- Time to first token
- Total generation time

## 5. Try Different Prompts

Edit `scripts/llm_inference_1b.py` and modify the prompt:

```python
prompt = "Your custom prompt here"
```

Run again and observe:
- Does generation time change with prompt length?
- How does output length affect total time?

## 6. Experiment with Parameters

Try changing generation parameters in the script:

```python
max_new_tokens=100  # Try 50, 100, 200
temperature=0.7     # Try 0.1 (deterministic) or 1.0 (creative)
```

## What You Should Notice

- **Slow first token**: Time to generate the first token can be several seconds
- **High memory usage**: 7-8B models require significant GPU memory
- **Sequential processing**: Each token is generated one at a time
- **No concurrent requests**: Cannot handle multiple requests simultaneously
- **GPU underutilization**: GPU may not be fully utilized during generation

## Limitations of This Approach

1. **No batching**: Processes one request at a time
2. **Memory inefficiency**: KV cache grows linearly with sequence length
3. **Low throughput**: Cannot handle concurrent requests
4. **Static memory allocation**: Cannot dynamically adjust memory
5. **No optimization**: Missing PagedAttention, continuous batching, etc.

These are the problems that **vLLM** solves, which we'll explore in LAB 2.

---

## Summary

You've successfully run baseline inference with a 7-8B model on a single GPU. You now understand the limitations that modern inference engines like vLLM are designed to address.

**Key takeaways:**
- Simple transformers inference works but has significant limitations
- Memory usage is high and inefficient
- Throughput is limited to processing requests sequentially
- GPU utilization is suboptimal

