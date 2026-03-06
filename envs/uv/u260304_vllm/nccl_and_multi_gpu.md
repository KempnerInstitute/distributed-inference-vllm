# NCCL and Multi-GPU Usage with vLLM

## Current Setup

Your environment **has NCCL installed** (`nvidia-nccl-cu12 2.27.5`), but it's not being used because you're running on a **single GPU**.

```
GPU 0: NVIDIA H200 (143GB VRAM)
```

## When NCCL is Used

NCCL (NVIDIA Collective Communications Library) enables efficient multi-GPU communication. vLLM automatically uses NCCL when:

### 1. **Tensor Parallelism (Same Node, Multiple GPUs)**

Split a large model across multiple GPUs on one node:

```python
from vllm import LLM

# Example: Using 4 GPUs with tensor parallelism
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # Split across 4 GPUs
)
```

**When used:**
- Model is too large for single GPU
- Multiple GPUs available on same node
- NCCL handles inter-GPU communication

### 2. **Pipeline Parallelism**

Different layers on different GPUs (less common in inference):

```python
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    pipeline_parallel_size=2,
    tensor_parallel_size=2,
)
```

### 3. **Multi-Node Inference**

Distribute model across GPUs on multiple nodes:

```bash
# On each node, vLLM uses NCCL for cross-node communication
# Requires Ray for orchestration
```

## Why You Don't See NCCL Usage

With **1 GPU**, vLLM runs in single-GPU mode:
- All model weights fit in one GPU (or use quantization)
- No inter-GPU communication needed
- NCCL library is installed but idle

## Testing Multi-GPU with NCCL

To test NCCL usage, you need:

### Option 1: Request Multiple GPUs on Same Node

```bash
# SLURM example
salloc -N 1 --gres=gpu:4 -t 1:00:00  # 4 GPUs, 1 node
```

Then run with tensor parallelism:

```python
from vllm import LLM

# This will use NCCL for GPU communication
llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=4,
)

outputs = llm.generate(["Hello world"], max_tokens=20)
print(outputs[0].outputs[0].text)
```

### Option 2: Use a Larger Model

Even with 1 GPU, if you try a model larger than your VRAM:

```python
# This would fail or need quantization on single H200
llm = LLM(model="meta-llama/Llama-2-405b")  # Too large!

# But with multiple GPUs it works:
llm = LLM(
    model="meta-llama/Llama-2-405b",
    tensor_parallel_size=8,  # Split across 8 GPUs
)
```

## Verifying NCCL Installation

Your environment is ready for multi-GPU:

```python
import torch
print(f"NCCL available: {torch.cuda.nccl.is_available()}")  
print(f"NCCL version: {torch.cuda.nccl.version()}")
```

Or check packages:

```bash
pip list | grep nccl
# Output: nvidia-nccl-cu12  2.27.5
```

## Performance Considerations

### Single GPU (Your Current Setup)
- **Pros:** Simple, no communication overhead
- **Cons:** Limited by single GPU memory
- **Good for:** Small to medium models (< 70B parameters)

### Multi-GPU with NCCL
- **Pros:** Can run massive models, higher throughput
- **Cons:** Communication overhead, more complex setup
- **Good for:** Large models (70B+ parameters), high-throughput serving

## Example: Enabling NCCL Logging

To see NCCL in action when you have multiple GPUs:

```bash
# Enable NCCL debug logging
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# Run vLLM with tensor parallelism
python -c "
from vllm import LLM
llm = LLM(model='facebook/opt-6.7b', tensor_parallel_size=2)
"
```

You'll see output like:
```
NCCL INFO Bootstrap : Using eth0:192.168.1.1<0>
NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [1]mlx5_1:1/IB
NCCL INFO Channel 00/02 : 0[0] -> 1[1] via P2P/IPC
```

## Summary

| Setup | NCCL Used? | Your Case |
|-------|------------|-----------|
| 1 GPU | ❌ No | ✅ **Current** |
| 2-8 GPUs (same node) | ✅ Yes | Needs multi-GPU allocation |
| Multiple nodes | ✅ Yes | Needs cluster setup |

Your environment is **NCCL-ready**. You just need to request multiple GPUs to see it in action!
