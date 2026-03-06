# Meta-Llama-3.1-405B-Instruct-FP8 - Multi-node Server

## Overview

This workflow demonstrates how to set up and run the Meta-Llama 3.1 405B model quantized to FP8 precision on a multi-node server environment using vLLM. The FP8 quantization provides significant memory savings and improved throughput compared to FP16/BF16, while maintaining model quality. This workflow covers environment setup, model access, multi-GPU configuration across nodes, and best practices for optimal performance in HPC environments.

## Environment

**Environment used:**
```
envs/uv/u260304_vllm
```

**Repository commit:**
```
<commit-hash>
```

## Model Information

**Model:** neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8

**HuggingFace Link:** [neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8](https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8)

**Model Size:** 405B parameters

**Precision:** FP8 (8-bit floating point quantization)

**Context Length:** 128K tokens

**License:** Llama 3.1 Community License

**Storage Requirements:** Approximately 382GB

## Hardware Configuration

**H100 Configuration (2-node):**
- **GPU Type:** NVIDIA H100 80GB
- **Number of GPUs:** 8 (4 per node)
- **Number of Nodes:** 2
- **GPUs per Node:** 4
- **Network:** InfiniBand
- **Total GPU Memory:** 640GB
- **CPU per Task:** 32 cores
- **Memory per Node:** 500GB

**H200 Configuration (1-node):**
- **GPU Type:** NVIDIA H200 141GB
- **Number of GPUs:** 4
- **Number of Nodes:** 1
- **GPUs per Node:** 4
- **Network:** InfiniBand
- **Total GPU Memory:** 564GB
- **CPU per Task:** 32 cores
- **Memory per Node:** 500GB

## Prerequisites

### Access Requirements

**1. Request Model Access on Hugging Face**

The model `neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8` is gated and requires manual approval:

1. Go to the model page: https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
2. Read the terms of use and click "Agree and access repository"
3. Wait for approval (you will receive a notification once granted access)

> [!NOTE]
> Although Neural Magic models are available under the RedHatAI namespace on some platforms, the correct model path to use in your code is `neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8`. Please make sure to use this path when specifying the model.

**2. Set Up Your Hugging Face API Token**

To download and access the model, configure your Hugging Face API token:

1. Go to your Hugging Face account settings: https://huggingface.co/settings/tokens
2. Create a new API token (read access is sufficient for inference)
3. Copy the generated token
4. Set it as an environment variable:
   ```bash
   export HF_TOKEN=<your_token_here>
   ```

### Storage Configuration

The model requires approximately **382GB** of storage. Ensure you have:
- Sufficient space on shared storage accessible by all nodes
- High-performance storage (e.g., Lustre with proper striping) to minimize weight loading time
- Proper mounting on all cluster nodes

Configure the model cache location using the `HF_HOME` environment variable:

```bash
export HF_HOME=<your_model_cache_path>
```

For this workflow, you can use the cluster's `scratch` space, which is a high-performance storage solution optimized for AI workloads. Consult with your system administrator for storage optimization recommendations.

### Environment Setup

Activate the vLLM environment:

```bash
# Modify the path below to point to your specific environment activation script
source /n/holylfs06/LABS/kempner_dev/.../envs/uv/u260304_vllm/vllm_env/bin/activate
```

For more details on environment setup, see the environment documentation at [envs/uv/u260304_vllm](../../envs/uv/u260304_vllm/).

## Parallelism Configuration

**H100 Configuration (8 GPUs across 2 nodes):**
- **Tensor Parallel Size:** 8
- **Pipeline Parallel Size:** 1
- **Total Parallel Size:** 8

**H200 Configuration (4 GPUs on 1 node):**
- **Tensor Parallel Size:** 4
- **Pipeline Parallel Size:** 1
- **Total Parallel Size:** 4

> [!NOTE]
> The model has 128 attention heads. The tensor parallel size must be a divisor of 128 (e.g., 1, 2, 4, 8, 16, 32, 64, 128). Pipeline parallelism has known issues in vLLM v0.11.2 and should be kept at 1.

## Step-by-Step Instructions

### 1. Download the Model (One-Time Setup)

Once access is granted and your environment is configured, download the model weights.

> [!WARNING]
> Running compute, storage, or network intensive workloads on the login node is strictly prohibited. Always use `srun` or `sbatch` to allocate compute resources for this step.

Allocate a compute node and download the model:

```bash
srun --nodes=1 --gres=gpu:1 --mem=100G --time=2:00:00 --pty bash
source /path/to/vllm_env/bin/activate
export HF_HOME=<your_model_cache_path>
export HF_TOKEN=<your_token>
hf download neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
```

This will take approximately 1 hour depending on network speed. Workshop participants can ask admins for pre-downloaded weights to skip this step.

Verify the download:

```bash
ls -lh $HF_HOME/models--neuralmagic--Meta-Llama-3.1-405B-Instruct-FP8/snapshots/<snapshot_id>/
```

### 2. Launch Multi-node Server

Choose the appropriate SLURM script for your GPU type:

**For H100 GPUs (2 nodes, 4 GPUs per node):**
```bash
sbatch setup_vllm_server_h100.sh
```

**For H200 GPUs (1 node, 4 GPUs):**
```bash
sbatch setup_vllm_server_h200.sh
```

**What happens under the hood:**

1. The `init_cluster.sh` script initializes a Ray cluster across the allocated nodes
2. The vLLM server starts on the head node with the specified parallelism configuration
3. Ray distributes the model across all GPUs using tensor parallelism
4. The script waits for the model weights to load and the server to become healthy
5. Connection details are output to the log file

### 3. Connect to the Server

For security reasons, the server is not exposed to the public network. You must SSH into the head node to access it.

Check your SLURM output file (`vllm_405b_<jobid>.out`) for connection details:

```bash
ssh <your_username>@<head_node>
```

Once connected, activate your environment:

```bash
source /path/to/vllm_env/bin/activate
```

> [!NOTE]
> You don't need the same environment on the client side. Any environment with `curl` or Python with the `requests` library can send HTTP requests to the server.

### 4. Submit Inference Requests

**Simple inference request using curl:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8",
    "messages": [
      {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

**Batch inference using Python:**

This workflow includes `batch_processing_100.py` which demonstrates asynchronous batch processing:

```bash
python batch_processing_100.py
```

This script uses the OpenAI-compatible async client to send 100 concurrent requests, demonstrating the server's ability to handle high throughput.

### 5. Monitor Server Performance

SSH into the head node or worker nodes and monitor resource usage:

**GPU monitoring:**
```bash
watch -n 1 nvidia-smi
#or 
nvtop
```

**CPU and memory monitoring:**
```bash
htop
```

You should see all GPUs across the cluster being utilized during inference.

## Performance Notes

**Benefits of FP8 Quantization:**
- **Reduced Memory Footprint:** ~50% reduction compared to FP16/BF16 (382GB vs ~810GB)
- **Higher Throughput:** Faster inference due to reduced memory bandwidth requirements
- **Larger Batch Sizes:** More memory available for KV cache enables larger batches
- **Minimal Quality Degradation:** Neural Magic's quantization maintains output quality

**Optimization Tips:**
- Use `--gpu-memory-utilization 0.90` to maximize KV cache size
- Adjust `--max-model-len` based on your use case (max 128K tokens)
- Monitor NCCL communication overhead on multi-node setups
- Ensure InfiniBand is properly configured (`NCCL_SOCKET_IFNAME=ib0`)

**Expected Performance:**
- H100 (8 GPUs): ~16-20 tokens/second with batch size 1
- H200 (4 GPUs): ~8-12 tokens/second with batch size 1
- Higher throughput with larger batch sizes and shorter sequences

## Troubleshooting

**Issue: Out of Memory (OOM) Errors**
- **Solution:** Reduce `--max-model-len` (try 8192 or 4096)
- **Solution:** Lower `--gpu-memory-utilization` to 0.85 or 0.80
- **Solution:** Reduce concurrent request count

**Issue: Slow Model Loading**
- **Solution:** Ensure model weights are on high-performance storage (e.g., VASR)
- **Solution:** In case of Lustre, Check Lustre striping configuration (recommend 16 stripes for large files)
- **Solution:** Verify all nodes have access to the same storage path

**Issue: NCCL Communication Errors**
- **Solution:** Verify InfiniBand configuration: `export NCCL_SOCKET_IFNAME=ib0`
- **Solution:** Check NCCL environment variables in the SLURM script
- **Solution:** Ensure `NCCL_IB_HCA` is set correctly for your cluster

**Issue: Ray Cluster Initialization Fails**
- **Solution:** Check that all nodes can communicate over InfiniBand
- **Solution:** Verify SLURM allocation includes all requested nodes
- **Solution:** Review `init_cluster.sh` output for specific errors

**Issue: Tensor Parallel Size Errors**
- **Solution:** Use a divisor of 128 (the number of attention heads): 1, 2, 4, 8, 16, 32, 64, or 128
- **Solution:** Match TP size to total GPU count

## Known Limitations

- Pipeline parallelism has bugs in vLLM v0.11.2 and should be set to 1
- The model requires at least 4× H200 GPUs or 8× H100 GPUs for inference
- Tensor parallel size must divide evenly into 128 (number of attention heads)
- Maximum sequence length: 128K tokens (hardware dependent)

## Future Enhancements

- [ ] Add OpenAI-compatible web UI (e.g., OpenWebUI or Gradio)
- [ ] Performance benchmarking across different batch sizes
- [ ] Comparison with FP16 version throughput
- [ ] Integration with prompt caching for repeated prefixes

## References

- [Meta Llama 3.1 Model Card](https://huggingface.co/meta-llama/Meta-Llama-3.1-405B-Instruct)
- [Neural Magic FP8 Quantization](https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM FP8 Quantization Guide](https://docs.vllm.ai/en/latest/quantization/fp8.html)
- Related workflows:
  - [Llama-3.1-405B (FP16) Multi-node Server](../Llama-3.1-405B_multinode-server/)
  - [Llama-3.1-70B Multi-node Server](../Llama-3.1-70B_multinode-server/)

## Maintainer

- Created by: Naeem Khoshnevis
- Date: 2026-03-06
- Last updated: 2026-03-06 