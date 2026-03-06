# Llama 3.1 405B - Multi-Node Server

This workflow demonstrates deploying the [Llama 3.1 405B](https://huggingface.co/meta-llama/Llama-3.1-405B) model using vLLM on a multi-node multi-GPU setup with 16×H100 GPUs.

## Overview

Llama 3.1 405B is a 405 billion parameter language model from Meta AI, one of the largest openly available models. Based on GPU memory requirements:
- **Weights VRAM**: 810 GB
- **KV Cache VRAM** (128k tokens): 123.05 GB
- **Minimum GPUs**: 11×H100 for weights only, 12×H100 for full context length

This workflow uses **16×H100 GPUs** across multiple nodes to ensure sufficient memory for both model weights and KV cache with performance headroom.

**Environment**: [c250609_vllm085](../../envs/conda/c250609_vllm085/)  
**Model**: [meta-llama/Llama-3.1-405B](https://huggingface.co/meta-llama/Llama-3.1-405B)  
**Precision**: Standard (FP16/BF16)  
**Context Length**: 128k tokens  
**Maintainers**: Timothy Ngotiaoco, Max Shad

## Requirements

- **Hardware**: 16×NVIDIA H100 80GB GPUs (multi-node)
- **Environment**: conda environment `c250609_vllm085` with vLLM 0.8.5.post1
- **Model Path**: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B`

## Quick Start

### 1. Set Up Environment

```bash
cd ../../envs/conda/c250609_vllm085
# Follow README.md for conda environment setup
conda activate vllm-inference
```

### 2. Configure SLURM Script

Edit `llama_3.1_405b_slurm.sh` to set your SLURM parameters:

```bash
--account: SLURM Fairshare Account
--output and --error: Log file paths
--partition: Partition name
--job-name: Job name
--time: Job duration
```

### 3. Submit SLURM Job

```bash
sbatch llama_3.1_405b_slurm.sh
```

The script will:
1. Create a Ray cluster across multiple nodes
2. Start a vLLM server on the first node
3. Load model weights onto GPUs (this can take **up to a couple hours** due to the model size)

Monitor progress in the error logs:
```
Loading safetensors checkpoint shards:   0% Completed | 0/191 [00:00<?, ?it/s]
Loading safetensors checkpoint shards:   1% Completed | 1/191 [00:04<13:59,  4.42s/it]
Loading safetensors checkpoint shards:   1% Completed | 2/191 [00:09<15:58,  5.07s/it]
```

When ready, you'll see:
```
INFO:     Started server process [XXXXX]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Usage

### Find Head Node

Check the SLURM logs or use `squeue` to find the first node:
```
Head node: holygpu8aXXXXX
```

### SSH to Head Node

```bash
ssh holygpu8aXXXXX
```

### Send Requests

The server runs on `localhost:8000`. Use the `/v1/completions` endpoint:

**cURL example:**
```bash
curl http://localhost:8000/v1/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

**Python example:**
```python
import requests

response = requests.post('http://localhost:8000/v1/completions', json={
    "model": "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B",
    "prompt": "San Francisco is a",
    "max_tokens": 500,
    "temperature": 0
})

output = response.json()
print(output)
```

Additional sampling parameters are available: `top_k`, `min_p`, etc. See [vLLM sampling docs](https://docs.vllm.ai/en/latest/dev/sampling_params.html).

## Performance

| Metric | Value |
|--------|-------|
| GPUs | 16×H100 80GB |
| Context Length | 128k tokens |
| Weights VRAM | 810 GB |
| KV Cache VRAM | 123.05 GB (128k tokens) |

## Files

- `llama_3.1_405b_slurm.sh` - SLURM job script for 16×H100 multi-node deployment
- `README.md` - This file

## References

- [Meta Blog: Llama 3.1](https://ai.meta.com/blog/meta-llama-3-1/)
- [HuggingFace: Llama 3.1 GPU Requirements](https://huggingface.co/blog/llama31)
- [vLLM Documentation](https://docs.vllm.ai/en/latest/index.html)

## Troubleshooting

**Q: Model loading is very slow**  
A: Loading 810 GB of weights can take up to 2 hours. This is expected for such a large model. Monitor progress in the SLURM error logs.

**Q: Out of memory error**  
A: Ensure you're using 16×H100 80GB GPUs across multiple nodes. Reduce `max_model_len` in the SLURM script if needed.

**Q: Can't connect to server**  
A: Make sure you've SSH'd to the head node (the first node in your SLURM allocation) and the server has finished loading. Check for "Uvicorn running" message in logs.

**Q: Ray cluster not starting**  
A: Verify that all nodes can communicate with each other. Check firewall rules and network configuration.

---

**Maintainers**: Timothy Ngotiaoco, Max Shad  
**Last Updated**: 2026-03-04
