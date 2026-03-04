# DeepSeek-R1-0528 - Multi-Node Server

## Overview

This workflow demonstrates deploying DeepSeek-R1-0528, an upgraded version of the 671B parameter DeepSeek-R1 model with enhanced reasoning and inference through algorithmic optimizations and increased compute. This version shows improvements in math, programming, and logic, rivaling top models like O3 and Gemini 2.5 Pro.

**Note:** DeepSeek-R1-0528 uses the same deployment infrastructure as [DeepSeek-R1](../DeepSeek-R1_multinode-server/). The primary difference is the model checkpoint path. For detailed deployment instructions, performance benchmarks, and troubleshooting, see the [DeepSeek-R1 workflow](../DeepSeek-R1_multinode-server/).

## Environment

**Environment used:**
```
envs/conda/c250609_vllm085
```

**Repository commit:**
```
cf2de1a3815c10c59e33349cd81aef64e68ea0ae
```

## Model Information

**Model:** DeepSeek-R1-0528

**HuggingFace Link:** https://huggingface.co/deepseek-ai/DeepSeek-R1-0528

**Model Size:** 671B parameters

**License:** MIT (check model card for latest)

**Precision:** FP8

**Context Length:** 163,840 tokens (max_position_embeddings)

**Architecture:** Advanced reasoning model with FlashMLA attention (upgraded from DeepSeek-R1)

**Improvements over DeepSeek-R1:**
- Enhanced reasoning capabilities
- Improved performance in math, programming, and logic tasks
- Algorithmic optimizations with increased compute
- Competitive with O3 and Gemini 2.5 Pro

## Hardware Configuration

Same as [DeepSeek-R1 workflow](../DeepSeek-R1_multinode-server/#hardware-configuration):

### Configuration 1: 16 × NVIDIA H100 80GB
- 4 nodes × 4 GPUs per node
- Total GPU Memory: 1,280 GB
- VRAM Used: ~1,037 GB (81% utilization)

### Configuration 2: 8 × NVIDIA H200 141GB
- 2 nodes × 4 GPUs per node
- Total GPU Memory: 1,128 GB

See [DeepSeek-R1 Hardware Configuration](../DeepSeek-R1_multinode-server/README.md#hardware-configuration) for complete specifications.

## Prerequisites

Same prerequisites as [DeepSeek-R1 workflow](../DeepSeek-R1_multinode-server/#prerequisites):

1. Conda environment: [c250609_vllm085](../../envs/conda/c250609_vllm085/)
2. Multi-node SLURM cluster with H100 or H200 GPUs
3. Model checkpoint access

**Model Path:** Update to DeepSeek-R1-0528 checkpoint location:
```bash
MODEL_PATH="/path/to/DeepSeek-R1-0528"
```

## Parallelism Configuration

Identical to [DeepSeek-R1](../DeepSeek-R1_multinode-server/#parallelism-configuration):

- **16 × H100:** Tensor Parallel Size = 16
- **8 × H200:** Tensor Parallel Size = 8

## Quick Start

### 1. Set Up Environment

```bash
cd envs/conda/c250609_vllm085
mamba env create -f environment.yml
conda activate vllm-inference
```

### 2. Configure SLURM Script

Edit the model path in the appropriate SLURM script:

**For 16×H100:**
```bash
# In deepseek_r1_0528_h100_slurm.sh
MODEL_PATH="/path/to/DeepSeek-R1-0528"
```

**For 8×H200:**
```bash
# In deepseek_r1_0528_h200_slurm.sh
MODEL_PATH="/path/to/DeepSeek-R1-0528"
```

Update SLURM parameters (job name, partition, account) as needed.

### 3. Submit Job

```bash
# For 16×H100
sbatch workflows/DeepSeek-R1-0528_multinode-server/deepseek_r1_0528_h100_slurm.sh

# For 8×H200
sbatch workflows/DeepSeek-R1-0528_multinode-server/deepseek_r1_0528_h200_slurm.sh
```

### 4. Send Inference Requests

Once the server is running (look for `INFO: Starting vLLM API server on http://0.0.0.0:8000` in logs):

**Using curl:**
```bash
curl -sS http://<head-node>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/path/to/DeepSeek-R1-0528",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful assistant."
      },
      {
        "role": "user",
        "content": "Explain quantum entanglement in detail."
      }
    ]
  }' | jq -r '.choices[0].message | {reasoning: .reasoning_content, final: .content}'
```

**Using Python:**
```python
import requests

res = requests.post(
    "http://<head-node>:8000/v1/chat/completions",
    json={
        "model": "/path/to/DeepSeek-R1-0528",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant."
            },
            {
                "role": "user",
                "content": "Explain quantum entanglement in detail."
            }
        ]
    }
)

data = res.json()
msg = data['choices'][0]['message']

print("Reasoning:", msg.get('reasoning_content', '[No reasoning_content found]'))
print("Final:", msg['content'])
```

## Performance Notes

DeepSeek-R1-0528 uses the same deployment infrastructure and should achieve similar throughput characteristics as DeepSeek-R1. See [DeepSeek-R1 Performance Notes](../DeepSeek-R1_multinode-server/README.md#performance-notes) for:

- Throughput benchmarks (H100 and H200)
- Performance tuning recommendations
- vLLM parameter configurations
- Memory optimization strategies

**Expected Improvements:**
- Better reasoning quality compared to DeepSeek-R1
- Enhanced performance on math, programming, and logic tasks
- Similar or potentially improved inference efficiency

## Step-by-Step Instructions

For complete step-by-step deployment instructions, including:
- Detailed SLURM configuration
- Ray cluster setup
- Server monitoring
- Batch processing
- Troubleshooting guide

See the comprehensive [DeepSeek-R1 workflow documentation](../DeepSeek-R1_multinode-server/README.md#step-by-step-instructions).

**Key Difference:** Replace all instances of the model path with DeepSeek-R1-0528:
```bash
MODEL_PATH="/path/to/DeepSeek-R1-0528"
```

## Troubleshooting

Same troubleshooting steps as [DeepSeek-R1](../DeepSeek-R1_multinode-server/README.md#troubleshooting):

- Model loading times (20-40 minutes)
- Out of memory issues
- Ray cluster connection problems
- SLURM job allocation issues

## Differences from DeepSeek-R1

| Aspect | DeepSeek-R1 | DeepSeek-R1-0528 |
|--------|-------------|------------------|
| Model Size | 671B params | 671B params (same) |
| Reasoning | Advanced | Enhanced with optimizations |
| Math/Programming | Strong | Improved |
| Performance | High-end | Competitive with O3/Gemini 2.5 Pro |
| Deployment | Multi-node FP8 | Same infrastructure |
| Model Path | DeepSeek-R1 | DeepSeek-R1-0528 |

## References

- **Model Card:** https://huggingface.co/deepseek-ai/DeepSeek-R1-0528
- **Base Model Workflow:** [DeepSeek-R1](../DeepSeek-R1_multinode-server/)
- **Environment:** [c250609_vllm085](../../envs/conda/c250609_vllm085/README.md)
- **vLLM Documentation:** https://docs.vllm.ai/

## Maintainer

- Created by: Max Shad
- Date: 2025-06-09
- Last updated: 2026-03-04
