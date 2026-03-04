# Distributed Inference with vLLM

Executable documentation and knowledge base for running distributed LLM inference using vLLM on HPC clusters.

## Overview

This repository provides reproducible recipes for deploying large language model inference at scale. Each workflow includes complete environment specifications, step-by-step instructions, and performance benchmarks tested on real GPU clusters.

**Key Features:**
- **Fully Reproducible** - Exact package versions, commit hashes, and hardware configs
- **Production-Ready** - Tested on HPC clusters with real workloads
- **Comprehensive Documentation** - From environment setup to troubleshooting
- **Multiple Parallelism Options** - Single GPU, tensor parallel, and multi-node setups

## Quick Start

```bash
# 1. Set up environment
cd envs/uv/u260304_vllm
source vllm_env/bin/activate

# 2. Run a workflow
cd ../../../workflows/Qwen2.5-32B-Instruct_single-gpu-inference
python simple_inference_test.py
```

See [workflows/](workflows/) for all available models and configurations.

## Repository Structure

```
├── envs/          # Reproducible runtime environments
├── workflows/     # Model inference recipes and examples
├── reports/       # Benchmarking and evaluation studies
├── workshops/     # Training and educational materials
├── scripts/       # Utility scripts and tools
└── CONTRIBUTING.md # Detailed contribution guidelines
```

## Getting Started

### 1. Choose Your Path

**For Quick Testing:**
Start with a single-GPU workflow:
- [Qwen2.5-32B-Instruct](workflows/Qwen2.5-32B-Instruct_single-gpu-inference/) - 32B parameter model on A100/H100

**For Production Deployment:**
Review environment specifications in [envs/](envs/) and select the appropriate workflow from [workflows/](workflows/)

### 2. Set Up Environment

Each workflow specifies its required environment. Navigate to the environment directory and follow setup instructions:

```bash
cd envs/uv/u260304_vllm
# Follow README.md for installation
```

### 3. Run Workflow

Navigate to your chosen workflow and follow its README:

```bash
cd workflows/Qwen2.5-32B-Instruct_single-gpu-inference
# Follow README.md for execution
```
## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## License

See [LICENSE](LICENSE) for details.

## NEWS


- **2026-03-04**: First uv environment ([u260304_vllm](envs/uv/u260304_vllm/)) and workflow ([Qwen2.5-32B-Instruct single-GPU inference](workflows/Qwen2.5-32B-Instruct_single-gpu-inference/)). Includes vLLM 0.11.2 with CUDA 12.9 support and comprehensive documentation following the new contribution guidelines.
- **2025-06-09**: DeepSeek-R1 multi-node deployment. New conda environment ([c250609_vllm085](envs/conda/c250609_vllm085/)) with vLLM 0.8.5.post1 and comprehensive [workflow](workflows/DeepSeek-R1_multinode-server/) for deploying 671B parameter model with FP8 precision on 16×H100 or 8×H200 GPUs. Includes throughput benchmarks and SLURM scripts.


---

<!-- 
OLD README CONTENT - TO BE REMOVED

# Distributed Inference of Large Language Models with vLLM

This repository explains how to run inference on the following models across multiple GPUs using the [vLLM](https://docs.vllm.ai/en/latest/index.html) library. vLLM is an open-source library that allows for easy setup of inference servers for both Llama 3.1 models as well as DeepSeek-R1 on an AI cluster. The library supports model sharding through both pipeline parallelism (PP) and tensor parallelism (TP), which users can configure as needed to optimize performance.

## Available Models

Follow the instruction page for each models to deploy them on an AI cluster.

| Model            | Model Size | Huggine Face                                                     | Instruction Page                  |
|------------------|------------|------------------------------------------------------------------|-----------------------------------|
| Llama 3.1        | 70B        | [HF Link](https://huggingface.co/meta-llama/Llama-3.1-70B)       | [Link](README_Llama3.1.md)        |
| Llama 3.1        | 405B       | [HF Link](https://huggingface.co/meta-llama/Llama-3.1-405B)      | [Link](README_Llama3.1.md)        |
| DeepSeek-R1      | 671B       | [HF Link](https://huggingface.co/deepseek-ai/DeepSeek-R1)        | [Link](README_DeepSeekR1.md)      |
| DeepSeek-R1-0528 | 671B       | [HF Link](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528)   | [Link](README_DeepSeekR1-0528.md) |


> [!NOTE]
> Follow this repository for regular updates on deployment instructions for the latest models on AI clusters.

-->
