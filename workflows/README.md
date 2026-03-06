# Workflows

This directory contains **practical examples of running inference workflows** using vLLM.

## Available Workflows

| Model | Params | Precision | Context Length | Envs | Notes |
|-------|--------|-----------|----------------|------|-------|
| [Qwen2.5-32B-Instruct](Qwen2.5-32B-Instruct_single-gpu-inference/) | 32B | bfloat16 | 4096 | [u260304_vllm](../envs/uv/u260304_vllm/) | Single GPU inference on A100/H100 80GB |
| [Llama-3.1-70B](Llama-3.1-70B_multinode-server/) | 70B | FP16/BF16 | 128k | [c250609_vllm085](../envs/conda/c250609_vllm085/) | Multi-GPU server on 4×H100 |
| [Llama-3.1-405B](Llama-3.1-405B_multinode-server/) | 405B | FP16/BF16 | 128k | [c250609_vllm085](../envs/conda/c250609_vllm085/) | Multi-node server on 16×H100 |
| [Llama-3.1-405B-Instruct-FP8](Meta-Llama-3.1-405B-Instruct-FP8_multinode-server/) | 405B | FP8 | 128k | [u260304_vllm](../envs/uv/u260304_vllm/) | Multi-node server with FP8 quantization for improved throughput |
| [DeepSeek-R1](DeepSeek-R1_multinode-server/) | 671B | FP8 | 163840 | [c250609_vllm085](../envs/conda/c250609_vllm085/) | Multi-node server on 16×H100 or 8×H200 with reasoning capabilities |
| [DeepSeek-R1-0528](DeepSeek-R1-0528_multinode-server/) | 671B | FP8 | 163840 | [c250609_vllm085](../envs/conda/c250609_vllm085/) | Upgraded version with enhanced math, programming, and logic reasoning |


## Contributing

To contribute a new workflow, see the detailed guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).


