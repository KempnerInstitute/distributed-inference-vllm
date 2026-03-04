# Workflows

This directory contains **practical examples of running inference workflows** using vLLM.

## Available Workflows

| Model | Params | Precision | Context Length | Envs | Notes |
|-------|--------|-----------|----------------|------|-------|
| [Qwen2.5-32B-Instruct](Qwen2.5-32B-Instruct_single-gpu-inference/) | 32B | bfloat16 | 4096 | [u260304_vllm](../envs/uv/u260304_vllm/) | Single GPU inference on A100/H100 80GB |
| [DeepSeek-R1](DeepSeek-R1_multinode-server/) | 671B | FP8 | 163840 | [c250609_vllm085](../envs/conda/c250609_vllm085/) | Multi-node server on 16×H100 or 8×H200 with reasoning capabilities |


## Contributing

To contribute a new workflow, see the detailed guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).


