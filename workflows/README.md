# Workflows

This directory contains **practical examples of running inference workflows** using vLLM.

## Purpose

Each workflow represents a specific model with a particular scenario (single GPU, multi-GPU, multi-node, etc.). Workflows provide complete, reproducible recipes for:

- Setting up the environment
- Loading and running models
- Configuring parallelism
- Optimizing performance
- Troubleshooting issues

## Available Workflows

| Model | Params | Precision | Context Length | Envs | Notes |
|-------|--------|-----------|----------------|------|-------|
| [Qwen2.5-32B-Instruct](Qwen2.5-32B-Instruct_single-gpu-inference/) | 32B | bfloat16 | 4096 | [u260304_vllm](../envs/uv/u260304_vllm/) | Single GPU inference|


## Contributing

To contribute a new workflow, see the detailed guidelines in [CONTRIBUTING.md](../CONTRIBUTING.md).


