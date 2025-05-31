# Distributed Inference of Large Language Models with vLLM

This repository explains how to run inference on the following models across multiple GPUs using the [vLLM](https://docs.vllm.ai/en/latest/index.html) library. vLLM is an open-source library that allows for easy setup of inference servers for both Llama 3.1 models as well as DeepSeek-R1 on an AI cluster. The library supports model sharding through both pipeline parallelism (PP) and tensor parallelism (TP), which users can configure as needed to optimize performance.

## Available Models

Follow the instruction page for each models to deploy them on an AI cluster.

| Model           | Model Size | Huggine Face | Instruction Page
|----------------|------------|------------------------------------------|-----------------|
| Llama 3.1 | 70B | [HF Link](https://huggingface.co/meta-llama/Llama-3.1-70B) | [Link](README_Llama3.1.md) |
| Llama 3.1 | 405B | [HF Link](https://huggingface.co/meta-llama/Llama-3.1-405B) | [Link](README_Llama3.1.md) |
| DeepSeek-R1 | 671B | [HF Link](https://huggingface.co/deepseek-ai/DeepSeek-R1) | [Link](README_DeepSeekR1.md) |


> **Note:** Follow this repository for regular updates on deployment instructions for the latest models on AI clusters.