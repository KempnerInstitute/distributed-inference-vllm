# Large Language Model Distributed Inference

Welcome to this hands-on workshop on distributed inference for large language models. In this session, we'll explore how to deploy and scale LLMs across multiple GPUs and nodes using vLLM, covering essential concepts like tensor parallelism, pipeline parallelism, and optimizing inference performance for production workloads.

## Workshop Team

**Instructor:** 
- Naeem Khoshnevis, Lead ML Research Engineer at Kempner Institute

**Associate Director of Education:** 
- Denise Yoon

**Teaching Assistants (listed alphabetically):** 
- Bala Desinghu
- Yasin Mazloumi
- Nihal Vivekanand Nayak
- Timothy Ngotiaoco


## Objective

TBD


## Prework 

Before attending the workshop, please complete the following prework to ensure environment and conceptual readiness:

1. **Set up Hugging Face access** - Create a Hugging Face account and generate an access token
2. **Run inference on Google Colab** - Download and run a small language model (distilgpt2) to verify your setup
3. **Accept model terms** - Visit Hugging Face pages for workshop models (Qwen2.5-32B, Llama-3.1-70B/405B, DeepSeek-R1, etc.) and accept their terms and conditions
4. **Submit confirmation** - Screenshot your inference output and submit via the provided form

This prework ensures everyone starts with a shared baseline and helps you understand why cluster-based distributed inference is necessary for production-grade models.

**Prework Notebook** 
- [https://colab.research.google.com/drive/1ghoOJy_KEggZBo1iG_pwYgYwmKWQVDYi](https://colab.research.google.com/drive/1ghoOJy_KEggZBo1iG_pwYgYwmKWQVDYi) 


## Workshop Agenda

### The Foundation
- Understanding inference
- LAB 1.a
- LAB 1.b

### The Optimization
- Introducing vLLM and solving bottlenecks
- LAB 2

### The Production
- Deploying the vLLM server
- LAB 3 


## Workshop Schedule

| Time | Topic | Description |
|------|-------|-------------|
| 9:30 AM | Welcome and Introduction | Workshop overview and logistics |
| 9:40 AM | **The Foundation** | |
| | Understanding inference ||
| | LLM Lifecycle ||
| | AI System Landscape ||
| | Physical Infrastructure ||
| | Kempner Cluster Infrastructure ||
| | The Software Backbone - Systems & Frameworks ||
| | The Inference Engine ||
| | The Model ||
| | Forms of the LLM Inference ||
| 10:15 AM | **LAB 1.a** | Hands-on: Connect to the Cluster &  Setup Environment |
| | Llama 3.1 Architecture ||
| | The Self-Attention Mechanism ||
| | Attention Implementation, Compute and Memory ||
| | The Anatomy of Inference (Pipeline Overview)||
| | KV Caching ||
| | Prefill vs. Decode  ||
| | The Generation Loop ||
| | The Memory Explosion (MHA Baseline) ||
| | Memory requirements ||
| | Memory Cost of a Single Inference Request ||
| | Performance Metrics ||
| 10:45 AM | **LAB 1.b** | Hands-on: Performance baseline |
| | **Break** | Coffee break |
| 11:00 AM | **The Optimization** | |
| | Why Inference Systems Exist | |
| | What is vLLM? | |
| | vLLM Ecosystem | |
| | The Naive Batching Problem | |
| | The Continuous Batching | |
| | vLLM’s Original Innovation: Paged Attention ||
| | The Connectivity Hierarchy to Go Beyond One GPU || 
| | Key Facts for Multi-GPU Inference ||
| | Types of Parallelism ||
| | vLLM Runtime Parameters || 
| | Offline/Batch vLLM Inference with Python API ||
| | **LAB 2** | Hands-on: Optimizing inference with vLLM |
| 11:45 AM | **The Production** | |
| | Why Deploy and Inference Server? | |
| | vLLM Engine Architecture | |
| | Deploying the vLLM Server on HPC Clusters | |
| | vLLM Output Format | |
| | **LAB 3** | Hands-on: Deploy and scale vLLM server |
| | Advanced Topics | |
| 12:20 PM | Wrap-up and Q&A | Final questions and next steps |
| 12:30 PM | End | | 




# Additional Resources


