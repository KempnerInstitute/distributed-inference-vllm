# Large Language Model Distributed Inference

Welcome to this hands-on workshop on distributed inference for large language models. In this session, we'll explore how to deploy and scale LLMs across multiple GPUs and nodes using vLLM, covering essential concepts like tensor parallelism, pipeline parallelism, and optimizing inference performance for production workloads.

## Workshop Team

**Instructor** 
- Naeem Khoshnevis, Lead ML Research Engineer at Kempner Institute

**Associate Director of Education** 
- Denise Yoon

**Teaching Assistants (listed alphabetically)** 
- Bala Desinghu
- Yasin Mazloumi
- Nihal Vivekanand Nayak
- Timothy Ngotiaoco


## Objective

By the end of this workshop, you will be able to:

- Explain the core steps of LLM inference and the generation loop.
- Identify key performance challenges in LLM inference systems.
- Understand how vLLM improves inference efficiency.
- Run LLM inference on the Kempner HPC cluster.
- Deploy and interact with a vLLM inference server.


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
| 10:15 AM | **[LAB 1.a](lab-1a.md)** | Hands-on: Connect to the Cluster &  Setup Environment |
| | Llama 3.1 Architecture ||
| | The Self-Attention Mechanism ||
| | Attention Implementation ||
| | The Anatomy of Inference (Pipeline Overview)||
| | KV Caching ||
| | Prefill vs. Decode  ||
| | The Generation Loop ||
| | The Memory Explosion (MHA Baseline) ||
| | Memory requirements ||
| | Memory Cost of a Single Inference Request ||
| | Performance Metrics ||
| 10:40 AM | **[LAB 1.b](lab-1b.md)** | Hands-on: Performance baseline |
| 10:50 AM| **Break** | Coffee break |
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
| | Offline/Batch vLLM Inference with Python API ||
|11:45 AM | **[LAB 2](lab-2.md)** | Hands-on: Optimizing inference with vLLM |
| | **The Production** | |
| | Why Deploy an Inference Server? | |
| | vLLM Engine Architecture | |
| | Deploying the vLLM Server on HPC Clusters | |
| | vLLM Output Format | |
|12:15 PM | **[LAB 3](lab-3.md)** | Hands-on: Deploy and scale vLLM server |
| | Advanced Topics | |
| | Wrap-up and Q&A | Final questions and next steps |
| 12:30 PM | End | | 




# Additional Resources
### **Additional Resources**

#### **Cluster & Institute Guides**
* [FASRC Official Documentation](https://docs.rc.fas.harvard.edu/)
* [Kempner Engineering Handbook](https://handbook.eng.kempnerinstitute.harvard.edu/intro.html)

#### **Core Documentation & Guides**
* [vLLM Official Documentation](https://docs.vllm.ai/en/latest/)
* [Mastering LLM Techniques: Inference Optimization](https://developer.nvidia.com/blog/mastering-llm-techniques-inference-optimization/)
* [Baseten: Guide to Inference Engineering](https://www.baseten.co/inference-engineering/)
* [Hugging Face: Introducing Llama 3.1](https://huggingface.co/blog/llama31)

#### **Technical Deep Dives & Concepts**
* [Hugging Face: Everything You Need to Know About KV Cache](https://huggingface.co/blog/kv-cache)
* [Video: Understanding Continuous Batching and Throughput](https://youtu.be/z2M8gKGYws4?si=3phXlNKCSWuh9TXD&t=187)

#### **Research Papers**
* [PagedAttention: Efficient Memory Management for LLM Serving](https://arxiv.org/abs/2309.06180)
* [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
* [Splitwise: Efficient Generative LLM Inference Using Phase Splitting](https://arxiv.org/pdf/2311.18677)

#### **Hardware Reference**
* [Lenovo ThinkSystem SD665-N V3 (GPU Compute Node) Specifications](https://lenovopress.lenovo.com/lp1613-thinksystem-sd665-n-v3-server)