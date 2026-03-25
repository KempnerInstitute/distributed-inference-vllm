#!/usr/bin/env python3
"""
LAB 2.b: vLLM Multi-GPU Inference with Tensor Parallelism
This script demonstrates vLLM's tensor parallelism with Llama-3.1-70B across 2 GPUs.
"""

import time
import torch
from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("LAB 2.b: vLLM Multi-GPU Inference with Llama-3.1-70B (2 GPUs)")
    print("=" * 80)
    print()
    
    # Model configuration
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    tensor_parallel_size = 2
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. This lab requires GPUs.")
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    if num_gpus < tensor_parallel_size:
        print(f"ERROR: This lab requires at least {tensor_parallel_size} GPUs")
        print(f"Please allocate a node with --gres=gpu:{tensor_parallel_size}")
        return
    
    for i in range(tensor_parallel_size):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} "
              f"({torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB)")
    print()
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
    )
    
    # Sample prompts for batch processing
    prompts = [
        "Explain how tensor parallelism enables running large models across multiple GPUs.",
        "What are the trade-offs between tensor parallelism and pipeline parallelism?",
        "Describe the communication patterns in tensor parallel inference.",
        "How does model sharding work in distributed LLM inference?",
    ]
    
    print(f"Processing {len(prompts)} prompts with tensor parallelism...")
    print(f"Model will be sharded across {tensor_parallel_size} GPUs")
    print()
    
    # Initialize vLLM with tensor parallelism
    print("Initializing vLLM with tensor parallelism... (this may take 1-2 minutes)")
    start_init = time.time()
    
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=tensor_parallel_size,  # Shard across 2 GPUs
        gpu_memory_utilization=0.95,  # Use up to 95% of GPU memory
    )
    
    init_time = time.time() - start_init
    print(f"✓ vLLM initialized with tensor parallelism in {init_time:.2f} seconds")
    print(f"✓ Model sharded across {tensor_parallel_size} GPUs")
    print()
    
    # Run batch inference
    print("Running batch inference with tensor parallelism...")
    print("-" * 80)
    start_time = time.time()
    
    outputs = llm.generate(prompts, sampling_params)
    
    end_time = time.time()
    total_time = end_time - start_time
    print("-" * 80)
    print()
    
    # Display results
    print("Generated Outputs:")
    print("=" * 80)
    total_tokens = 0
    
    for i, output in enumerate(outputs):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        total_tokens += num_tokens
        
        print(f"\n[Prompt {i+1}]")
        print(f"Input: {prompt[:80]}...")
        print(f"\nOutput ({num_tokens} tokens):")
        print(generated_text)
        print("-" * 80)
    
    # Performance metrics
    throughput = total_tokens / total_time
    avg_time_per_prompt = total_time / len(prompts)
    
    print("\n" + "=" * 80)
    print("Performance Metrics:")
    print(f"  Total processing time: {total_time:.2f} seconds")
    print(f"  Total tokens generated: {total_tokens}")
    print(f"  Throughput: {throughput:.2f} tokens/second")
    print(f"  Average time per prompt: {avg_time_per_prompt:.2f} seconds")
    print(f"  Prompts processed: {len(prompts)}")
    print()
    
    # Get memory usage per GPU
    if torch.cuda.is_available():
        print("  GPU Memory Usage:")
        for i in range(tensor_parallel_size):
            memory_used = torch.cuda.max_memory_allocated(i) / (1024**3)
            print(f"    GPU {i}: {memory_used:.2f} GB")
    
    print("=" * 80)
    print()
    
    print("Key Observations:")
    print("  ✓ Tensor parallelism: Model sharded across multiple GPUs")
    print("  ✓ Larger model: 70B parameters vs 32B in Part A")
    print("  ✓ Memory distribution: ~70GB per GPU instead of 140GB on one GPU")
    print("  ✓ GPU communication: Synchronization between GPUs during inference")
    print("  ✓ Quality improvement: Larger model provides better responses")
    print()
    print("Trade-offs to consider:")
    print("  • Communication overhead between GPUs")
    print("  • Lower throughput per GPU compared to single GPU setups")
    print("  • Enables models that don't fit in single GPU memory")
    print("=" * 80)

if __name__ == "__main__":
    main()
