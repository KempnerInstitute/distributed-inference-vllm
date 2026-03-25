#!/usr/bin/env python3
"""
LAB 2.a: vLLM Batch Inference on Single GPU
This script demonstrates vLLM's offline batch inference with Qwen2.5-32B-Instruct.
"""

import time
import torch
from vllm import LLM, SamplingParams

def main():
    print("=" * 80)
    print("LAB 2.a: vLLM Batch Inference with Qwen2.5-32B-Instruct (1 GPU)")
    print("=" * 80)
    print()
    
    # Model configuration
    model_name = "Qwen/Qwen2.5-32B-Instruct"
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("ERROR: No GPU available. This lab requires a GPU.")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print()
    
    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=150,
    )
    
    # Sample prompts for batch processing
    prompts = [
        "Explain the benefits of PagedAttention in vLLM in two sentences.",
        "What is continuous batching and why does it matter for LLM serving?",
        "Describe tensor parallelism in the context of large language models.",
        "How does vLLM improve GPU memory utilization compared to standard transformers?",
        "What are the key differences between online and offline inference modes?",
    ]
    
    print(f"Processing {len(prompts)} prompts in batch mode...")
    print()
    
    # Initialize vLLM
    print("Initializing vLLM... (this may take a minute)")
    start_init = time.time()
    
    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        tensor_parallel_size=1,  # Single GPU
        gpu_memory_utilization=0.9,  # Use up to 90% of GPU memory
    )
    
    init_time = time.time() - start_init
    print(f"✓ vLLM initialized in {init_time:.2f} seconds")
    print()
    
    # Run batch inference
    print("Running batch inference...")
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
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"  Peak GPU Memory: {memory_used:.2f} GB")
    
    print("=" * 80)
    print()
    
    print("Key Observations:")
    print("  ✓ Batch processing: All prompts processed efficiently together")
    print("  ✓ PagedAttention: Efficient KV cache memory management")
    print("  ✓ Higher throughput: Compared to sequential baseline (LAB 1.b)")
    print("  ✓ Better GPU utilization: Continuous batching optimizes compute")
    print()
    print("Compare these results with LAB 1.b baseline performance!")
    print("=" * 80)

if __name__ == "__main__":
    main()
