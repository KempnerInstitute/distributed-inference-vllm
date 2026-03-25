#!/usr/bin/env python3
"""
LAB 1.b: Baseline LLM Inference without vLLM
This script demonstrates simple inference using Hugging Face Transformers
with Meta-Llama-3.1-8B-Instruct on a single GPU.
"""

import torch
import time
import psutil
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Ensure only one GPU is visible
if 'CUDA_VISIBLE_DEVICES' not in os.environ:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def get_gpu_memory_usage():
    """Get current GPU memory usage in GB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024**3)
    return 0

def main():
    print("=" * 80)
    print("LAB 1.b: Baseline Inference with Meta-Llama-3.1-8B-Instruct")
    print("=" * 80)
    print()
    
    # Model configuration
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: Running on CPU. This will be very slow!")
        print("Please allocate a GPU node for this lab.")
        return
    
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("✓ Tokenizer loaded")
    print()
    
    # Load model
    print("Loading model... (this may take a few minutes)")
    start_load = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,  # Use bfloat16 for memory efficiency
        device_map="cuda:0",   # Explicitly use only GPU 0
    )
    
    load_time = time.time() - start_load
    print(f"✓ Model loaded in {load_time:.2f} seconds")
    print(f"GPU Memory Used: {get_gpu_memory_usage():.2f} GB")
    print()
    
    # Prepare prompt
    prompt = "Explain the concept of distributed inference for large language models in two paragraphs."
    
    print("Prompt:")
    print(f"  {prompt}")
    print()
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = inputs.input_ids.shape[1]
    print(f"Input tokens: {input_length}")
    print()
    
    # Run inference
    print("Generating response...")
    print("-" * 80)
    
    start_time = time.time()
    
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    end_time = time.time()
    
    # Decode output
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_length = output_ids.shape[1] - input_length
    
    print(generated_text)
    print("-" * 80)
    print()
    
    # Performance metrics
    total_time = end_time - start_time
    tokens_per_second = output_length / total_time
    
    print("Performance Metrics:")
    print(f"  Total generation time: {total_time:.2f} seconds")
    print(f"  Output tokens: {output_length}")
    print(f"  Throughput: {tokens_per_second:.2f} tokens/second")
    print(f"  Average time per token: {total_time/output_length:.3f} seconds")
    print(f"  Peak GPU Memory: {get_gpu_memory_usage():.2f} GB")
    print()
    
    print("=" * 80)
    print("Key Observations:")
    print("  • High memory usage (KV cache grows with sequence length)")
    print("  • Sequential token generation (no parallelization)")
    print("  • Cannot handle concurrent requests")
    print("  • No memory optimization (e.g., PagedAttention)")
    print()
    print("In LAB 2, we'll see how vLLM addresses these limitations!")
    print("=" * 80)

if __name__ == "__main__":
    main()
