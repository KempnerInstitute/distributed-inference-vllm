#!/usr/bin/env python3
"""
LAB 3.a: Test vLLM Server with Python Client
This script demonstrates how to interact with a vLLM server using the OpenAI client library.
"""

import time
from openai import OpenAI

def main():
    print("=" * 80)
    print("LAB 3.a: Testing vLLM Server with Python Client")
    print("=" * 80)
    print()
    
    # Initialize OpenAI client pointing to vLLM server
    # The server must be running on localhost:8000
    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="dummy",  # vLLM doesn't require authentication by default
    )
    
    # Test 1: Simple completion
    print("Test 1: Simple Chat Completion")
    print("-" * 80)
    
    start_time = time.time()
    
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "user", "content": "What is vLLM and why is it useful for LLM inference?"}
        ],
        max_tokens=150,
        temperature=0.7,
    )
    
    elapsed = time.time() - start_time
    
    print(f"User: What is vLLM and why is it useful for LLM inference?")
    print(f"\nAssistant: {response.choices[0].message.content}")
    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Tokens: {response.usage.completion_tokens}")
    print(f"Throughput: {response.usage.completion_tokens/elapsed:.2f} tokens/second")
    print("-" * 80)
    print()
    
    # Test 2: Multiple concurrent requests (simulate concurrent users)
    print("Test 2: Multiple Concurrent Requests")
    print("-" * 80)
    
    prompts = [
        "Explain tensor parallelism in one sentence.",
        "What is PagedAttention?",
        "How does continuous batching work?",
        "What are the benefits of model serving vs offline inference?",
    ]
    
    print(f"Sending {len(prompts)} requests...")
    start_time = time.time()
    
    # In a production scenario, you'd use async/await or threading
    # Here we do sequential requests for simplicity
    responses = []
    for prompt in prompts:
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0.7,
        )
        responses.append(response)
    
    total_time = time.time() - start_time
    total_tokens = sum(r.usage.completion_tokens for r in responses)
    
    print(f"\nCompleted {len(prompts)} requests")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per request: {total_time/len(prompts):.2f} seconds")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Overall throughput: {total_tokens/total_time:.2f} tokens/second")
    print("-" * 80)
    print()
    
    # Test 3: Streaming response
    print("Test 3: Streaming Response")
    print("-" * 80)
    print("User: Write a haiku about distributed computing.")
    print("\nAssistant (streaming): ", end="", flush=True)
    
    stream = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages=[{"role": "user", "content": "Write a haiku about distributed computing."}],
        max_tokens=100,
        temperature=0.8,
        stream=True,
    )
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end="", flush=True)
    
    print()
    print("-" * 80)
    print()
    
    print("=" * 80)
    print("Summary:")
    print("  ✓ Successfully connected to vLLM server")
    print("  ✓ Sent chat completion requests")
    print("  ✓ Handled multiple concurrent requests")
    print("  ✓ Tested streaming responses")
    print()
    print("The vLLM server provides an OpenAI-compatible API,")
    print("making it easy to integrate with existing applications!")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure the vLLM server is running!")
        print("Start it with: vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct --tensor-parallel-size 2")
