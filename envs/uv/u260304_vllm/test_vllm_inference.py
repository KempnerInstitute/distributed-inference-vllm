#!/usr/bin/env python3
"""
Full inference test with a small model.
This script downloads and runs inference with a small language model.
Requires GPU access.
"""

import sys

def test_simple_inference():
    """Test vLLM inference with a tiny model."""
    print("=" * 60)
    print("Testing vLLM Inference with facebook/opt-125m")
    print("=" * 60)
    print("\nNote: This will download ~250MB model on first run.\n")
    
    try:
        from vllm import LLM, SamplingParams
        import torch
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. This test requires a GPU.")
            print("  Make sure you're running on a GPU node.")
            return False
        
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        
        # Use a very small model for quick testing
        model_name = "facebook/opt-125m"
        print(f"\nLoading model: {model_name}")
        
        # Initialize vLLM with the model
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=0.3,  # Use only 30% of GPU memory
            max_model_len=512,  # Limit context length for faster loading
        )
        print("Model loaded successfully")
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50
        )
        
        # Test prompts
        prompts = [
            "Hello, my name is",
            "The capital of France is",
            "In machine learning,",
        ]
        
        print("\n" + "-" * 60)
        print("Running inference...")
        print("-" * 60)
        
        # Generate outputs
        outputs = llm.generate(prompts, sampling_params)
        
        # Print results
        for i, output in enumerate(outputs):
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"\nPrompt {i+1}: {prompt}")
            print(f"Generated: {generated_text}")
            print("-" * 60)
        
        print("\nInference completed successfully!")
        return True
        
    except Exception as e:
        print(f"\nInference test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run the inference test."""
    print(f"\nPython executable: {sys.executable}")
    
    success = test_simple_inference()
    
    print("\n" + "=" * 60)
    if success:
        print("Full inference test passed!")
        print("\nYour vLLM environment is fully functional and ready to use.")
        return 0
    else:
        print("Inference test failed.")
        print("\nTroubleshooting:")
        print("  1. Make sure you're on a GPU node")
        print("  2. Check CUDA modules are loaded")
        print("  3. Verify GPU is accessible with: nvidia-smi")
        return 1


if __name__ == "__main__":
    sys.exit(main())
