#!/usr/bin/env python3
"""
Simple test script to verify vLLM installation and basic functionality.
This script tests basic imports and environment checks without loading a model.
"""

import sys

def test_imports():
    """Test that all critical imports work."""
    print("=" * 60)
    print("Testing vLLM Installation")
    print("=" * 60)
    
    try:
        import vllm
        print(f"vLLM imported successfully (version {vllm.__version__})")
    except ImportError as e:
        print(f"Failed to import vLLM: {e}")
        return False
    
    try:
        import torch
        print(f"PyTorch imported successfully (version {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError as e:
        print(f"Failed to import PyTorch: {e}")
        return False
    
    try:
        import transformers
        print(f"Transformers imported successfully (version {transformers.__version__})")
    except ImportError as e:
        print(f"Failed to import Transformers: {e}")
        return False
    
    return True


def test_vllm_basic():
    """Test basic vLLM functionality without loading a model."""
    print("\n" + "=" * 60)
    print("Testing vLLM Basic Functionality")
    print("=" * 60)
    
    try:
        from vllm import LLM, SamplingParams
        print("vLLM classes imported successfully")
        
        # Test SamplingParams creation
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=100)
        print("SamplingParams created successfully")
        print(f"  Temperature: {sampling_params.temperature}")
        print(f"  Top-p: {sampling_params.top_p}")
        print(f"  Max tokens: {sampling_params.max_tokens}")
        
        return True
    except Exception as e:
        print(f"vLLM basic test failed: {e}")
        return False


def main():
    """Run all tests."""
    print(f"\nPython version: {sys.version}")
    print(f"Python executable: {sys.executable}\n")
    
    # Run tests
    imports_ok = test_imports()
    basic_ok = test_vllm_basic()
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    if imports_ok and basic_ok:
        print("All tests passed! Your vLLM environment is working correctly.")
        print("\nTo run a full inference test with a model, use:")
        print("  python test_vllm_inference.py")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
