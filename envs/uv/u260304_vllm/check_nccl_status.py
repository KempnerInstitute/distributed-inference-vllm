#!/usr/bin/env python3
"""
Script to check NCCL availability and show multi-GPU configuration.
This script demonstrates what would happen with multiple GPUs.
"""

import sys

def check_nccl_status():
    """Check NCCL installation and GPU setup."""
    print("=" * 60)
    print("NCCL and Multi-GPU Status Check")
    print("=" * 60)
    
    try:
        import torch
        
        # GPU count
        gpu_count = torch.cuda.device_count()
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"GPU count: {gpu_count}")
        
        # List all GPUs
        if gpu_count > 0:
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"  GPU {i}: {name} ({mem_gb:.1f} GB)")
        
        # NCCL availability - try different approaches
        print()
        nccl_available = False
        try:
            # Check if NCCL backend is available
            nccl_available = torch.distributed.is_nccl_available()
            print(f"NCCL backend available: {nccl_available}")
        except:
            # Fallback: check if NCCL module exists
            try:
                import torch.cuda.nccl
                nccl_available = True
                print(f"NCCL module available: True")
            except:
                print(f"NCCL not available")
        
        # NCCL version
        if nccl_available:
            try:
                version = torch.cuda.nccl.version()
                print(f"NCCL version: {version}")
            except:
                pass
        
        # Check for NCCL package
        try:
            import pkg_resources
            nccl_pkgs = [p for p in pkg_resources.working_set 
                        if 'nccl' in p.project_name.lower()]
            if nccl_pkgs:
                print(f"NCCL package: {nccl_pkgs[0]}")
        except:
            pass
        
        # Multi-GPU recommendations
        print("\n" + "=" * 60)
        print("Multi-GPU Configuration")
        print("=" * 60)
        
        if gpu_count == 1:
            print("\nSingle GPU detected (NCCL not needed)")
            print("\nCurrent mode: Single-GPU inference")
            print("  - No inter-GPU communication")
            print("  - NCCL installed but idle")
            print("  - Maximum model size limited to GPU memory")
            
            print("\nTo use NCCL with tensor parallelism:")
            print("  1. Request multiple GPUs from your cluster")
            print("     Example (SLURM): salloc --gres=gpu:4")
            print("  2. Use vLLM with tensor_parallel_size:")
            print("     llm = LLM(model='...', tensor_parallel_size=4)")
            
        elif gpu_count >= 2:
            print(f"\n{gpu_count} GPUs detected - NCCL will be used automatically!")
            print("\nExample tensor parallelism configurations:")
            
            for tp_size in [2, 4, 8]:
                if tp_size <= gpu_count:
                    print(f"\n  {tp_size} GPUs:")
                    print(f"    llm = LLM(")
                    print(f"        model='meta-llama/Llama-2-70b-hf',")
                    print(f"        tensor_parallel_size={tp_size}")
                    print(f"    )")
            
            print("\nNCCL will handle all inter-GPU communication")
            print("Can run much larger models than single GPU")
        
        print("\n" + "=" * 60)
        return True
        
    except Exception as e:
        print(f"\nError checking NCCL status: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_example_configs():
    """Show example configurations for different GPU counts."""
    import torch
    gpu_count = torch.cuda.device_count()
    
    print("\nExample vLLM Configurations")
    print("=" * 60)
    
    examples = {
        1: [
            ("Small model", "facebook/opt-125m", 1),
            ("Medium model", "facebook/opt-13b", 1),
            ("Large model (quantized)", "meta-llama/Llama-2-70b-hf (8-bit)", 1),
        ],
        4: [
            ("Large model", "meta-llama/Llama-2-70b-hf", 4),
            ("Very large model", "meta-llama/Llama-2-405b-hf", 4),
        ],
        8: [
            ("Massive model", "meta-llama/Llama-2-405b-hf", 8),
        ]
    }
    
    for required_gpus in sorted(examples.keys()):
        if required_gpus <= gpu_count:
            print(f"\nWith {required_gpus} GPU(s) available:")
            for desc, model, tp_size in examples[required_gpus]:
                print(f"  • {desc}: {model}")
                if tp_size > 1:
                    print(f"    → Use tensor_parallel_size={tp_size}")
                    print(f"    → NCCL handles GPU communication")


def main():
    """Run NCCL status check."""
    print(f"\nPython executable: {sys.executable}\n")
    
    success = check_nccl_status()
    
    if success:
        show_example_configs()
    
    print("\n" + "=" * 60)
    print("For more details, see: nccl_and_multi_gpu.md")
    print("=" * 60 + "\n")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
