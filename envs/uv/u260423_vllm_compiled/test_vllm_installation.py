#!/usr/bin/env python3
"""
Smoke test for u260423_vllm_compiled.

Verifies the locally-compiled vLLM loads, its C extension is present, and
CUDA is visible. Does not load a model — see test_vllm_inference.py for that.
"""

import os
import sys


def test_imports() -> bool:
    print("=" * 60)
    print("Testing vLLM (from-source) installation")
    print("=" * 60)

    try:
        import vllm
        print(f"vLLM imported successfully (version {vllm.__version__})")
        print(f"  Module file: {vllm.__file__}")
    except ImportError as e:
        print(f"Failed to import vLLM: {e}")
        return False

    # The C extension must be present for this env — if it's missing, the
    # build step was skipped or VLLM_USE_PRECOMPILED was set somewhere.
    try:
        import vllm._C  # noqa: F401
        print("vllm._C loaded (locally-compiled C extension present)")
    except ImportError as e:
        print(f"vllm._C missing: {e}")
        print("  This env is supposed to build kernels locally. Check the build step.")
        return False

    try:
        import torch
        print(f"PyTorch imported successfully (version {torch.__version__})")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        print(f"  CUDA version (toolkit wheel): {torch.version.cuda}")
        # is_available() can return True even when a subsequent context init
        # fails (driver shim reports device count but refuses cuInit). Probe
        # one GPU op to catch that case and report it explicitly rather than
        # letting cudnn.version()/get_device_name() crash the test.
        if torch.cuda.is_available():
            try:
                torch.cuda.init()
                print(f"  cuDNN version:  {torch.backends.cudnn.version()}")
                print(f"  GPU count:      {torch.cuda.device_count()}")
                for i in range(torch.cuda.device_count()):
                    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                x = torch.randn(8, 8, device="cuda:0")
                print(f"  Sanity tensor op: cuda:0 sum = {x.sum().item():.4f}")
            except Exception as e:
                print(f"  CUDA context init FAILED despite is_available()=True: {e}")
                print("  This is almost always a cluster-level issue")
                print("  (cgroup / capability grant). The build itself is fine.")
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


def test_vllm_basic() -> bool:
    print("\n" + "=" * 60)
    print("Testing vLLM basic API")
    print("=" * 60)

    try:
        from vllm import LLM, SamplingParams  # noqa: F401
        print("vllm.LLM and vllm.SamplingParams imported")

        sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
        print(f"SamplingParams created: temperature={sp.temperature}, "
              f"top_p={sp.top_p}, max_tokens={sp.max_tokens}")
        return True
    except Exception as e:
        print(f"Basic API check failed: {e}")
        return False


def main() -> int:
    print(f"Python: {sys.version.splitlines()[0]}")
    print(f"Executable: {sys.executable}")
    print(f"CUDA_HOME: {os.environ.get('CUDA_HOME', '<unset>')}")
    print()

    ok_imports = test_imports()
    ok_basic = test_vllm_basic()

    print("\n" + "=" * 60)
    if ok_imports and ok_basic:
        print("All checks passed.")
        print("Next: python test_vllm_inference.py")
        return 0
    print("Some checks failed — review the output above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
