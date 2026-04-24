#!/usr/bin/env python3
"""
End-to-end inference test for u260423_vllm_compiled.

Loads a tiny open model (no gating, no HF token required), runs two prompts,
prints generated text + basic timing.

Defaults to tensor_parallel_size=1 — works on any number of visible GPUs.
To exercise multi-GPU tensor parallelism, set TEST_TP_SIZE to a divisor of
the model's attention-head count (14 for Qwen2.5-0.5B: use 1, 2, 7, or 14).

Model choice: Qwen/Qwen2.5-0.5B-Instruct - ~1 GB download, fits on any H100/H200.
"""

import os
import sys
import time


def main() -> int:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as e:
        print(f"Import failed: {e}")
        return 1

    if not torch.cuda.is_available():
        print("CUDA not available — this test requires a GPU.")
        return 1

    # is_available() can return True even when cuInit fails at actual use;
    # force an init to surface cluster-level driver issues up front.
    try:
        torch.cuda.init()
    except Exception as e:
        print(f"CUDA context init failed ({e}) — skipping inference test.")
        print("This indicates a cluster-level driver / cgroup issue, not a vLLM problem.")
        return 1

    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s):")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    model_id = os.environ.get("TEST_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")
    # Default to TP=1 (works for any model regardless of head count). Set
    # TEST_TP_SIZE explicitly to use multiple GPUs — it must divide the
    # model's attention-head count (14 for Qwen2.5-0.5B, so 1, 2, 7, or 14).
    tp_size = int(os.environ.get("TEST_TP_SIZE", "1"))
    print(f"\nLoading {model_id} with tensor_parallel_size={tp_size} ...")

    t0 = time.time()
    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=0.80,
        max_model_len=2048,
        disable_log_stats=True,
    )
    load_s = time.time() - t0
    print(f"Model loaded in {load_s:.1f}s\n")

    prompts = [
        "Write one short sentence about Boston.",
        "What is GPU inference in simple terms?",
    ]

    sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=64)

    t0 = time.time()
    outputs = llm.generate(prompts, sampling)
    gen_s = time.time() - t0
    print(f"Generation completed in {gen_s:.1f}s for {len(prompts)} prompt(s).\n")

    for i, o in enumerate(outputs):
        print(f"--- Prompt {i} ---")
        print(f"Q: {prompts[i]}")
        print(f"A: {o.outputs[0].text.strip()}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
