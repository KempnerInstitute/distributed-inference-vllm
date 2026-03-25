import torch
from vllm import LLM, SamplingParams

def main():
    print("torch.cuda.is_available():", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("device_count:", torch.cuda.device_count())
        print("device_name:", torch.cuda.get_device_name(0))

    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        dtype="float16",
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        disable_log_stats=True,
    )

    prompts = [
        "Write one short sentence about Boston.",
        "What is GPU inference in simple terms?",
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=64,
    )

    outputs = llm.generate(prompts, sampling_params)

    for i, output in enumerate(outputs):
        print(f"\n=== Prompt {i} ===")
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()