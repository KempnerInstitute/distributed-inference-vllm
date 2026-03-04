import time
from vllm import LLM, SamplingParams

# --- Configuration ---
model_id = "Qwen/Qwen2.5-32B-Instruct"

# --- Prompts ---
prompts = [
    "Explain the theory of relativity in simple terms. And let me know who you are.",
    "Write a short poem about a robot discovering a flower.",
    "What are the main differences between TCP and UDP?"
]

print(f"Loading model: {model_id}...")

# Initialize vLLM
# - tensor_parallel_size=1 (Since we are using 1 GPU)
# - max_model_len: Limits context to prevent OOM on the 70GB FP8 model
llm = LLM(
    model=model_id, 
    tensor_parallel_size=1, 
    max_model_len=4096,  # Conservative context length to fit in 80GB
    trust_remote_code=True
)

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=256)

# Run Inference
print("Generating responses...")
start_time = time.time()

outputs = llm.generate(prompts, sampling_params)

end_time = time.time()

# Print Results
print("-" * 50)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt}")
    print(f"Response: {generated_text!r}")
    print("-" * 50)

print(f"Inference completed in {end_time - start_time:.2f} seconds.")


# To run the script, use the following command:

# Activate your Python environment if needed, then run:
# python simple_inference_test.py