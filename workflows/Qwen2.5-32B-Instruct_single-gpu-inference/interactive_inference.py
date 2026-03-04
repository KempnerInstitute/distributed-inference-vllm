import sys
from vllm import LLM, SamplingParams

# --- Configuration ---
model_id = "Qwen/Qwen2.5-32B-Instruct"

print(f"Loading model: {model_id}...")

# Initialize vLLM (This happens once and takes a minute)
llm = LLM(
    model=model_id, 
    tensor_parallel_size=1, 
    max_model_len=4096, 
    trust_remote_code=True
)

# Set sampling parameters
# Adjusted max_tokens to 512 for longer responses
sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=512)

print("\n" + "="*50)
print(f"Model {model_id} loaded successfully!")
print("Type 'exit' or 'quit' to stop.")
print("="*50 + "\n")

# --- Interactive Loop ---
while True:
    try:
        # 1. Get user input
        user_prompt = input("\n>>> Your Prompt: ")
        
        # Check for exit command
        if user_prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        if not user_prompt.strip():
            continue

        # 2. Format the prompt correctly for chat models
        # Qwen and Llama use specific chat templates. 
        # For simplicity, we pass the raw text, but strictly speaking,
        # chat models prefer: <|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant
        
        # Simple raw prompt for now:
        inputs = [user_prompt]

        # 3. Generate
        print("Thinking...", end="", flush=True)
        outputs = llm.generate(inputs, sampling_params, use_tqdm=False)
        print("\r", end="") # Clear "Thinking..."

        # 4. Print Response
        generated_text = outputs[0].outputs[0].text
        print(f"Response:\n{generated_text}")
        print("-" * 50)

    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        print("\nExiting...")
        break
    except Exception as e:
        print(f"\nAn error occurred: {e}")