"""
batch_prompt_sender.py

This script processes a batch of prompts by sending them to a specified API endpoint.
It uses multithreading to handle multiple requests concurrently, allowing for efficient processing
of large prompt sets while respecting the server's maximum sequence limits.
"""

import requests
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configuration
MAX_SEQS_PER_BATCH = 112  # Set this to match --max-num-seqs or vLLM default (e.g., 1024)
REQUEST_TIMEOUT = 600     # 10 minutes
API_URL = "http://<hostname>:8000/v1/chat/completions"  # Change <hostname> to vLLM head node e.g., holygpu8a13402
MODEL_PATH = "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/DeepSeek-R1"

# Load input prompts
with open("input_prompts.json", "r") as f:
    data = json.load(f)

system_message = {"role": "system", "content": data["system_prompt"]}
user_prompts = data["user_prompts"]
total_prompts = len(user_prompts)

print(f"Total prompts: {total_prompts}")
print(f"Batch size (--max-num-seqs): {MAX_SEQS_PER_BATCH}")

# Function to send a single prompt
def send_prompt(item):
    user_message = {"role": "user", "content": item["content"]}
    payload = {
        "model": MODEL_PATH,
        "messages": [system_message, user_message]
    }
    try:
        res = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        res.raise_for_status()
        msg = res.json()["choices"][0]["message"]
        answer = {
            "reasoning": msg.get("reasoning_content", "[no reasoning_content]"),
            "final": msg.get("content", "[no content]")
        }
    except Exception as e:
        answer = {
            "reasoning": "[ERROR]",
            "final": f"[ERROR] {str(e)}"
        }
    return {
        "question_tag": item["question_tag"],
        "response": answer
    }

# Batch processing
results = []

for i in range(0, total_prompts, MAX_SEQS_PER_BATCH):
    batch = user_prompts[i:i + MAX_SEQS_PER_BATCH]
    print(f"Processing batch {i // MAX_SEQS_PER_BATCH + 1} with {len(batch)} prompts...")

    with ThreadPoolExecutor(max_workers=len(batch)) as executor:
        futures = [executor.submit(send_prompt, item) for item in batch]
        for future in as_completed(futures):
            results.append(future.result())

print(f"All batches processed. Total responses: {len(results)}")

# Save output
with open("output_responses.json", "w") as f:
    json.dump(results, f, indent=2)

print("Responses saved to output_responses.json")