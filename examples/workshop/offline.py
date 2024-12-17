import argparse
import itertools
import json
import requests
import time

import pandas as pd

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass


SERVER_ADDRESS = "localhost"


@dataclass
class InferenceRequestParams:
    model: str # Should be set to MODEL_PATH
    prompt: str
    max_tokens: int
    min_tokens: int = 0
    temperature: float = 1.0
    frequency_penalty: float = 0.0
    top_k: int = -1
    logprobs: int | None = None
    prompt_logprobs: int | None = None

    def dict(self) -> dict:
        return asdict(self)
    

def check_ready() -> bool:
    """
    Checks if the server is ready through the health check
    """
    try:
        response = requests.get(f"http://{SERVER_ADDRESS}:8000/health")
        response.raise_for_status()
        return True
    except Exception as e:
        return False
    

def wait_ready():
    """
    Waits until the server is ready with a good health check
    """
    while not check_ready():
        time.sleep(30)


def send_request(request_params: InferenceRequestParams):
    response = requests.post(f'http://{SERVER_ADDRESS}:8000/v1/completions', json = request_params.dict())
    return response.json()


def get_prompts(filename: str) -> list[str]:
    """
    Get prompts from the given file. Expects the file to be a JSON array of strings.
    """
    with open(filename, "r") as f:
        return json.load(f)
    
    
def write_outputs(data, filename: str) -> None:
    """
    Writes the provided data to filename provided.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-file", 
        help="Name of the input file to read prompts from", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--output-file", 
        help="Name of the output file to print the outputs to", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--model-path", 
        help="Path to the model weights being used on the server", 
        type=str, 
        required=True
    )
    parser.add_argument(
        "--num-threads", 
        help="Number of parallel requests", 
        type=int, 
        required=True
    )
    args = parser.parse_args()

    print("Waiting for server to finish startup", flush=True)
    wait_ready()
    print("Server is ready", flush=True)

    input_file = args.input_file
    output_file = args.output_file
    model_path = args.model_path
    num_threads = args.num_threads

    prompts = get_prompts(input_file)

    # Run each prompt 100 times to see diversity of output with temperature=1.0
    # flattened_params will look like [A, A, A, ..., A, B, B, B, ..., B, C, C, ...]
    params_list = [[InferenceRequestParams(model_path, prompt, max_tokens=100, temperature=1.0)]*100 for prompt in prompts]
    flattened_params = list(itertools.chain(*params_list))

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        outputs = pool.map(send_request, flattened_params)

    generated_outputs = [output["choices"][0]["text"] for output in outputs]

    write_outputs(generated_outputs, output_file)
