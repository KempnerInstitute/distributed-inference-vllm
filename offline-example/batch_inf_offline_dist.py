## Adopted from the example provided in the vllm git repo: https://github.com/vllm-project/vllm/tree/main

import argparse
from typing import Any, Dict, List

import numpy as np
import ray
from packaging.version import Version
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from vllm import LLM, SamplingParams

# Check Ray version
assert Version(ray.__version__) >= Version(
    "2.22.0"), "Ray version must be at least 2.22.0"


# Parse input arguments
parser = argparse.ArgumentParser(description="Run batch inference with Ray Data.")
parser.add_argument('--prompt_file', type=str, required=True, help="Path to the prompt text file.")
parser.add_argument('--tensor_parallel_size', type=int, default=1, help="Tensor parallel size per instance.")
parser.add_argument('--model_location', type=str, required=True, help="Model location for the LLM.")
args = parser.parse_args()

# Set tensor parallelism and prompt file from arguments
tensor_parallel_size = args.tensor_parallel_size
prompt_file = args.prompt_file
model_location = args.model_location

# Set sampling parameters
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
num_instances = 1  # Set the number of instances. Each instance will use tensor_parallel_size GPUs.


# Create a class to do batch inference.
class LLMPredictor:
    def __init__(self):
        # Create an LLM with specified tensor parallel size.
        self.llm = LLM(model=model_location, tensor_parallel_size=tensor_parallel_size)

    def __call__(self, batch: Dict[str, np.ndarray]) -> Dict[str, list]:
        # Generate texts from the prompts.
        outputs = self.llm.generate(batch["text"], sampling_params)
        prompt: List[str] = []
        generated_text: List[str] = []
        for output in outputs:
            prompt.append(output.prompt)
            generated_text.append(' '.join([o.text for o in output.outputs]))
        return {
            "prompt": prompt,
            "generated_text": generated_text,
        }


# Load the prompt file into Ray Data
ds = ray.data.read_text(prompt_file)


# Define scheduling strategy for tensor parallelism
def scheduling_strategy_fn():
    # One bundle per tensor parallel worker
    pg = ray.util.placement_group(
        [{
            "GPU": 1,
            "CPU": 1
        }] * tensor_parallel_size,
        strategy="STRICT_PACK",
    )
    return dict(scheduling_strategy=PlacementGroupSchedulingStrategy(
        pg, placement_group_capture_child_tasks=True))


resources_kwarg: Dict[str, Any] = {}
if tensor_parallel_size == 1:
    # For tensor_parallel_size == 1, set num_gpus=1.
    resources_kwarg["num_gpus"] = 1
else:
    # For tensor_parallel_size > 1, use placement groups
    resources_kwarg["num_gpus"] = 0
    resources_kwarg["ray_remote_args_fn"] = scheduling_strategy_fn

# Apply batch inference for all input data
ds = ds.map_batches(
    LLMPredictor,
    concurrency=num_instances,
    batch_size=32,
    **resources_kwarg,
)

# Peek first 10 results
#outputs = ds.take(limit=10)
outputs = ds.take_all()

# Peek first 10 results and write them to a file
with open("response_output.txt", "w") as f:
    for output in outputs:
        prompt = output["prompt"]
        generated_text = output["generated_text"]
        result = f"Prompt: {prompt!r}, Generated text: {generated_text!r}\n"
        f.write(result)
        print(result)  # Print to console as well for quick verification

# Optionally write full inference output data as Parquet files to S3
# ds.write_parquet("s3://<your-output-bucket>")



