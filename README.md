# Distributed Inference of Llama 3.1 Models with vLLM

This repository explains how to run inference on the Llama 3.1 70B and 405B models across multiple GPUs using the vLLM library.

From the [Llama 3.1 announcement](https://huggingface.co/blog/llama31), we have the following estimates for the memory requirements of the 70B and 405B models.

- The 70B model requires 140 GB of VRAM to store the model weights, and an additional 39.06 GB to hold the KV cache for a request using 128k tokens.
- The 405B model requires 810 GB of VRAM to store the model weights, and an additional 123.05 GB of VRAM to hold the KV cache for a request using 128k tokens.

Since an [Nvidia H100 SXM GPU](https://resources.nvidia.com/en-us-tensor-core/nvidia-tensor-core-gpu-datasheet) holds 80 GB of VRAM, these models are large enough that they require a multi-GPU setup for 70B model and multi-node setup for 405B model.

Based on the memory requirements above,

- The 70B model requires at least two H100 GPUs to store the weights and at least three H100 GPUs to handle a full context-length request.
- The 405B model requires at least eleven H100 GPUs to store the weights and at least twelve to handle a full context-length request.

In this repository, we provide scripts to set up inference servers using 4 H100 GPUs for the 70B model and 16 H100 GPUs for the 405B model with [vLLM](https://docs.vllm.ai/en/latest/index.html). vLLM is an open-source library that allows for easy setup of inference servers for both Llama 3.1 models on an AI cluster. The library supports model sharding through both pipeline parallelism (PP) and tensor parallelism (TP), which users can configure as needed to optimize performance.

By following the instructions below, one should be able to set up a server running on a SLURM job for use with other experiments.

## Setup Instructions

1. Create a conda environment.
    
    ```bash
    conda create -n vllm-inference python=3.10
    ```

    Adjust the name of your Conda environment in the SLURM scripts (default is `vllm-inference`).

1. Activate the conda environment.
    
    ```bash
    conda activate vllm-inference
    ```

1. Install python dependencies.
    
    ```bash
    pip install -r requirements.txt
    ```

1. Set parameters in the SLURM scripts
  
    Find the SLURM scripts in the `server/` directory,


    ```
    --account: SLURM Fairshare Account
    --output and --error: Logs
    --partition: Partition
    --job-name: Job Name
    --time: Job Duration 
    ```

1. Run the SLURM script for the desired model.

    If you want to run the 405B model, you should run the following command,

    ```bash
    sbatch server/405b_slurm.sh
    ```

    The script will create a [Ray cluster](https://docs.ray.io/en/latest/cluster/getting-started.html) and then start a vLLM server that hosts the model in the first node of the SLURM job. This can take some time (up to a couple hours for 405B) since the model weights will need to be loaded onto the GPUs.

    You can check the progress of the model loading by looking at the error logs for the SLURM job, which should have lines like

    ```
    Loading safetensors checkpoint shards:   0% Completed | 0/191 [00:00<?, ?it/s]

    Loading safetensors checkpoint shards:   1% Completed | 1/191 [00:04<13:59,  4.42s/it]

    Loading safetensors checkpoint shards:   1% Completed | 2/191 [00:09<15:58,  5.07s/it]

    Loading safetensors checkpoint shards:   2% Completed | 3/191 [00:26<32:54, 10.50s/it]
    ```

    When the model is fully loaded and the server is ready to handle requests, you should see lines like

    ```
    INFO:     Started server process [2405764]
    INFO:     Waiting for application startup.
    INFO:     Application startup complete.
    INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
    ```

## Using the vLLM Server

The vLLM server will be set up on the first node of the SLURM job.
This is marked in the head node in the SLURM job logs through a line similar to,

```
Head node: holygpu8a15303
```

but can also be found by using `squeue` to look at the nodes provisioned for the job and taking the first one. You can then ssh into this gpu node. Following the example above, if the first node is `holygpu8a15303`, we can run the following command to enter the gpu node.

```bash
ssh holygpu8a15303
```

The server will then be running on `localhost:8000`. You can send HTTP requests to the `/v1/completions` endpoint to run the model on your prompts.

```
curl http://localhost:8000/v1/completions \
    -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "model": "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```
Note that the model field for the JSON needs to be the directory of the model being served. vLLM uses the model directory to identify the model, so you should use the following strings for the `model` field.

- Llama 3.1 70B: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-70B`
- Llama 3.1 405B: `/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B`

For Python applications, you can use the `requests` library to send your HTTP requests.

```python
import requests

response = requests.post('http://localhost:8000/v1/completions', json = {
    "model": "/n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-405B",
    "prompt": "San Francisco is a",
    "max_tokens": 500,
    "temperature": 0
})

output = response.json()
```
Additional arguments like `top_k` and `min_p` are also available for adjusting how tokens are sampled. See the [vLLM docs](https://docs.vllm.ai/en/latest/dev/sampling_params.html) for the available arguments.
