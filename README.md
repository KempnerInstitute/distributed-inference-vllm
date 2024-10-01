# Distributed Inference for Llama 3.1 with vLLM 

This repository details how to run inference on the Llama 3.1 70B and 405B models on multiple gpus using vLLM.
Looking at the [Llama 3.1 announcement](https://huggingface.co/blog/llama31), we see that the 70B model requires 140 GB of VRAM to store the model weights, and an additional 39.06 GB to hold the KV cache for a request using 128k tokens.
If we look at the 405B model instead, it requires 810 GB of VRAM to store the model weights, and an additional 123.05 GB of VRAM to hold the KV cache for a request using 128k tokens.
Since an Nvidia H100 GPU holds 80 GB of VRAM, these models are large enough that they require a multi-gpu setup for 70B and multi-node setup for 405B.
Based on the memory requirements above, the 70B model requires at least 2 H100s to hold the weights and at least 3 H100s to perform a full context length request, while the 405B model requires at least 11 H100s to hold the weights and at least 12 to perform a full context length request.
In this repository, we give scripts to set up inference servers using 4 GPUs for 70B and 16 GPUs for 405B with vLLM.

[vLLM](https://docs.vllm.ai/en/latest/index.html) is an open source library that allows us to easily set up an inference server for both Llama 3.1 models on our HPC cluster. The library can shard models using both pipeline parallelism and tensor parallelism, which users can configure as needed to optimize performance.
Following the instructions below, one should be able to get a server running on a SLURM job for use with other experiments.

## Setup Instructions

1. Create a conda environment.

```bash
conda create -n vllm-inference python=3.10
```
The SLURM scripts assume the conda environment is named `vllm-inference`, so if you name the environment something else, then please adjust the SLURM scripts as necessary.

1. Install python dependencies.

```bash
pip install -r requirements.txt
```

1. Route SLURM logs to desired directory.

You should set output and error logs in the SLURM scripts so that stdout and stderr are routed to accessible files. For example, if one wants to use `/n/home01/username/vllm-inference` for the server log dir, one should replace
```
#SBATCH --output <server_log_dir>/%x_%j/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error <server_log_dir>/%x_%j/error_%j.out  # File to which STDERR will be written, %j inserts jobid
```
with
```
#SBATCH --output /n/home01/username/vllm-inference/%x_%j/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /n/home01/username/vllm-inference/%x_%j/error_%j.out  # File to which STDERR will be written, %j inserts jobid
```
in the SLURM scripts.

1. Run the SLURM script for the desired model. For example, if you want to run the 405B model, you should run the following command.

```bash
sbatch server/405b_slurm.sh
```
This will create a server running in the first node of the SLURM job.
The script will create a Ray cluster and then start a server that hosts the model.
This can take some time (~40 minutes for 405B) since the model weights will need to be loaded onto the GPUs.
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
