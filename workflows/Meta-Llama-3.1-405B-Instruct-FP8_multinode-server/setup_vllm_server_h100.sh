#!/bin/bash
#SBATCH --job-name=vLLM_405B_Server
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --partition=<partition_name>    <-- replace with your partition>
#SBATCH --account=<account_name>        <-- replace with your account>
#SBATCH --time=12:00:00
#SBATCH --output=vllm_405b_%j.out
#SBATCH --error=vllm_405b_%j.err
#SBATCH --mail-user=<EMAIL_ADDRESS>     <-- replace with your email>
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --constraint="h100"

# Export hugging face cache

export HF_HOME= <path_to_huggingface_cache>  # <-- replace with your Hugging Face cache path


# 1. Load your specific environment
source /n/holylfs06/LABS/.../vllm_env_dt/bin/activate # <-- replace with your environment activation command>

# 2. Source your existing init_cluster script
# Note: We source it so the exported variables like VLLM_HOST_IP stay in this shell
echo "========================================================================="
echo "Initializing Ray Cluster..."
source ./init_cluster.sh

echo "Done with Ray Cluster Initialization."
echo "========================================================================="



# 3. Export the Debugging/Performance variables we discovered
# Use the IP detected by your script for the Ray address
export RAY_ADDRESS="$head_node_ip:6379"
export NCCL_SOCKET_IFNAME=ib0
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5
export NCCL_IP_LOCAL_INSTAD_OF_HOSTNAME=1
export NCCL_SOCKET_FAMILY=AF_INET
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='{"placement_group_options":{"strategy":"SPREAD"}}'

# 4. Launch the vLLM server
echo "Starting vLLM API Server on $head_node_ip..."
python -m vllm.entrypoints.openai.api_server \
    --model neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8 \
    --tensor-parallel-size 8 \
    --max-model-len 16384 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager &

# 5. Wait for the server to be healthy
echo "Waiting for weights to load ..."
until curl -s "http://$head_node_ip:8000/v1/models" > /dev/null; do
    sleep 20
done

# 6. Final Instructions Output
echo "---------------------------------------------------------------"
echo "SERVER IS LIVE"
echo "URL: http://$head_node_ip:8000/v1"
echo "---------------------------------------------------------------"
echo "To connect from your local machine:"
echo "---------------------------------------------------------------"
echo "Head node information:"
echo "Head node name: $VLLM_HEAD_NODE"
echo "You must manually ssh to the head node to use the server."
echo "This node is not broadcasted to public."
echo "---------------------------------------------------------------"
echo "  curl http://$head_node_ip:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8\", \"messages\": [{\"role\": \"user\", \"content\": \"Test!\"}]}'"
echo "---------------------------------------------------------------"

# Keep the script alive so the background Ray/vLLM processes don't die
wait