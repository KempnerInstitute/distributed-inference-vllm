#!/bin/bash
#SBATCH --job-name=vLLM_MultiNode_Server
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --partition=gpu            # Replace with your partition
#SBATCH --account=your_account     # Replace with your account
#SBATCH --time=12:00:00
#SBATCH --output=vllm_multinode_%j.out
#SBATCH --error=vllm_multinode_%j.err
#SBATCH --mail-user=your_email@domain.edu  # Replace with your email
#SBATCH --mail-type=BEGIN,FAIL

# ============================================================================
# Multi-Node vLLM Server Setup
# ============================================================================
#
# This script sets up a multi-node vLLM server for large language models
# that require distribution across multiple nodes (e.g., 405B+ models).
#
# Requirements:
# - 2+ nodes with GPUs
# - Ray installed in environment
# - High-speed interconnect (InfiniBand recommended)
# - Shared storage accessible from all nodes
#
# ============================================================================

echo "========================================================================="
echo "Multi-Node vLLM Server Deployment"
echo "Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "========================================================================="

# ============================================================================
# Configuration
# ============================================================================

# Model configuration
MODEL_NAME="meta-llama/Meta-Llama-3.1-405B-Instruct"  # Adjust as needed
TENSOR_PARALLEL_SIZE=8  # Total GPUs (4 GPUs/node × 2 nodes)
MAX_MODEL_LEN=16384     # Adjust based on model and use case
GPU_MEMORY_UTIL=0.90    # GPU memory utilization (0.85-0.95)

# Environment paths
export HF_HOME=/path/to/huggingface_cache  # Replace with your HF cache path
VLLM_ENV_PATH=/path/to/vllm/env            # Replace with your vLLM environment

# ============================================================================
# 1. Activate Environment
# ============================================================================

echo "Activating vLLM environment..."
source $VLLM_ENV_PATH/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Failed to activate environment at $VLLM_ENV_PATH"
    exit 1
fi

echo "✓ Environment activated"

# ============================================================================
# 2. Initialize Ray Cluster
# ============================================================================

echo ""
echo "========================================================================="
echo "Initializing Ray Cluster..."
echo "========================================================================="

# Source init_cluster.sh to set up Ray across nodes
# This must be sourced (not executed) to export variables
# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "$SCRIPT_DIR/init_cluster.sh"

if [ $? -ne 0 ]; then
    echo "Error: Ray cluster initialization failed"
    exit 1
fi

echo "✓ Ray cluster initialized"

# ============================================================================
# 3. Configure NCCL for Multi-Node Communication
# ============================================================================

echo ""
echo "Configuring NCCL for multi-node communication..."

# NCCL settings for InfiniBand (adjust for your cluster)
export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5  # InfiniBand adapters (check with ibstat)
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_DEBUG=INFO  # Set to INFO for debugging, WARN for production

# vLLM distributed settings
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_DISTRIBUTED_EXECUTOR_CONFIG='{"placement_group_options":{"strategy":"SPREAD"}}'

echo "✓ NCCL configured"

# ============================================================================
# 4. Start vLLM Server
# ============================================================================

echo ""
echo "========================================================================="
echo "Starting vLLM Server..."
echo "========================================================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --max-model-len $MAX_MODEL_LEN \
    --distributed-executor-backend ray \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager &

# Store server PID
SERVER_PID=$!

# ============================================================================
# 5. Wait for Server to be Ready
# ============================================================================

echo ""
echo "Waiting for server to initialize and load model weights..."
echo "This may take several minutes for large models..."

# Wait for the /health endpoint to respond
max_attempts=60
attempt=0
until curl -s "http://$VLLM_HOST_IP:8000/health" > /dev/null 2>&1; do
    sleep 10
    attempt=$((attempt + 1))
    
    if [ $attempt -ge $max_attempts ]; then
        echo "Error: Server failed to start within expected time"
        echo "Check the logs above for errors"
        kill $SERVER_PID 2>/dev/null
        exit 1
    fi
    
    echo "  Waiting... (attempt $attempt/$max_attempts)"
done

# ============================================================================
# 6. Server is Ready
# ============================================================================

echo ""
echo "========================================================================="
echo "✓ SERVER IS LIVE AND READY"
echo "========================================================================="
echo "Server URL: http://$VLLM_HOST_IP:8000"
echo "OpenAI API endpoint: http://$VLLM_HOST_IP:8000/v1"
echo ""
echo "Head Node: $VLLM_HEAD_NODE"
echo "Head Node IP: $VLLM_HOST_IP"
echo ""
echo "========================================================================="
echo "Connection Instructions"
echo "========================================================================="
echo ""
echo "The server is NOT exposed to the public network."
echo "You must SSH to the head node to access it:"
echo ""
echo "  ssh $USER@$VLLM_HEAD_NODE"
echo ""
echo "Then test with curl:"
echo ""
echo "  curl http://$VLLM_HOST_IP:8000/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{"
echo "      \"model\": \"$MODEL_NAME\","
echo "      \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}],"
echo "      \"max_tokens\": 100"
echo "    }'"
echo ""
echo "========================================================================="

# ============================================================================
# 7. Keep Server Running
# ============================================================================

echo ""
echo "Server is running. Press Ctrl+C in the terminal to stop."
echo "Job will run for maximum allocated time: $SLURM_TIMELIMIT"
echo ""

# Wait for background server process
wait $SERVER_PID

echo ""
echo "========================================================================="
echo "Server stopped at: $(date)"
echo "========================================================================="
