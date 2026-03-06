#!/bin/bash

# 1. Get the node names
# For interactive sessions, we need to query the job info
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: Not running in a SLURM job."
    echo "Please allocate nodes first: salloc --nodes=1 --ntasks-per-node=1 --gres=gpu:4"
    return 1
fi

# Extract NodeList from the job
nodelist=$(scontrol show job $SLURM_JOB_ID | grep " NodeList=" | head -1 | sed 's/.*NodeList=//' | awk '{print $1}')

if [ -z "$nodelist" ]; then
    echo "Error: Could not detect node allocation from job $SLURM_JOB_ID"
    return 1
fi

echo "Detected node allocation: $nodelist"

nodes=$(scontrol show hostnames "$nodelist")
nodes_array=($nodes)
num_nodes=${#nodes_array[@]}

echo "Number of nodes: $num_nodes"
echo "Nodes: ${nodes_array[@]}"

# Detect number of GPUs from SLURM allocation
# SLURM_GPUS_ON_NODE contains the number of GPUs on the current node
# SLURM_JOB_GPUS or SLURM_STEP_GPUS contains GPU IDs
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    num_gpus=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_GPUS_PER_NODE" ]; then
    num_gpus=$SLURM_GPUS_PER_NODE
else
    # Try to count from SLURM_JOB_GPUS or CUDA_VISIBLE_DEVICES
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
    else
        # Default fallback - query from scontrol
        num_gpus=$(scontrol show job $SLURM_JOB_ID | grep -oP 'GRES=gpu:\K\d+' | head -1)
        if [ -z "$num_gpus" ]; then
            num_gpus=4  # Ultimate fallback
            echo "Warning: Could not detect GPU count, defaulting to 4"
        fi
    fi
fi

echo "GPUs per node: $num_gpus"

head_node=${nodes_array[0]}

export VLLM_HEAD_NODE=$head_node

echo "Head Node: $head_node"

# 2. Get the IP address of the head node (on the high-speed interface)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379

echo "Starting Ray Head on $head_node ($head_node_ip) with $num_gpus GPUs..."

# 3. Start Ray on the HEAD node (current node)
# We export VLLM_HOST_IP so vLLM binds to the correct network interface
export VLLM_HOST_IP=$head_node_ip
ray start --head --node-ip-address="$head_node_ip" --port=$port --num-gpus=$num_gpus --block &
sleep 10 # Give it a moment to initialize

# 4. Start Ray on WORKER nodes (if any)
if [ $num_nodes -gt 1 ]; then
    echo "Starting Ray Workers on ${num_nodes} worker node(s)..."
    for ((i=1; i<$num_nodes; i++)); do
        worker_node=${nodes_array[$i]}
        echo "  Starting worker on $worker_node..."
        srun --nodes=1 --ntasks=1 -w "$worker_node" \
            bash -c "export VLLM_HOST_IP=\$(hostname --ip-address | awk '{print \$1}') && \
            ray start --address='$head_node_ip:$port' --num-gpus=$num_gpus --block" &
        sleep 2
    done
    sleep 5
else
    echo "Single node setup - no worker nodes to start"
fi

echo ""
echo "========== Ray Cluster Summary =========="
echo "Head Node: $head_node ($head_node_ip)"
echo "Total Nodes: $num_nodes"
echo "GPUs per Node: $num_gpus"
if [ $num_nodes -gt 1 ]; then
    echo "Worker Nodes: ${nodes_array[@]:1}"
fi
echo "========================================="
