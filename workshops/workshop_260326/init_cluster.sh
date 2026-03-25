#!/bin/bash
#
# Multi-Node Ray Cluster Initialization Script
# This script initializes a Ray cluster across multiple SLURM-allocated nodes
#
# Usage: source ./init_cluster.sh (must be sourced to export variables)
#

echo "========================================================================="
echo "Initializing Ray Cluster for Multi-Node vLLM Deployment"
echo "========================================================================="

# 1. Verify we're running in a SLURM job
if [ -z "$SLURM_JOB_ID" ]; then
    echo "Error: Not running in a SLURM job."
    echo "Please allocate nodes first:"
    echo "  salloc --nodes=2 --ntasks-per-node=1 --gres=gpu:4 --mem=500G --time=12:00:00"
    return 1
fi

# 2. Get node list from SLURM
nodelist=$(scontrol show job $SLURM_JOB_ID | grep " NodeList=" | head -1 | sed 's/.*NodeList=//' | awk '{print $1}')

if [ -z "$nodelist" ]; then
    echo "Error: Could not detect node allocation from job $SLURM_JOB_ID"
    return 1
fi

echo "Detected node allocation: $nodelist"

# 3. Parse nodes from allocation
nodes=$(scontrol show hostnames "$nodelist")
nodes_array=($nodes)
num_nodes=${#nodes_array[@]}

echo "Number of nodes: $num_nodes"
echo "Nodes: ${nodes_array[@]}"

# 4. Detect GPU count
if [ -n "$SLURM_GPUS_ON_NODE" ]; then
    num_gpus=$SLURM_GPUS_ON_NODE
elif [ -n "$SLURM_GPUS_PER_NODE" ]; then
    num_gpus=$SLURM_GPUS_PER_NODE
else
    num_gpus=$(scontrol show job $SLURM_JOB_ID | grep -oP 'GRES=gpu:\K\d+' | head -1)
    if [ -z "$num_gpus" ]; then
        num_gpus=4  # Default fallback
        echo "Warning: Could not detect GPU count, defaulting to 4"
    fi
fi

echo "GPUs per node: $num_gpus"

# 5. Set head node
head_node=${nodes_array[0]}
export VLLM_HEAD_NODE=$head_node

echo "Head Node: $head_node"

# 6. Get head node IP (use high-speed network interface)
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
port=6379

echo "Head Node IP: $head_node_ip"
echo "Ray Port: $port"

# 7. Start Ray Head on the first node
echo ""
echo "Starting Ray Head on $head_node..."
export VLLM_HOST_IP=$head_node_ip

ray start --head \
    --node-ip-address="$head_node_ip" \
    --port=$port \
    --num-gpus=$num_gpus \
    --block &

sleep 10  # Wait for head to initialize

# 8. Start Ray Workers on remaining nodes
if [ $num_nodes -gt 1 ]; then
    echo ""
    echo "Starting Ray Workers on $((num_nodes - 1)) worker node(s)..."
    
    for ((i=1; i<$num_nodes; i++)); do
        worker_node=${nodes_array[$i]}
        echo "  Starting worker on $worker_node..."
        
        srun --nodes=1 --ntasks=1 -w "$worker_node" \
            bash -c "export VLLM_HOST_IP=\$(hostname --ip-address | awk '{print \$1}') && \
                     ray start --address='$head_node_ip:$port' --num-gpus=$num_gpus --block" &
        sleep 2
    done
    
    sleep 5
    echo "All Ray workers started"
else
    echo "Single node setup - no worker nodes needed"
fi

# 9. Export Ray address for vLLM
export RAY_ADDRESS="$head_node_ip:$port"

# 10. Summary
echo ""
echo "========================================================================="
echo "Ray Cluster Initialization Complete"
echo "========================================================================="
echo "Head Node: $head_node ($head_node_ip)"
echo "Total Nodes: $num_nodes"
echo "GPUs per Node: $num_gpus"
echo "Total GPUs: $((num_nodes * num_gpus))"
if [ $num_nodes -gt 1 ]; then
    echo "Worker Nodes: ${nodes_array[@]:1}"
fi
echo "Ray Address: $RAY_ADDRESS"
echo "========================================================================="
echo ""
echo "Ray cluster is ready for vLLM server deployment"
echo ""
