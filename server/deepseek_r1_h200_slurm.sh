#! /bin/bash
#SBATCH --job-name=<job-name>           # Job name
#SBATCH --partition=<partition-name>    # Partition name
#SBATCH --account=<account-name>        # Account name
#SBATCH --nodes=2                       # Number of nodes
#SBATCH --ntasks-per-node=1             # Number of tasks per node
#SBATCH --gpus-per-node=4               # Number of GPUs per node
#SBATCH --cpus-per-task=96              # Number of CPUs per task
#SBATCH --mem=1440G                     # Total memory per node (adjust as needed) 
#SBATCH --exclusive                     # Exclusive node allocation
#SBATCH --constraint=h200               # Constraint for H100 GPUs
#SBATCH --time=03-00:00:00              # Time limit DD-HH:MM:SS
#SBATCH -o job.%N.%j.out                # Output file
#SBATCH -e job.%N.%j.err                # Error file

# Run vLLM with DeepSeek-R1
module load python/3.10.13-fasrc01
conda deactivate
conda activate vllm-inference

# Set model path (FileSystem: Lustre, # of stripes: 16)
# Use VAST storage for faster caching in repetitive runs
MODEL_PATH="/n/netscratch/kempner_dev/Lab/mmsh/DeepSeek-R1"

# Setup Ray GPU config
if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

# Set head node and port
nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(getent hosts "$head_node" | awk '{print $1}')
used_ports=$(ss -Htan | awk -F':' '{print $NF}' | sort -u)
head_port=$(comm -23 <(seq 15000 20000) <(echo "$used_ports") | shuf -n 1)

echo "Head node: $head_node"
echo "Head node IP: $head_node_ip"
echo "Head port: $head_port"

# Export Ray head address
export RAY_HEAD_ADDR="$head_node_ip:$head_port"

# Start Ray head
echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# Wait for head node to start
sleep 10

# Start Ray workers
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 5
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"
export LOGLEVEL=DEBUG

# Run vLLM with DeepSeek-R1 671B model
vllm serve $MODEL_PATH \
  --tensor-parallel-size $((SLURM_NNODES * SLURM_GPUS_ON_NODE)) \
  --enable-reasoning \
  --reasoning-parser deepseek_r1 \
  --trust-remote-code \
  --enforce-eager

# Additional flags to fine-tune throughput and concurrency
#   --max-num-seqs 144 \
#   --max-model-len 4096 \
#   --max-num-batched-tokens 98304 \
#   --gpu-memory-utilization 0.93 \
#   --distributed-executor-backend ray \
#   --port 8000