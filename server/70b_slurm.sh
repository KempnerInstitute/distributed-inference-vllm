#! /bin/bash
#SBATCH --job-name=vllm-inference-70b          # Change job name as needed
#SBATCH --account=                             # Example account: kempner-dev
#SBATCH --partition=                           # Example partition: kempner_h100
#SBATCH --output %x_%j/output_%j.out           # Output file
#SBATCH --error %x_%j/error_%j.out             # Error file
#SBATCH --time=4:00:00                         # Change job duration as needed
#SBATCH --nodes=1                              # Number of nodes
#SBATCH --ntasks-per-node=1                    # Number of tasks per node
#SBATCH --gpus-per-node=4                      # Number of GPUs per node (4 per node)
#SBATCH --cpus-per-task=96                     # CPUs per task (96 in H100 nodes)
#SBATCH --mem=0                                # All memory on the node
#SBATCH --exclusive                            # Use entire node resources

module load python/3.10.13-fasrc01
conda deactivate
conda activate vllm-inference

if [[ -z $SLURM_GPUS_ON_NODE ]]; then
    RAY_NUM_GPUS=0
else
    RAY_NUM_GPUS=$SLURM_GPUS_ON_NODE
fi

# choose available port on the head node
head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export RAY_HEAD_ADDR="$head_node_ip:$head_port"
echo "Head address: $RAY_HEAD_ADDR"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" --temp-dir /tmp/$USER/$SLURM_JOB_ID/ray \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 5

# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ray start --address="$RAY_HEAD_ADDR" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $RAY_NUM_GPUS --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 5
done

export RAY_ADDRESS="$RAY_HEAD_ADDR"

vllm serve /n/holylfs06/LABS/kempner_shared/Everyone/testbed/models/Llama-3.1-70B --tensor-parallel-size 4
