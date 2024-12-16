#! /bin/bash
#SBATCH --job-name=vllm_ray
#SBATCH --account=kempner_dev
#SBATCH --output /n/home02/tngotiaoco/inference_workshop/%x_%j/output_%j.out  # File to which STDOUT will be written, %j inserts jobid
#SBATCH --error /n/home02/tngotiaoco/inference_workshop/%x_%j/error_%j.out  # File to which STDERR will be written, %j inserts jobid
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=96
#SBATCH --exclusive
#SBATCH --time=3:00:00
#SBATCH --mem=0			# All memory on the node
#SBATCH --partition=kempner_h100

module load python/3.10.13-fasrc01
conda deactivate
conda activate /n/netscratch/kempner_dev/Everyone/software/vllm-inference

# choose available port on the head node
head_port=`comm -23 <(seq 15000 20000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
nodes=`scontrol show hostnames $SLURM_JOB_NODELIST`
nodes_array=( $nodes )
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node: $head_node"
echo "Head node ip: $head_node_ip"
echo "Head port: $head_port"
export head_addr="$head_node_ip:$head_port"
echo "Head address: $head_addr"

echo "Starting Ray head on $head_node"
srun -N 1 -n 1 -w "$head_node" ray start --head --node-ip-address="$head_node_ip" \
    --port=$head_port --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --min-worker-port 20001 --max-worker-port 30000 --block &

# wait for head node to start
sleep 30

# start ray on the rest of the nodes
worker_num=$((SLURM_NNODES - 1))
for (( i = 1; i <= worker_num; i++ )); do
    node=${nodes_array[$i]}
    echo "Starting Ray worker on $node"
    srun -N 1 -n 1 -w "$node" ${SINGULARITY_WRAP} ray start --address="$head_addr" \
        --num-cpus $SLURM_CPUS_PER_TASK --num-gpus $SLURM_GPUS_PER_NODE --min-worker-port 20001 --max-worker-port 30000 --block &
    sleep 5
done

vllm serve /n/netscratch/kempner_dev/Everyone/models/Llama-3.1-70B --tensor-parallel-size 4
