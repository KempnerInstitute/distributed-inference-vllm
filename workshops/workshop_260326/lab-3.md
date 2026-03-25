# LAB 3 — Production Server Deployment with vLLM

## Estimated Time
40–50 minutes

---

## Objective

By the end of this lab, you will:

- Deploy vLLM as an OpenAI-compatible API server
- Understand single-node vs multi-node server configurations
- Send requests to the server using curl and Python clients
- Learn about Ray cluster setup for multi-node deployments
- Understand production deployment considerations

---

## Why Deploy as a Server?

In LAB 2, you used vLLM's offline batch mode. For production use cases, you need a **server** that:

- Accepts requests via HTTP API (OpenAI-compatible)
- Handles concurrent requests efficiently
- Provides standardized endpoints for clients
- Supports streaming responses
- Can scale across multiple nodes

---

## Part A: Single-Node Server Deployment (Hands-on)

In this part, you'll deploy Llama-3.1-70B as an OpenAI-compatible server on a single node with 2 GPUs using tensor parallelism.

### 1. Allocate GPU Resources

Request an interactive session with 2 GPUs:

```bash
salloc -p kempner_eng --reservation=inference_workshop    --nodes=1 --ntasks=1   --cpus-per-task=32   --mem=256G   --gres=gpu:2   -t 00-8:00:00
```

Once allocated, SSH into the node:

```bash
ssh $SLURM_NODELIST
```

### 2. Activate Your Environment

```bash
cd /path/to/your/environment
source .venv/bin/activate
```

### 3. Start the vLLM Server

For single-node deployments, you can start vLLM directly without Ray:

```bash
vllm serve meta-llama/Meta-Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90
```

**What's happening:**
- vLLM loads the 70B model and shards it across 2 GPUs
- The server starts on `http://localhost:8000`
- OpenAI-compatible endpoints are available at `/v1/chat/completions`, `/v1/completions`, etc.
- The server waits for incoming requests

Wait for the message:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 4. Test the Server with curl

Open a **new terminal**, SSH to the same node, and send a test request:

```bash
ssh $SLURM_NODELIST

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "messages": [
      {"role": "user", "content": "Please help me understand this concept in detail with clear examples, practical applications, and any common mistakes people usually make."}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "messages": [
      {"role": "user", "content": "Please help me understand this concept in detail with clear step-by-step reasoning, including how it connects to related ideas in the field."}
    ],
    "max_tokens": 1000,
    "temperature": 0.7
  }'
```

You should receive a JSON response with the generated text.

### 5. Test with Python Client

We've provided `test_server_3a.py` to demonstrate programmatic access:

```bash
python test_server_3a.py
```

The script sends multiple requests and measures latency and throughput.

### 6. Monitor Server Performance

In another terminal, monitor GPU utilization:

```bash
ssh $SLURM_NODELIST
nvtop
```

**What to observe:**
- Both GPUs actively processing requests
- Memory usage distributed across GPUs (~70GB each)
- GPU utilization during inference
- Multiple requests handled concurrently

### 7. Experiment with Concurrent Requests

Modify `test_server_3a.py` to send more concurrent requests and observe:
- How vLLM handles batching automatically
- Throughput improvements with concurrent requests
- Response time distribution

### 8. Stop the Server

When done, return to the terminal running the server and press `Ctrl+C` to stop it.

---

## Part B: Multi-Node Server Deployment (Documentation Only)

> **Note:** This section is for **documentation purposes**. Workshop participants will not have access to sufficient GPU resources for multi-node deployment. The instructors will demonstrate this setup, and you can use these instructions for future reference when you have access to multi-node clusters.

### Overview

For very large models (e.g., Llama-3.1-405B, DeepSeek-R1), you need to distribute the model across multiple nodes. This requires:

1. **Ray cluster setup** - Coordinate multiple nodes
2. **Multi-node networking** - InfiniBand or high-speed networking
3. **Distributed execution backend** - vLLM with Ray backend

### Multi-Node Architecture

```
Node 1 (Head)                Node 2 (Worker)
├── Ray Head (port 6379)    ├── Ray Worker
├── vLLM Server             │
├── 4x GPUs                 ├── 4x GPUs
└── Tensor Parallel: 0-3    └── Tensor Parallel: 4-7
```

### Prerequisites for Multi-Node Setup

1. **Multiple nodes allocated** via SLURM
2. **High-speed interconnect** (InfiniBand recommended)
3. **Shared storage** accessible from all nodes
4. **Ray installed** in your environment
5. **NCCL configured** for multi-node communication

### Step-by-Step Multi-Node Setup

#### 1. Allocate Multiple Nodes

Submit a SLURM job requesting 2 nodes with 4 GPUs each:

```bash
sbatch setup_vllm_multinode_server.sh
```

The SLURM script header:

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --time=12:00:00
```

#### 2. Initialize Ray Cluster

The `init_cluster.sh` script automatically:

- Detects allocated nodes
- Identifies the head node
- Starts Ray head on the primary node
- Connects worker nodes to the head
- Configures networking

```bash
source ./init_cluster.sh
```

**What happens:**
- Ray head starts on port 6379
- Worker nodes connect to head node IP
- All GPUs across nodes become available to vLLM

#### 3. Start vLLM with Ray Backend

Launch vLLM server with distributed execution:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-405B-Instruct \
    --pipeline-parallel-size 2 \
    --tensor-parallel-size 4 \
    --max-model-len 16384 \
    --distributed-executor-backend ray \
    --gpu-memory-utilization 0.90 \
    --trust-remote-code \
    --disable-custom-all-reduce \
    --enforce-eager
```

**Key parameters:**
- `--tensor-parallel-size 4`: Shard across 4 GPUs (4 per node × 2 nodes)
- `--distributed-executor-backend ray`: Use Ray for multi-node coordination
- `--disable-custom-all-reduce`: Use NCCL for GPU communication

#### 4. Test Multi-Node Server

Connect to the head node and send requests:

```bash
curl http://<head_node_ip>:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain distributed inference."}
    ],
    "max_tokens": 200
  }'
```

#### 5. Network Configuration (Important!)

For optimal multi-node performance, configure NCCL:

```bash
export NCCL_SOCKET_IFNAME=ib0  # Use InfiniBand
export NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4,mlx5_5  # InfiniBand adapters
export NCCL_SOCKET_FAMILY=AF_INET
```

### When to Use Multi-Node Setup

| Model Size         | Single Node (4 GPUs) | Multi-Node Required   | 
|--------------------|----------------------|-----------------------|
| 8B                 | Easily fits          | Not needed          |
| 70B                | With H200 (141GB)    | Not needed          |
| 405B FP8           | Requires ~380GB      | Required (2 nodes)  |
| 671B (DeepSeek-R1) | Requires ~600GB+     | Required (2+ nodes) |

### Multi-Node Challenges

**Communication Overhead:**
- GPUs on different nodes must communicate over network
- Adds latency compared to single-node setup
- Requires high-bandwidth interconnect

**Complexity:**
- Ray cluster setup and management
- Network configuration (NCCL, InfiniBand)
- Debugging across multiple nodes

**Benefits:**
- Enables models that don't fit on single node
- Scales to arbitrary model sizes
- Production-grade deployment capability

### Reference Scripts

All scripts for multi-node setup are available:

- `init_cluster.sh` - Ray cluster initialization
- `setup_vllm_multinode_server.sh` - Complete SLURM job script
- Check `workflows/Meta-Llama-3.1-405B-Instruct-FP8_multinode-server/` for full examples

---

## Comparison: Offline vs Server Mode

| Aspect | Offline Mode (LAB 2) | Server Mode (LAB 3) |
|--------|---------------------|---------------------|
| Use Case | Batch processing | Real-time serving |
| API | Python API | HTTP REST API |
| Requests | Pre-defined batch | Dynamic, ongoing |
| Deployment | Script execution | Long-running service |
| Clients | Same process | Any HTTP client |
| Scaling | Manual batching | Automatic concurrency |

---

## Production Deployment Considerations

When deploying vLLM in production:

1. **Resource Allocation**
   - Allocate sufficient GPU memory
   - Use `--gpu-memory-utilization` wisely (0.85-0.95)
   - Monitor and adjust based on workload

2. **Performance Tuning**
   - Adjust `--max-model-len` based on use case
   - Enable continuous batching (default in vLLM)
   - Use appropriate tensor/pipeline parallelism

3. **Reliability**
   - Set up health checks (`/health` endpoint)
   - Monitor GPU utilization and memory
   - Implement request timeouts
   - Plan for graceful shutdowns

4. **Security**
   - Don't expose server to public internet directly
   - Use authentication/authorization
   - Implement rate limiting
   - Validate input requests

5. **Monitoring**
   - Track throughput (requests/second)
   - Monitor latency distributions
   - Watch GPU utilization
   - Log errors and warnings

---

## Summary

You've now learned:

**Part A - Single-Node Deployment:**
- Deploy OpenAI-compatible vLLM server
- Send requests via curl and Python
- Monitor server performance
- Handle concurrent requests

**Part B - Multi-Node Deployment (Reference):**
- Understand Ray cluster architecture
- Learn multi-node setup steps
- Know when multi-node is necessary
- Recognize deployment challenges

**Key Takeaways:**
- vLLM server mode provides production-ready LLM serving
- Single-node setup is straightforward for models up to 70B
- Multi-node setup enables very large models (405B+)
- OpenAI-compatible API makes integration simple
- Ray enables distributed inference across nodes

---

## Next Steps

- Explore vLLM documentation: https://docs.vllm.ai/
- Try deploying with different models
- Experiment with pipeline parallelism
- Set up monitoring and logging
- Integrate with your applications

Congratulations on completing the workshop!
