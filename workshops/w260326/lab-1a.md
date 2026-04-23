# LAB 1.a — Connect to Compute Node & Set Up Environment

## Estimated Time
10–15 minutes

---

## Objective

By the end of this lab, you will:

- Connect to the Kempner cluster
- Allocate a GPU compute node
- Set up a Python environment for inference

---

## 1. Connect to the Cluster

From your local machine:

```bash
ssh <your_username>@login.rc.fas.harvard.edu
```

## 2. Allocate a GPU Compute Node

```bash
salloc -p kempner_eng --reservation=inference_workshop    --nodes=1 --ntasks=1   --cpus-per-task=32   --mem=256G   --gres=gpu:1   -t 00-8:00:00
```

## 3. Check GPU Availability

```bash
nvidia-smi
```

## 4. Set Up Python Environment

In order to set up Python environment, we have several options. You can see the list of available approaches in the [envs](../envs) directory. For this lab, we will use the [vLLM Source Environment](../envs/uv/u260324_vllm_source) approach.


Here are the steps to set up the vLLM environment from source:

### Get the code

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout a32783bb35c6120df88d57609c8e454f9131f0a1
```
### Install editable vllm package

> [!Warning]  
> `uv` comes with significant amount of files. So it is important to use common cache directory to avoid hitting quota issues and to avoid unnecessary downloads. Please set your cache directory to a common location (e.g. in your lab directory or provided by the workshop) by setting the `UV_CACHE_DIR` environment variable before running `uv` commands.

```bash
export UV_CACHE_DIR=/n/netscratch/kempner_lab/Everyone/inference_workshop/uv_cache_dir  
```

> [!IMPORTANT]
> Load the FASRC `python/3.12.11-fasrc02` module **before** creating the venv. `uv`'s default standalone Python ships without the C development headers (`Python.h`), which causes `torch.compile` to fail at runtime in Lab 2 (`fatal error: Python.h: No such file or directory`). Pointing `uv` at the FASRC Python (which includes the full dev layout) avoids the failure.

```bash
module load python/3.12.11-fasrc02
uv venv --python $(which python)
source .venv/bin/activate
VLLM_USE_PRECOMPILED=1 uv pip install --editable .
```

Install other dependencies:

```bash
uv pip install accelerate
``` 


### Verify Environment Setup

```bash
module load gcc/13.2.0-fasrc01 
python -c "import vllm; print(vllm.__version__)"
```


You should see the following output:

```bash
0.18.1rc1.dev101+ga32783bb3
```

Done!
