# vLLM Environment — Built Entirely From Source

Complete guide for compiling vLLM entirely from source on Kempner / FASRC HPC nodes. Every CUDA kernel, attention backend, and C++ extension in the resulting vLLM is built locally against the FASRC CUDA 12.9 toolchain — no precompiled wheel, no bundled-kernels tarball.

This environment has been end-to-end verified on both Hopper platforms (H100 and H200). The recipe is identical across the two — same `sm_90a` architecture target, same build targets, same extension modules, same 183-package frozen requirements snapshot. The only observable difference is wall-clock timing. Both platforms' numbers are shown in the Tested table below.

## Overview

Use this environment when you need the vLLM binary to come from *your* source tree, not upstream:

- Running vLLM with local patches to CUDA kernels, C++ ops, or CMake targets.
- Targeting a specific GPU arch with custom `TORCH_CUDA_ARCH_LIST` flags.
- Reproducing a bug against an exact upstream commit with full control over build flags.
- Debugging kernel compilation or linking issues.

If you only need to run stock vLLM on the cluster, prefer an env that installs a precompiled wheel (much faster setup — minutes instead of an hour).

---

## Tested end-to-end on 2026-04-24

| Run | Node | Allocation | Build wall time | Inference (Qwen2.5-0.5B, 1 GPU) |
|---|---|---|---|---|
| **H100** | 4× H100 80GB HBM3 | `--gres=gpu:4` | **71 min** (392 ninja targets, `MAX_JOBS=16 NVCC_THREADS=2 TORCH_CUDA_ARCH_LIST="9.0"`) | model load 38 s, 2 prompts 0.5 s, ~100+ tok/s output, FlashAttention 3 active |
| **H200** | 4× H200 | `--gres=gpu:4` | **63 min** (same 392 targets, same flags) | model load 44.5 s, 2 prompts 0.8 s, ~160 tok/s output, FlashAttention 3 active |

Same 8 `.abi3.so` extension modules produced in both runs (~1.95 GB total in the vllm source tree):

| Module | Size | Notes |
|---|---|---|
| `_C.abi3.so` | 331–508 MB | vLLM main C extension — paged attention, GPTQ/AWQ/FP8 quant, cutlass kernels, cudagraph wrappers. Size depends on build flags. |
| `_C_stable_libtorch.abi3.so` | 164 MB | Stable-libtorch-ABI build of a subset of ops; CMake generates sm_89 PTX alongside sm_90 for this target. |
| `_moe_C.abi3.so` | 177 MB | Mixture-of-experts kernels (Marlin, CUTLASS MOE). |
| `_flashmla_C.abi3.so` | 34 MB | FlashMLA (multi-head latent attention, optional). |
| `_flashmla_extension_C.abi3.so` | 9.9 MB | FlashMLA extension ops. |
| `_vllm_fa2_C.abi3.so` | 289 MB | FlashAttention 2 (sm_80+ general). |
| `_vllm_fa3_C.abi3.so` | 914 MB | FlashAttention 3 (Hopper-only, sm_90a) — biggest module due to the combinatorial explosion of headdim × dtype (fp16/bf16/fp8-e4m3) × paged × split × softcap × packgqa instantiations. |
| `cumem_allocator.abi3.so` | 96 KB | CUDA memory allocator helpers. |

> [!IMPORTANT]
> Tests were run in an allocation holding **all GPUs on the node** (`salloc --gres=gpu:4` on these 4-GPU Hopper nodes). On a partial-GPU allocation on the same partition, CUDA context creation fails with `cuInit → NOT_INITIALIZED` because SLURM's cgroup device isolation interacts badly with CUDA's device enumeration. This is a cluster-level issue, not a vLLM one — the build itself works under any allocation. See [Known Issues → cuInit NOT_INITIALIZED](#cuinit-returns-not_initialized-on-a-partial-gpu-allocation) for the full diagnosis and the workaround.

---

## Specifications

| Component | Version | Source |
|---|---|---|
| Python | 3.12.11 | FASRC `python/3.12.11-fasrc02` (Miniforge3 distribution) |
| GCC | 13.2.0 | FASRC `gcc/13.2.0-fasrc01` (nvcc 12.9 requires gcc ≤ 13) |
| CUDA toolkit | 12.9.1 | FASRC `cuda/12.9.1-fasrc01` |
| CMake | 3.31.6 | FASRC `cmake/3.31.6-fasrc01` (also acceptable: pip-installed `cmake>=3.26`) |
| PyTorch | 2.11.0+cu129 | PyTorch cu129 index; pinned by vLLM's `requirements/build/cuda.txt` |
| torchvision | 0.26.0+cu129 | PyTorch cu129 index (**must be the `+cu129` variant — see Known Issues**) |
| torchaudio | 2.11.0+cu129 | PyTorch cu129 index |
| cuDNN | 9.17.1 | **Bundled with the torch wheel** (the FASRC `cudnn/*` modules must NOT be loaded — see Known Issues) |
| NCCL | 2.28.9 | Bundled with torch |
| FlashInfer | 0.6.8.post1 | PyPI (runtime, JIT kernels) |
| vLLM | `0.19.2rc1.dev171+ge9f331d72` | Source, commit `e9f331d72e90f34614363101528afe6c6fcdf7c5` (main tip as of 2026-04-24 01:33 UTC) |
| vllm-flash-attn (bundled) | commit `f5bc33cfc02c744d24a2e9d50e6db656de40611c` | Source, fetched via CMake FetchContent from `vllm-project/flash-attention` |

`requirements-frozen.txt` in this directory is the 183-package snapshot from the successful run. The H100 and H200 runs produced byte-identical lock state — no platform-specific wheels.

---

## Prerequisites

### Hardware / allocation

- A compute node with a CUDA-12.9-capable driver. H100 and H200 both confirmed.
- **Allocate ALL GPUs on the node** (e.g. `--gres=gpu:4` on these 4-GPU H100/H200 nodes). A partial-GPU allocation causes `cuInit NOT_INITIALIZED` at runtime — build still works, but you can't import torch-CUDA / run vLLM. Full explanation in Known Issues.
- ≥ 32 CPU cores recommended (ninja + nvcc parallelism).
- ≥ 200 GB host RAM at `MAX_JOBS=16` (≈ 96 GB peak for nvcc).
- ≥ 20 GB free disk for source + build artifacts + venv. The 8 `.abi3.so` outputs alone are ~2 GB.

### Software

- `uv` ≥ 0.9 on `PATH`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- A shared `UV_CACHE_DIR` on netscratch or lab storage to avoid home-quota issues (torch and its CUDA sub-wheels total ~4 GB).
- Optional: `HF_HOME` pointed at a shared path if you plan to download HuggingFace models through vLLM.

### Time budget

| Scenario | Wall time |
|---|---|
| H100 node, warm wheel cache | **~71 min** (59 min compile + 12 min install/deps + 0 min download) |
| H200 node, warm wheel cache | ~63 min |
| Cold wheel cache (first run on new netscratch) | add ~10 min for torch+cu129 (~4 GB across base wheel + CUDA sub-wheels) |

The ~8 min H100 vs H200 gap is CPU-vintage differences between the partitions; GPU type does not matter for ninja/nvcc (compilation is CPU-bound).

---

## Quick Start

```bash
# 1. Allocate a full-GPU node (4× H100 or 4× H200)
salloc -p kempner_h100 --gres=gpu:4 --cpus-per-task=32 --mem=200G -t 02:00:00

# 2. Point uv at a shared cache
export UV_CACHE_DIR=/n/netscratch/<lab>/<user>/uv_cache

# 3. Run setup (loads modules — NOT cudnn — creates venv, clones vLLM,
#    installs build deps from requirements/build/cuda.txt, compiles vLLM,
#    verifies torchvision/torchaudio are cu129, prints DONE)
cd envs/uv/u260423_vllm_compiled
bash setup.sh

# 4. Verify
source vllm_env/bin/activate
python test_vllm_installation.py
python test_vllm_inference.py
```

---

## Detailed Setup Instructions

### Step 1 — Allocate a compute node

```bash
# 4× H100 (canonical full-node allocation)
salloc -p kempner_h100 --gres=gpu:4 --cpus-per-task=32 --mem=200G -t 02:00:00
```

### Step 2 — Load modules

```bash
module purge
module load python/3.12.11-fasrc02     # Miniforge Python 3.12 with C dev headers
module load gcc/13.2.0-fasrc01          # nvcc-compatible gcc
module load cuda/12.9.1-fasrc01         # nvcc, CUDA runtime libs, CUDA_HOME
module load cmake/3.31.6-fasrc01        # or install cmake via uv pip
# DO NOT load cudnn — torch 2.11 bundles its own (9.17.1); loading the FASRC
# cudnn/9.10.2 module would shadow the bundled copy via LD_LIBRARY_PATH and
# cause a cuDNN version incompatibility at torch CUDA init time.
```

### Step 3 — Create the virtual environment

```bash
export UV_CACHE_DIR=/n/netscratch/<lab>/<user>/uv_cache

# Bind the venv to the FASRC Miniforge Python. Use $PYTHON_HOME (set by the
# python module) rather than `which python` — the module's load hook runs
# `conda deactivate` which strips Miniforge's bin/ from PATH, so `which python`
# fails in a fresh shell.
uv venv vllm_env --python "$PYTHON_HOME/bin/python" --seed
source vllm_env/bin/activate
```

### Step 4 — Fetch vLLM source at the pinned commit

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout e9f331d72e90f34614363101528afe6c6fcdf7c5
git submodule update --init --recursive
```

### Step 5 — Install build dependencies from vLLM's own list

`pip install --no-build-isolation` in Step 6 tells pip **not** to resolve `[build-system].requires` from `pyproject.toml`, so we have to stage every build-time dependency ourselves (`setuptools_scm`, `jinja2`, the exact torch pin, etc.). vLLM ships the authoritative list in `requirements/build/cuda.txt`.

```bash
uv pip install -r requirements/build/cuda.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
```

At the pinned commit this resolves to:

- `cmake>=3.26.1`, `ninja`, `packaging>=24.2`, `wheel`
- `setuptools>=77.0.3,<81.0.0`, `setuptools-scm>=8`, `build`
- `torch==2.11.0+cu129` (~1.2 GB wheel + CUDA sub-wheels totaling ~4 GB)
- `jinja2>=3.1.6`, `regex`, `protobuf>=5.29.6`

Verify torch:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# Expected: 2.11.0+cu129 12.9 True
```

### Step 6 — Build vLLM from source

```bash
# Cap build parallelism. 16 × ~6 GB peak ≈ 96 GB RAM; lower MAX_JOBS if tight.
export MAX_JOBS=16
export NVCC_THREADS=2

# Restrict to your target arch(s) — cuts build time ~3x.
# H100 = 9.0, H200 = 9.0, A100 = 8.0, L4 = 8.9. Multiple allowed: "8.0;9.0".
export TORCH_CUDA_ARCH_LIST="9.0"

# Editable install, no build isolation (so the build picks up the torch
# installed above), AND pass the cu129 extra index so install_requires
# (torchvision, torchaudio, flashinfer, ...) also resolve to cu129 wheels —
# without this pip pulls the generic PyPI torchvision which is a CUDA 13 build
# and will ABI-mismatch at import time (see Known Issues).
unset VLLM_USE_PRECOMPILED
pip install --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -v -e .
```

Expect `[NNNN/392] Building CUDA object ...` lines for ~55–70 minutes depending on CPU vintage. The final link is `_vllm_fa3_C.abi3.so` (914 MB — Hopper flash-attn 3 with all its fp16/bf16/fp8-e4m3 × headdim × paged/split/softcap/packgqa instantiations).

### Step 7 — Verify the cu129 torchvision / torchaudio pin

On a clean wheel cache and a fresh venv, the `--extra-index-url` flag passed in Step 6 has reliably produced `+cu129` variants. If you hit a case where pip picks the generic wheel anyway (observable as `ldd .../torchvision/_C.so | grep cudart` showing `libcudart.so.13` instead of `.so.12`), force-reinstall:

```bash
uv pip install --force-reinstall --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"
```

`setup.sh` runs this check automatically after the pip install step.

### Step 8 — Run the tests

```bash
cd /path/to/envs/uv/u260423_vllm_compiled
python test_vllm_installation.py      # imports, CUDA visibility, kernel smoke, tensor op
python test_vllm_inference.py         # end-to-end Qwen2.5-0.5B on one visible GPU
```

---

## Re-using the environment in a new session

Every new shell must re-load modules and re-activate the venv. **Remember: do NOT load cudnn.**

```bash
module purge
module load python/3.12.11-fasrc02 gcc/13.2.0-fasrc01 cuda/12.9.1-fasrc01 cmake/3.31.6-fasrc01
source /path/to/envs/uv/u260423_vllm_compiled/vllm_env/bin/activate
```

---

## Rebuilding after source changes

**Python-only changes** (any `.py` under `vllm/`) — no rebuild needed; the editable install picks them up on next import.

**CUDA / C++ / CMake changes** (`*.cu`, `*.cpp`, `CMakeLists.txt`, kernel headers):

```bash
cd /path/to/vllm
pip install --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -v -e . --force-reinstall --no-deps
```

Clean rebuild:

```bash
rm -rf build/ vllm/*.so vllm/vllm_flash_attn/*.so
pip install --no-build-isolation \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -v -e .
```

To speed up iterative rebuilds, install `ccache` and export `CCACHE_DIR=/n/netscratch/.../ccache` before the build — unchanged translation units become near-instant on subsequent runs.

---

## Build-time environment variables

| Variable | Purpose | Recommended |
|---|---|---|
| `MAX_JOBS` | Parallel ninja jobs (governs peak RAM). | 16 on 200 GB node, 8 on 100 GB, 4 on 64 GB |
| `NVCC_THREADS` | Threads per nvcc invocation. | 2 |
| `TORCH_CUDA_ARCH_LIST` | Compile kernels only for specified arch(s). | `"9.0"` for H100/H200 |
| `CMAKE_BUILD_TYPE` | `Release` (default), `RelWithDebInfo`, `Debug`. | `Release` |
| `CCACHE_DIR` | Persistent ccache directory (requires ccache installed). | `$PWD/.ccache` or a netscratch path |
| `UV_CACHE_DIR` | Where uv stores downloaded wheels. | A shared, writable path — not home |
| `HF_HOME` | Where HuggingFace caches downloaded models. | A shared, writable path — not home |

**Explicitly do NOT set `VLLM_USE_PRECOMPILED`** — that flag downloads a tarball of prebuilt kernels, overlays your source-tree Python files on top, and short-circuits the local compile. This environment's whole point is to avoid that.

---

## Known issues and fixes

Every non-obvious failure mode hit during validation is documented here so the next person doesn't repeat the debugging.

### `cuInit` returns NOT_INITIALIZED on a partial-GPU allocation

**Symptom.** `nvidia-smi -L` works, but:
- `python -c "import torch; torch.cuda.init()"` → `RuntimeError: CUDA driver initialization failed, you might not have a CUDA gpu.`
- `$CUDA_HOME/extras/demo_suite/deviceQuery` → `cudaGetDeviceCount returned 3 -> initialization error`
- Raw `ctypes.CDLL("libcuda.so.1").cuInit(0)` → returns `3` (`CUDA_ERROR_NOT_INITIALIZED`)

**Root cause.** CUDA `cuInit` enumerates *every* `/dev/nvidia<N>` present in `/dev`, regardless of `CUDA_VISIBLE_DEVICES`. SLURM's cgroup device filter grants only the allocated device nodes; the others exist in `/dev` but return `EPERM` on `open()`. `cuInit` collapses those `EPERM`s into an opaque `NOT_INITIALIZED` rather than propagating per-device errors. See [SchedMD bug 1421](https://support.schedmd.com/show_bug.cgi?id=1421) and the NVIDIA forum thread *"CUDA accessing ALL devices, even those which are blacklisted"*.

**Build unaffected.** nvcc only needs the CUDA toolkit, not a live driver context. So the build always succeeds; only runtime (torch CUDA ops, vLLM inference) fails.

**Fix that works (user-space, no admin).** Allocate all GPUs on the node:

```bash
salloc --gres=gpu:4 ...      # for 4-GPU H100/H200 nodes
```

With a full-node allocation the cgroup grants every `/dev/nvidia<N>`, cuInit has nothing to `EPERM` on, and enumeration succeeds. Wasteful if you only need one GPU, but zero gymnastics.



### `ModuleNotFoundError: No module named 'setuptools_scm'` during editable install

**Cause.** `pip install --no-build-isolation` disables the automatic installation of `[build-system].requires` from `pyproject.toml`. vLLM's build uses `setuptools_scm` to derive the version string from git metadata.

**Fix.** Install vLLM's canonical build-deps list first (this is Step 5 of setup):

```bash
uv pip install -r requirements/build/cuda.txt \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match
```

### `torch.compile` fails with `Python.h: No such file or directory`

**Cause.** The venv was created against `uv`'s default standalone Python (`python-build-standalone`), which ships without CPython's C development headers. `torch.compile`'s Inductor backend builds a small C extension at runtime (`cuda_utils.c`) that `#include <Python.h>`, and the interpreter's `sysconfig` include dir must contain that header.

**Fix.** Point uv at the FASRC Miniforge Python (source-built with `--enable-shared` and the full dev layout):

```bash
module load python/3.12.11-fasrc02
uv venv vllm_env --python "$PYTHON_HOME/bin/python" --seed
```

The setup script uses `$PYTHON_HOME` (set by the FASRC python module) rather than `$(which python)` — the module's load hook runs `conda deactivate`, which strips Miniforge's `bin/` from `PATH`, so `which python` fails on a fresh shell even though the interpreter is still installed at `$PYTHON_HOME/bin/python`.

### `RuntimeError: operator torchvision::nms does not exist` at vLLM import

**Cause.** torchvision's C++ extension (`_C.so`) was linked against `libcudart.so.13` (CUDA 13) while torch is cu129. The generic PyPI wheel of torchvision 0.26.0 is a CUDA 13 build; the cu129 index has `torchvision-0.26.0+cu129` separately. If `pip install -e .` resolves vLLM's `install_requires` *without* the cu129 extra-index, it pulls the generic one and you get an ABI mismatch.

**Fix.** Ensure `--extra-index-url https://download.pytorch.org/whl/cu129` is passed to *both* the build-deps install AND the `pip install -e .` command. `setup.sh` does both. If you still end up with a mismatched torchvision:

```bash
uv pip install --force-reinstall --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match \
    "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"
```

`ldd .../site-packages/torchvision/_C.so | grep cudart` should report `libcudart.so.12`, not `.so.13`.

### `cuDNN version incompatibility: PyTorch was compiled against (9, 17, 1) but found runtime version (9, 10, 2)`

**Cause.** FASRC's `cudnn/9.10.2.21_cuda12-fasrc01` module prepends its cuDNN library path to `LD_LIBRARY_PATH`, shadowing the cuDNN 9.17.1 that torch 2.11 bundles inside its wheel.

**Fix.** Don't load the cudnn module. Torch 2.11 brings its own cuDNN and locates it correctly via its rpath when `LD_LIBRARY_PATH` doesn't contain a conflicting version. `setup.sh` does not load cudnn; the instructions above and the activate-in-new-session steps also skip it.

### `nvcc fatal: unsupported GNU version! gcc versions later than 14 are not supported!`

**Cause.** gcc 14+ loaded. CUDA 12.9's nvcc caps at gcc 13.

**Fix.** `module load gcc/13.2.0-fasrc01`.

### OOM killer during build

**Cause.** `MAX_JOBS` set too high for the node's memory.

**Fix.** Drop to 8 on 100 GB nodes, 4 on 64 GB. Each nvcc process can peak at 5–7 GB for the larger cutlass / flash-attn instantiations.

### Build succeeds but `import vllm._C` fails at runtime

**Cause.** Either `VLLM_USE_PRECOMPILED=1` was set somewhere (so the local build was skipped in favor of a downloaded tarball), or `pip install vllm` (non-editable) was run on top of the editable install.

**Fix.** Check that `vllm._C.__file__` points inside your source tree's `vllm/`:

```bash
python -c "import vllm._C; print(vllm._C.__file__)"
# Expected: /your/path/to/vllm/vllm/_C.abi3.so
```

If it points to site-packages, re-run the editable install:

```bash
cd /your/path/to/vllm
unset VLLM_USE_PRECOMPILED
pip install --no-build-isolation --force-reinstall --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    -e .
```

---

## Maintainer

- Created by: Naeem Khoshnevis
- Date: 2026-04-23
- Last updated: 2026-04-24
