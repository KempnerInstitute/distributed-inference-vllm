#!/bin/bash
#
# End-to-end setup for u260423_vllm_compiled — vLLM built 100% from source.
#
# Usage:   bash setup.sh
# Assumes: running on a Kempner / FASRC GPU node with internet access and
#          UV_CACHE_DIR already exported to a shared, writable path.
#
# Controls (override via env):
#   VLLM_COMMIT        upstream vLLM commit to pin (default matches u260324_vllm_source)
#   VLLM_SRC_DIR       where to clone vLLM (default: $PWD/vllm)
#   VENV_DIR           venv path (default: $PWD/vllm_env)
#   TORCH_CUDA_ARCH_LIST   target arch(s); default "9.0" (H100/H200)
#   MAX_JOBS           parallel ninja jobs; default 16
#   NVCC_THREADS       threads per nvcc; default 2

set -euo pipefail

: "${VLLM_COMMIT:=e9f331d72e90f34614363101528afe6c6fcdf7c5}"
: "${VLLM_SRC_DIR:=$PWD/vllm}"
: "${VENV_DIR:=$PWD/vllm_env}"
: "${TORCH_CUDA_ARCH_LIST:=9.0}"
: "${MAX_JOBS:=16}"
: "${NVCC_THREADS:=2}"

echo "======================================================================="
echo " u260423_vllm_compiled — vLLM from-source setup"
echo "======================================================================="
echo " VLLM_COMMIT            = $VLLM_COMMIT"
echo " VLLM_SRC_DIR           = $VLLM_SRC_DIR"
echo " VENV_DIR               = $VENV_DIR"
echo " TORCH_CUDA_ARCH_LIST   = $TORCH_CUDA_ARCH_LIST"
echo " MAX_JOBS               = $MAX_JOBS"
echo " NVCC_THREADS           = $NVCC_THREADS"
echo " UV_CACHE_DIR           = ${UV_CACHE_DIR:-<unset>}"
echo "======================================================================="

if [ -z "${UV_CACHE_DIR:-}" ]; then
    echo "ERROR: UV_CACHE_DIR is not set. Export it to a shared path before running." >&2
    echo "       Example: export UV_CACHE_DIR=/n/netscratch/.../uv_cache" >&2
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "ERROR: 'uv' not on PATH. Install it first:" >&2
    echo "       curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
fi

echo
echo "[1/6] Loading FASRC modules"
# `module` is an interactive shell function; when setup.sh is run via
# `bash setup.sh` it must be sourced explicitly.
if ! command -v module >/dev/null 2>&1; then
    if [ -f /etc/profile.d/lmod.sh ]; then
        # shellcheck disable=SC1091
        source /etc/profile.d/lmod.sh
    else
        echo "ERROR: 'module' command unavailable and /etc/profile.d/lmod.sh not found." >&2
        exit 1
    fi
fi
module purge
module load python/3.12.11-fasrc02
module load gcc/13.2.0-fasrc01
module load cuda/12.9.1-fasrc01
# NOTE: do NOT load the FASRC cudnn module — torch 2.11 bundles cuDNN 9.17.1
# and prepends its path; loading cudnn/9.10.2 here would shadow the bundled
# lib via LD_LIBRARY_PATH and cause "cuDNN version incompatibility" at torch
# CUDA init time. See README "Known issues" for the full story.
module load cmake/3.31.6-fasrc01
module list 2>&1 | sed 's/^/    /'

# The FASRC python module deactivates conda on load, which strips Miniforge's
# bin/ from PATH — so `which python` can fail even though the module loaded.
# $PYTHON_HOME is set reliably by the module and is the canonical reference.
if [ -n "${PYTHON_HOME:-}" ] && [ -x "$PYTHON_HOME/bin/python" ]; then
    PYBIN="$PYTHON_HOME/bin/python"
elif command -v python >/dev/null 2>&1; then
    PYBIN="$(command -v python)"
else
    echo "ERROR: could not locate python after module load (\$PYTHON_HOME=$PYTHON_HOME)." >&2
    exit 1
fi
echo "    Using python: $PYBIN ($($PYBIN --version))"

echo
echo "[2/6] Creating uv venv at $VENV_DIR"
if [ -d "$VENV_DIR" ]; then
    echo "    venv already exists; reusing. (Remove it first for a clean rebuild.)"
else
    uv venv "$VENV_DIR" --python "$PYBIN" --seed
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo
echo "[3/6] Cloning vLLM @ $VLLM_COMMIT into $VLLM_SRC_DIR"
if [ ! -d "$VLLM_SRC_DIR/.git" ]; then
    git clone https://github.com/vllm-project/vllm.git "$VLLM_SRC_DIR"
fi
(
    cd "$VLLM_SRC_DIR"
    git fetch --quiet origin
    git checkout "$VLLM_COMMIT"
    git submodule update --init --recursive
)

echo
echo "[4/6] Installing build deps from vLLM's build requirements file"
# --no-build-isolation means pip will NOT set up a fresh venv for the build
# and install the build-system requires from pyproject.toml. We therefore
# must pre-install every build-time dependency ourselves. vLLM ships the
# authoritative list in requirements/build/cuda.txt (newer layout) or
# requirements/build.txt (older layout — e.g. commit a32783bb3 and earlier).
# Detect which one this tree has and use it.
if [ -f "$VLLM_SRC_DIR/requirements/build/cuda.txt" ]; then
    BUILD_REQ="$VLLM_SRC_DIR/requirements/build/cuda.txt"
elif [ -f "$VLLM_SRC_DIR/requirements/build.txt" ]; then
    BUILD_REQ="$VLLM_SRC_DIR/requirements/build.txt"
else
    echo "ERROR: neither requirements/build/cuda.txt nor requirements/build.txt exists." >&2
    exit 1
fi
echo "    using $BUILD_REQ"
uv pip install -r "$BUILD_REQ" \
    --extra-index-url https://download.pytorch.org/whl/cu129 \
    --index-strategy unsafe-best-match

python -c "import torch; print(f'torch {torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}')"

echo
echo "[5/6] Building vLLM from source — this takes 60-90 minutes on H200 @ TORCH_CUDA_ARCH_LIST=9.0"
export MAX_JOBS NVCC_THREADS TORCH_CUDA_ARCH_LIST
unset VLLM_USE_PRECOMPILED 2>/dev/null || true
(
    cd "$VLLM_SRC_DIR"
    # Pass --extra-index-url so install_requires (torchvision, torchaudio, etc.)
    # resolve against cu129 wheels rather than generic PyPI (which ships CUDA 13
    # variants for torchvision 0.26+ and would ABI-mismatch against torch+cu129).
    pip install --no-build-isolation \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        -v -e .
)

# After editable install, double-check that torchvision/torchaudio came in as
# cu129 variants (pip's resolver occasionally picks the generic wheel despite
# the extra-index-url above). Force-reinstall the +cu129 variants if they're not.
echo
echo "    verifying torchvision / torchaudio are cu129 builds..."
if ! python - <<'PYCHECK'
import importlib.metadata as m
for name in ("torchvision", "torchaudio"):
    v = m.version(name)
    if "+cu" not in v:
        raise SystemExit(f"{name} is {v}, not a cu12x build")
print(f"  torchvision={m.version('torchvision')} torchaudio={m.version('torchaudio')}")
PYCHECK
then
    echo "    torchvision or torchaudio are not cu129; force-reinstalling..."
    uv pip install --force-reinstall --no-deps \
        --extra-index-url https://download.pytorch.org/whl/cu129 \
        --index-strategy unsafe-best-match \
        "torchvision==0.26.0+cu129" "torchaudio==2.11.0+cu129"
fi

echo
echo "[6/6] Smoke-checking the install"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import vllm._C; print('vllm._C loaded OK')"

echo
echo "======================================================================="
echo " DONE"
echo "======================================================================="
echo " Activate in future sessions with:"
echo "   module purge"
echo "   module load python/3.12.11-fasrc02 gcc/13.2.0-fasrc01 \\"
echo "               cuda/12.9.1-fasrc01 cmake/3.31.6-fasrc01"
echo "   # NOTE: do NOT load cudnn — torch 2.11 bundles its own (9.17.1)"
echo "   source $VENV_DIR/bin/activate"
echo
echo " Run verification:"
echo "   python test_vllm_installation.py"
echo "   python test_vllm_inference.py"
echo "======================================================================="
