#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

PY_VERSION="${PY_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.14.0}"
TORCH_INDEX="${TORCH_INDEX:-}"
INSTALL_VLLM="${INSTALL_VLLM:-1}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export UV_LINK_MODE=copy

command -v uv >/dev/null 2>&1 || pip install -q uv

install() {
  [ -x .venv/bin/python ] || uv venv -q -p "$PY_VERSION" .venv || { uv python install "$PY_VERSION" && uv venv -q -p "$PY_VERSION" .venv; }
  if [ -n "$TORCH_INDEX" ]; then
    uv pip install -q -p .venv/bin/python "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX"
  else
    uv pip install -q -p .venv/bin/python "torch==$TORCH_VERSION"
  fi
  uv pip install -q -p .venv/bin/python -r requirements.txt
  if [ "$INSTALL_VLLM" = "1" ]; then
    uv pip install -q -p .venv/bin/python -r requirements_vllm.txt
  fi
}

install || { echo "install failed once, retrying"; sleep 5; install; }

.venv/bin/python - <<'EOF'
import sys, torch, transformers, trl, sentence_transformers
print("venv ok", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("transformers", transformers.__version__, "trl", trl.__version__, "sentence_transformers", sentence_transformers.__version__)
EOF
