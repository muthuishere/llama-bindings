#!/usr/bin/env bash
# tests/e2e/download_model.sh
#
# Downloads a small Qwen2.5-0.5B-Instruct GGUF model for e2e testing.
#
# Model: Qwen/Qwen2.5-0.5B-Instruct-GGUF  (q4_k_m quantisation, ~400 MB)
# Source: https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF
#
# Usage:
#   bash tests/e2e/download_model.sh
#   # or via Taskfile:
#   task download-model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
CHAT_MODEL="${MODELS_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"
EMBED_MODEL="${MODELS_DIR}/qwen2.5-0.5b-instruct-q4_k_m.gguf"   # reuse for embed in tests

# HuggingFace direct download URL
CHAT_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"

mkdir -p "${MODELS_DIR}"

download() {
  local url="$1"
  local dest="$2"
  if [ -f "${dest}" ]; then
    echo "✓ Already downloaded: $(basename "${dest}")"
    return 0
  fi
  echo "↓ Downloading $(basename "${dest}") ..."
  if command -v curl &>/dev/null; then
    curl -L --progress-bar -o "${dest}" "${url}"
  elif command -v wget &>/dev/null; then
    wget -q --show-progress -O "${dest}" "${url}"
  else
    echo "ERROR: neither curl nor wget found" >&2
    exit 1
  fi
  echo "✓ Saved to ${dest}"
}

download "${CHAT_URL}" "${CHAT_MODEL}"

echo ""
echo "Model ready at: ${CHAT_MODEL}"
echo ""
echo "Run e2e tests with:"
echo "  task e2e"
echo "  # or individually:"
echo "  task e2e-go"
echo "  task e2e-java"
echo "  task e2e-js"
