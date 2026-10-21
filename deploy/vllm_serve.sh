#!/usr/bin/env bash

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/aiops-qwen3.5-9b-merged}"

SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-aiops-qwen3.5-9b}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"

VLLM_API_KEY="${VLLM_API_KEY:-}"

echo "[vllm_serve] MODEL_DIR           = ${MODEL_DIR}"
echo "[vllm_serve] SERVED_MODEL_NAME   = ${SERVED_MODEL_NAME}"
echo "[vllm_serve] HOST:PORT           = ${HOST}:${PORT}"

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "[vllm_serve] 错误: 找不到合并后的 checkpoint 目录 '${MODEL_DIR}'。" >&2
  echo "  先跑 deploy/merge_lora.py 产出合并 checkpoint，再通过 MODEL_DIR=<path> 环境变量" \
       "指定这个目录，或者放在默认路径 /data/models/aiops-qwen3.5-9b-merged 。" >&2
  exit 1
fi

if ! command -v vllm >/dev/null 2>&1; then
  echo "[vllm_serve] 错误: 找不到 vllm 命令。请先 pip install vllm（配合目标 GPU 的 CUDA 版本）。" >&2
  exit 1
fi

API_KEY_ARGS=()
if [[ -n "${VLLM_API_KEY}" ]]; then
  API_KEY_ARGS=(--api-key "${VLLM_API_KEY}")
fi

echo "[vllm_serve] 启动: vllm serve ${MODEL_DIR} --served-model-name ${SERVED_MODEL_NAME} ..."
vllm serve "${MODEL_DIR}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --dtype bfloat16 \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization 0.85 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  "${API_KEY_ARGS[@]}"
