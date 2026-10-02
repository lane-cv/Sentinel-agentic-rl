#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

LLAMAFACTORY_DIR="${LLAMAFACTORY_DIR:-${REPO_ROOT}/../LLaMA-Factory}"

TRAIN_CONFIG="${REPO_ROOT}/sft/llamafactory_config/qwen3_5_9b_lora_sft.yaml"
DATASET_INFO_SNIPPET="${REPO_ROOT}/data/cold_start/dataset_info_snippet_v9.json"

echo "[train_sft] REPO_ROOT           = ${REPO_ROOT}"
echo "[train_sft] LLAMAFACTORY_DIR    = ${LLAMAFACTORY_DIR}"
echo "[train_sft] TRAIN_CONFIG        = ${TRAIN_CONFIG}"

if [[ ! -d "${LLAMAFACTORY_DIR}" ]]; then
  echo "[train_sft] 错误: 找不到 LLaMA-Factory 目录 '${LLAMAFACTORY_DIR}'。" >&2
  echo "  clone 官方仓库后通过 LLAMAFACTORY_DIR=<path> 环境变量指定，或放在 ${REPO_ROOT}/../LLaMA-Factory 。" >&2
  exit 1
fi

LF_DATASET_INFO="${LLAMAFACTORY_DIR}/data/dataset_info.json"
if [[ ! -f "${LF_DATASET_INFO}" ]]; then
  echo "[train_sft] 错误: 没找到 ${LF_DATASET_INFO}，这不像一份正常的 LLaMA-Factory 安装目录。" >&2
  exit 1
fi

echo "[train_sft] 合并 dataset_info_snippet_v9.json 到 LLaMA-Factory 的 data/dataset_info.json ..."
python3 - "${LF_DATASET_INFO}" "${DATASET_INFO_SNIPPET}" "${REPO_ROOT}/data/cold_start" <<'PYEOF'
import json
import sys
from pathlib import Path

lf_dataset_info_path = Path(sys.argv[1])
snippet_path = Path(sys.argv[2])
cold_start_dir = Path(sys.argv[3])

lf_info = json.loads(lf_dataset_info_path.read_text(encoding="utf-8"))
snippet = json.loads(snippet_path.read_text(encoding="utf-8"))

for name, entry in snippet.items():
    entry = dict(entry)
    dataset = cold_start_dir / entry["file_name"]
    if not dataset.is_file():
        raise SystemExit(f"缺少 {dataset}。请先运行 data/prepare_training_data.sh。")
    entry["file_name"] = str(dataset)
    if name in lf_info and lf_info[name] != entry:
        print(f"  [WARN] 覆盖已存在的同名 dataset_info 注册项: {name}")
    lf_info[name] = entry

lf_dataset_info_path.write_text(
    json.dumps(lf_info, ensure_ascii=False, indent=2), encoding="utf-8"
)
print(f"  已合并 {list(snippet.keys())} 到 {lf_dataset_info_path}")
PYEOF

cd "${LLAMAFACTORY_DIR}"

echo "[train_sft] 启动训练: llamafactory-cli train ${TRAIN_CONFIG}"
llamafactory-cli train "${TRAIN_CONFIG}"

echo "[train_sft] 训练完成。LoRA 权重产出目录见 YAML 里的 output_dir。"
echo "[train_sft] 部署前用 deploy/merge_lora.py 将 LoRA adapter 合并到纯文本底座。参数见项目 README 的 GPU 训练与部署一节。"
