#!/usr/bin/env bash
set -euo pipefail

target_dir="${1:-./weights}"
if [[ -n "${MODELSCOPE_REPO_ID:-}" ]]; then
    python3 - "$target_dir" "$MODELSCOPE_REPO_ID" <<'PY'
import sys
from modelscope.hub.snapshot_download import snapshot_download

print(snapshot_download(sys.argv[2], local_dir=sys.argv[1]))
PY
    exit 0
fi

url='https://cloud.tsinghua.edu.cn/f/99e43c53c0ae49129463/?dl=1'
sha256='3f9d882880114b2b302bddaf44114a44252a7c2a7a7e96d9a4e9edeaf56ac012'
mkdir -p "$target_dir"
tmp="$(mktemp -d "$target_dir/.download.XXXXXXXX")"
trap 'rm -rf -- "$tmp"' EXIT

curl --fail --location --retry 3 --connect-timeout 20 --proto '=https' --proto-redir '=https' \
    --output "$tmp/weights.zip" "$url"
(
    cd "$tmp"
    printf '%s  weights.zip\n' "$sha256" | shasum -a 256 -c -
    unzip -q weights.zip
)

for checkpoint in sft-v8 sft-v9 grpo-step10 grpo-step15; do
    source_dir="$tmp/sentinel-agent-lora-weights/$checkpoint"
    mkdir -p "$target_dir/$checkpoint"
    mv -f "$source_dir/adapter_config.json" "$source_dir/adapter_model.safetensors" "$target_dir/$checkpoint/"
done
printf '四组 LoRA 权重已保存到 %s\n' "$target_dir"
