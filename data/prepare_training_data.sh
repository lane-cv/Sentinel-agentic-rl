#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision=b98839fc405f31be6ac5ca08008f7cb9289484d6
if [[ $# -gt 1 ]]; then
    echo "用法：bash data/prepare_training_data.sh [本地数据仓库]" >&2
    exit 2
fi

source_repo="${1:-${SENTINEL_DATA_REPO:-}}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

if [[ -z "$source_repo" ]]; then
    for candidate in "$root" "$root/../aiops-agentic-rl"; do
        if git -C "$candidate" cat-file -e "$revision^{commit}" 2>/dev/null; then
            source_repo="$candidate"
            break
        fi
    done
fi

if [[ -z "$source_repo" && -n "${SENTINEL_DATA_REMOTE:-}" ]]; then
    source_repo="$tmp/source"
    git init --bare -q "$source_repo"
    git -C "$source_repo" fetch -q --depth=1 -- "$SENTINEL_DATA_REMOTE" "$revision"
fi

if [[ -z "$source_repo" ]] || ! git -C "$source_repo" cat-file -e "$revision^{commit}" 2>/dev/null; then
    echo "完整数据集需要包含 $revision 的 Git 仓库。请传入本地路径，或设置 SENTINEL_DATA_REPO。" >&2
    echo "仓库内置样例可直接运行 bash smoke_test/run_smoke.sh。" >&2
    exit 2
fi

sft_file=data/cold_start/aiops_cold_start_sft_v9.jsonl
mkdir "$tmp/input"
git -C "$source_repo" archive "$revision" data/cold_start/trajectories_v9_final data/clean/split data/cold_start/ground_truth |
    tar -xf - -C "$tmp/input"

generate() {
    env -i PATH="$PATH" PYTHON_DOTENV_DISABLED=1 PYTHONDONTWRITEBYTECODE=1 python3 "$@"
}

generate "$root/data/cold_start/prefix_split.py" \
    --traj-dir "$tmp/input/data/cold_start/trajectories_v9_final" --out "$tmp/prefixes.json"
generate "$root/data/cold_start/to_llamafactory_format.py" \
    --in "$tmp/prefixes.json" --out "$tmp/aiops_cold_start_sft_v9.jsonl" \
    --dataset-info-out "$tmp/dataset_info_snippet_v9.json"

for split in train held_out; do
    generate "$root/grpo/to_verl_dataset.py" \
        --alerts-dir "$tmp/input/data/clean/split/$split" \
        --ground-truth-dir "$tmp/input/data/cold_start/ground_truth" \
        --split "$split" --out "$tmp/$split.jsonl"
done

(
    cd "$tmp"
    shasum -a 256 -c - <<'CHECKSUMS'
92e8310cc1af3e9a67a44b2a1d1a4c76a12f7f7f98333f6a506fc1d4f50f04e9  aiops_cold_start_sft_v9.jsonl
fa9afc38dfaf94e9fd4e559352c8059815ff8504879b27a5ba2e91dbe95ac0eb  train.jsonl
a050f833129f4e7143cdb1989bf8f776f830313f37c9ff4ae176e53a4cfc496f  held_out.jsonl
CHECKSUMS
)

mkdir -p "$root/grpo/verl_data"
cp "$tmp/aiops_cold_start_sft_v9.jsonl" "$root/$sft_file"
cp "$tmp/train.jsonl" "$tmp/held_out.jsonl" "$root/grpo/verl_data/"
wc -l "$root/$sft_file" "$root/grpo/verl_data/train.jsonl" "$root/grpo/verl_data/held_out.jsonl"
