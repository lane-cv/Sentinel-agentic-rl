from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AIOPS_AGENT_DIR = REPO_ROOT / "Sentinel-agent"

DEFAULT_ALERTS_DIR = REPO_ROOT / "data" / "clean" / "split" / "train"
DEFAULT_GROUND_TRUTH_DIR = REPO_ROOT / "data" / "cold_start" / "ground_truth"
DEFAULT_OUT = HERE / "verl_data" / "train.jsonl"

DATA_SOURCE = "aiops_diagnosis"
ABILITY = "aiops_diagnosis"

_GT_KEY_MAP = {
    "expect_route": "route",
    "remediation_type": "remediation_type",
    "suspect_service": "suspect_service",
    "kind": "kind",
    "suspect_repo": "suspect_repo",
    "suspect_commit_hint": "suspect_commit_hint",
    "suspect_file_hint": "suspect_file_hint",
}


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_prompt_fns():
    try:
        _ensure_aiops_agent_on_path()
        from agent.agents import prompts

        return prompts.diagnose_prompt, prompts.diagnose_append
    except Exception:

        def _fallback_prompt(alert: dict[str, Any]) -> str:
            return f"请诊断以下告警：\n```json\n{json.dumps(alert, ensure_ascii=False, indent=2)}\n```\n"

        def _fallback_append() -> str:
            return "你是 AIOps 故障诊断处置 Agent，只读排查后输出结构化 Diagnosis。"

        return _fallback_prompt, _fallback_append


_diagnose_prompt, _diagnose_append = _load_prompt_fns()
SYSTEM_PROMPT = _diagnose_append()


def strip_scenario_hint(alert: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in alert.items() if k != "scenario_hint"}


def map_ground_truth(gt: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in gt.items():
        if key == "expect_also_code_fix":
            continue
        out[_GT_KEY_MAP.get(key, key)] = value
    return out


def build_prompt_messages(alert: dict[str, Any]) -> list[dict[str, str]]:
    alert_for_prompt = strip_scenario_hint(alert)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _diagnose_prompt(alert_for_prompt)},
    ]


def build_row(alert_id: str, alert: dict[str, Any], ground_truth: dict[str, Any], index: int, split: str) -> dict[str, Any]:
    return {
        "data_source": DATA_SOURCE,
        "prompt": build_prompt_messages(alert),
        "ability": ABILITY,
        "reward_model": {"style": "rule", "ground_truth": map_ground_truth(ground_truth)},
        "extra_info": {
            "split": split,
            "index": index,
            "alert_id": alert_id,
            "alert": strip_scenario_hint(alert),
        },
    }


def convert(alerts_dir: Path, ground_truth_dir: Path, split: str) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    skipped: list[str] = []
    alert_files = sorted(alerts_dir.glob("*.json"))
    for idx, alert_path in enumerate(alert_files):
        alert_id = alert_path.stem
        gt_path = ground_truth_dir / f"{alert_id}.json"
        if not gt_path.is_file():
            skipped.append(alert_id)
            continue
        alert = json.loads(alert_path.read_text(encoding="utf-8"))
        ground_truth = json.loads(gt_path.read_text(encoding="utf-8"))
        rows.append(build_row(alert_id, alert, ground_truth, idx, split))
    return rows, skipped


def write_jsonl(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="把告警 + ground truth 转成 veRL GRPO 能读的 jsonl 数据集"
    )
    parser.add_argument("--alerts-dir", type=Path, default=DEFAULT_ALERTS_DIR)
    parser.add_argument("--ground-truth-dir", type=Path, default=DEFAULT_GROUND_TRUTH_DIR)
    parser.add_argument("--split", type=str, default="train", help="写进 extra_info.split 的标签")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows, skipped = convert(args.alerts_dir, args.ground_truth_dir, args.split)
    write_jsonl(rows, args.out)

    print(f"[to_verl_dataset] {len(rows)} 条 -> {args.out}")
    if skipped:
        print(
            f"[to_verl_dataset] 跳过 {len(skipped)} 条（{args.alerts_dir} 下没有对应的 "
            f"ground truth 文件，不能参与 reward 计算，未写入输出）："
        )
        for alert_id in skipped:
            print(f"  - {alert_id}")


if __name__ == "__main__":
    main()
