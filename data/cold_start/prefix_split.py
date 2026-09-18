from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
DEFAULT_TRAJ_DIR = HERE / "trajectories" / "mock"
DEFAULT_OUT_PATH = HERE / "prefix_subsamples.mock.json"


def _is_malformed_step(step: dict[str, Any]) -> bool:
    return "__unparsedToolInput" in json.dumps(step.get("tool_input") or {}, ensure_ascii=False)


def split_trajectory(traj: dict[str, Any]) -> list[dict[str, Any]]:
    steps = [s for s in (traj.get("steps") or []) if not _is_malformed_step(s)]
    n = len(steps)
    alert = traj.get("alert")
    alert_id = traj.get("alert_id")
    subsamples: list[dict[str, Any]] = []

    for i in range(1, n + 1):
        if i < n:
            prefix_steps = steps[0 : i - 1]
            target_type = "tool_call"
            target = {"tool_name": steps[i - 1]["tool_name"], "tool_input": steps[i - 1]["tool_input"]}
        else:
            prefix_steps = steps[0:n]
            target_type = "diagnosis"
            target = traj.get("final_diagnosis")

        subsamples.append(
            {
                "alert_id": alert_id,
                "alert": alert,
                "step_index": i,
                "total_steps": n,
                "prefix_steps": prefix_steps,
                "target_type": target_type,
                "target": target,
            }
        )
    return subsamples


def split_directory(traj_dir: Path) -> list[dict[str, Any]]:
    all_subsamples: list[dict[str, Any]] = []
    for f in sorted(Path(traj_dir).glob("*.json")):
        traj = json.loads(f.read_text(encoding="utf-8"))
        all_subsamples.extend(split_trajectory(traj))
    return all_subsamples


def main() -> None:
    parser = argparse.ArgumentParser(description="把冷启轨迹按前缀拆成 step-level 训练子样本")
    parser.add_argument("--traj-dir", type=Path, default=DEFAULT_TRAJ_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    args = parser.parse_args()

    subsamples = split_directory(args.traj_dir)
    args.out.write_text(json.dumps(subsamples, ensure_ascii=False, indent=2), encoding="utf-8")

    n_traj = len(list(Path(args.traj_dir).glob("*.json")))
    n_diag_targets = sum(1 for s in subsamples if s["target_type"] == "diagnosis")
    print(f"[prefix_split] {n_traj} 条轨迹 -> {len(subsamples)} 条子样本（其中 {n_diag_targets} 条是 diagnosis 收敛目标）")
    print(f"[prefix_split] 写入 {args.out}")


if __name__ == "__main__":
    main()
