from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from reward.trajectory import Step, Trajectory

REPO_ROOT = Path(__file__).resolve().parents[2]
MOCK_TRAJECTORY_DIR = REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"

_STEP_FIELDS = {
    "tool_name",
    "tool_input",
    "hook_decision",
    "observation",
    "hook_action",
    "hook_target",
    "signal_types_covered",
    "is_repeat_no_new_info",
    "tool_call_schema_valid",
}


def _step_from_dict(raw_step: dict[str, Any]) -> Step:
    return Step(**{k: v for k, v in raw_step.items() if k in _STEP_FIELDS})


def _post_action_health_from_dict(raw_health: dict[str, Any] | None) -> dict[str, Any] | None:
    if raw_health is None:
        return None
    return {
        "healthy_at_60s": bool(raw_health.get("healthy", False)),
        "new_alert_triggered": False,
    }


def load_mock_trajectory(path: Path) -> Trajectory:
    raw = json.loads(path.read_text())
    steps = [_step_from_dict(s) for s in raw["steps"]]
    return Trajectory(
        alert=raw["alert"],
        steps=steps,
        final_diagnosis=raw["final_diagnosis"],
        ground_truth=raw.get("ground_truth") or {},
        route=raw["route"],
        post_action_health=_post_action_health_from_dict(raw.get("post_action_health")),
    )


def load_all_mock_trajectories() -> dict[str, Trajectory]:
    return {p.stem: load_mock_trajectory(p) for p in sorted(MOCK_TRAJECTORY_DIR.glob("*.json"))}


def with_signal_coverage(traj: Trajectory, coverage_per_step: list[list[str]]) -> Trajectory:
    if len(coverage_per_step) != len(traj.steps):
        raise ValueError(
            f"coverage_per_step must have one entry per step "
            f"({len(traj.steps)} steps, got {len(coverage_per_step)} entries)"
        )
    out = copy.deepcopy(traj)
    for step, types in zip(out.steps, coverage_per_step):
        step.signal_types_covered = list(types)
    return out
