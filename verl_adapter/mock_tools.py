from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from reward.trajectory import Step, Trajectory
from verl_adapter import _repo_paths

DEFAULT_MOCK_TRAJ_DIR = _repo_paths.REPO_ROOT / "data" / "cold_start" / "trajectories" / "mock"
DEFAULT_GROUND_TRUTH_DIR = _repo_paths.REPO_ROOT / "data" / "cold_start" / "ground_truth"


async def _real_hook_verdict_for_bash_async(command: str) -> dict[str, Any]:
    remediation, guard_diagnose_ops = _repo_paths.load_real_hook_modules()
    input_data = {"tool_name": "Bash", "tool_input": {"command": command}, "hook_event_name": "PreToolUse"}
    result = await guard_diagnose_ops(input_data, "mock-tool-use-id", {})
    decision_raw = (result.get("hookSpecificOutput") or {}).get("permissionDecision")
    decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
    verdict = remediation.classify_command(command)
    action = target = None
    if verdict.get("decision") in ("allow", "deny"):
        action, target = verdict.get("action"), verdict.get("target")
    return {"hook_decision": decision, "hook_action": action, "hook_target": target}


def real_hook_verdict_for_bash(command: str) -> dict[str, Any]:
    return asyncio.run(_real_hook_verdict_for_bash_async(command))


@dataclass
class MockStepResult:
    observation: str
    hook_decision: str
    hook_action: Optional[str]
    hook_target: Optional[str]


class MockToolBackend:

    def __init__(self, raw_steps: list[dict[str, Any]]) -> None:
        self._raw_steps = raw_steps
        self._cursor = 0

    @property
    def exhausted(self) -> bool:
        return self._cursor >= len(self._raw_steps)

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> MockStepResult:
        if self.exhausted:
            raise RuntimeError("MockToolBackend: no more canned steps to play back for this scenario")
        expected = self._raw_steps[self._cursor]
        if expected["tool_name"] != tool_name:
            raise ValueError(
                f"MockToolBackend step {self._cursor}: expected tool_name={expected['tool_name']!r}, "
                f"got {tool_name!r} (canned scenarios replay a fixed order)"
            )
        self._cursor += 1
        observation = expected["observation"]
        if tool_name == "Bash":
            command = str(tool_input.get("command", ""))
            verdict = await _real_hook_verdict_for_bash_async(command)
        else:
            verdict = {"hook_decision": "passthrough", "hook_action": None, "hook_target": None}
        return MockStepResult(observation=observation, **verdict)


def load_raw_mock_trajectory(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _load_ground_truth(alert_id: str, gt_dir: Optional[Path] = None) -> dict[str, Any]:
    gt_dir = Path(gt_dir) if gt_dir is not None else DEFAULT_GROUND_TRUTH_DIR
    gt_path = gt_dir / f"{alert_id}.json"
    raw = json.loads(gt_path.read_text(encoding="utf-8")) if gt_path.exists() else {}
    out = dict(raw)
    if "expect_route" in out:
        out["route"] = out.pop("expect_route")
    return out


def _translate_post_action_health(raw: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    if raw is None:
        return None
    if not raw.get("checked", True):
        return None
    return {
        "healthy_at_60s": bool(raw.get("healthy", False)),
        "new_alert_triggered": bool(raw.get("new_alert_triggered", False)),
    }


async def run_mock_rollout(raw_traj: dict[str, Any], gt_dir: Optional[Path] = None) -> Trajectory:
    backend = MockToolBackend(raw_traj["steps"])
    steps: list[dict[str, Any]] = []
    for raw_step in raw_traj["steps"]:
        result = await backend.execute(raw_step["tool_name"], raw_step["tool_input"])
        steps.append(
            {
                "tool_name": raw_step["tool_name"],
                "tool_input": raw_step["tool_input"],
                "hook_decision": result.hook_decision,
                "hook_action": result.hook_action,
                "hook_target": result.hook_target,
                "observation": result.observation,
            }
        )

    signals_mod = _repo_paths.load_signals_module()
    signals_mod.annotate_steps(steps)

    ground_truth = _load_ground_truth(raw_traj["alert_id"], gt_dir=gt_dir)
    post_action_health = _translate_post_action_health(raw_traj.get("post_action_health"))

    trajectory_steps = [
        Step(
            tool_name=s["tool_name"],
            tool_input=s["tool_input"],
            hook_decision=s["hook_decision"],
            observation=s["observation"],
            hook_action=s["hook_action"],
            hook_target=s["hook_target"],
            signal_types_covered=s["signal_types_covered"],
            is_repeat_no_new_info=s["is_repeat_no_new_info"],
        )
        for s in steps
    ]
    return Trajectory(
        alert=raw_traj["alert"],
        steps=trajectory_steps,
        final_diagnosis=raw_traj["final_diagnosis"],
        ground_truth=ground_truth,
        route=raw_traj["route"],
        post_action_health=post_action_health,
    )


def load_all_mock_trajectories(
    mock_dir: Optional[Path] = None,
    gt_dir: Optional[Path] = None,
) -> list[Trajectory]:
    mock_dir = Path(mock_dir) if mock_dir is not None else DEFAULT_MOCK_TRAJ_DIR
    trajectories = []
    for f in sorted(mock_dir.glob("*.json")):
        raw = load_raw_mock_trajectory(f)
        trajectories.append(asyncio.run(run_mock_rollout(raw, gt_dir=gt_dir)))
    return trajectories
