from __future__ import annotations

import json
import re
import sys
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __name__ not in sys.modules:
    import types as _types

    _self_shim = _types.ModuleType(__name__)
    _self_shim.__dict__.update(globals())
    sys.modules[__name__] = _self_shim

from grpo.reward_router import (
    compute_group_rewards,
    compute_trajectory_reward,
    route_trajectory_reward_to_variance_tracker,
)
from reward.trajectory import Step, Trajectory

__all__ = [
    "DEFAULT_GROUP_SIZE",
    "ROUTE_CONFIDENCE_THRESHOLD",
    "ParsedRollout",
    "compute_score",
    "extract_last_json_object",
    "infer_route_optimistic",
    "parse_solution_str",
    "parse_tool_call_block",
    "recompute_hook_verdict",
    "reset_group_accumulator",
    "trajectory_from_solution_str",
]

DEFAULT_GROUP_SIZE = 6

ROUTE_CONFIDENCE_THRESHOLD = 0.6


_TOOL_EVENT_RE = re.compile(
    r"<tool_call>.*?</tool_call>|<tool_response>.*?</tool_response>", re.DOTALL
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

_FUNCTION_NAME_RE = re.compile(r"<\s*function\s*=\s*([^<>]+?)\s*>")

_PARAM_RE = re.compile(
    r"<\s*parameter\s*=\s*([^>]*)>"
    r"(.*?)"
    r"(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))",
    re.DOTALL,
)


def _trim_wrapping_newlines(value: str) -> str:
    if value.startswith("\n"):
        value = value[1:]
    if value.endswith("\n"):
        value = value[:-1]
    return value


@dataclass
class ParsedRollout:

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    final_text: str = ""
    n_dropped_malformed_calls: int = 0


def parse_tool_call_block(block_text: str) -> Optional[dict[str, Any]]:
    name_match = _FUNCTION_NAME_RE.search(block_text)
    if name_match is None:
        return None
    tool_input: dict[str, Any] = {}
    for param_match in _PARAM_RE.finditer(block_text):
        param_name = param_match.group(1).strip()
        tool_input[param_name] = _trim_wrapping_newlines(param_match.group(2))
    return {"tool_name": name_match.group(1), "tool_input": tool_input}


def _strip_reasoning_and_role_words(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text)
    if "<think>" in text:
        text = text.split("<think>", 1)[0]
    lines = [ln for ln in text.split("\n") if ln.strip() not in ("assistant", "user")]
    return "\n".join(lines).strip()


def parse_solution_str(solution_str: str) -> ParsedRollout:
    parsed = ParsedRollout()
    events = list(_TOOL_EVENT_RE.finditer(solution_str or ""))
    last_end = 0
    for match in events:
        block = match.group(0)
        if block.startswith("<tool_call>"):
            call = parse_tool_call_block(block)
            if call is None:
                parsed.n_dropped_malformed_calls += 1
            else:
                parsed.tool_calls.append(call)
        else:
            parsed.observations.append(block[len("<tool_response>") : -len("</tool_response>")].strip("\n"))
        last_end = match.end()

    if len(parsed.observations) > len(parsed.tool_calls):
        parsed.observations = parsed.observations[: len(parsed.tool_calls)]
    while len(parsed.observations) < len(parsed.tool_calls):
        parsed.observations.append("")

    tail = solution_str[last_end:] if events else (solution_str or "")
    parsed.final_text = _strip_reasoning_and_role_words(tail)
    return parsed


def extract_last_json_object(text: str) -> Optional[dict[str, Any]]:
    candidates: list[str] = []
    depth = 0
    start: Optional[int] = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    candidates.append(text[start : i + 1])
    for candidate in reversed(candidates):
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


_REAL_MODULES: Optional[tuple[Any, Any, Any]] = None
_REAL_MODULES_LOCK = threading.Lock()


def _load_real_modules() -> tuple[Any, Any, Any]:
    global _REAL_MODULES
    with _REAL_MODULES_LOCK:
        if _REAL_MODULES is not None:
            return _REAL_MODULES
        aiops_dir = _REPO_ROOT / "Sentinel-agent"
        cold_start_dir = _REPO_ROOT / "data" / "cold_start"
        for p in (str(aiops_dir), str(cold_start_dir)):
            if p not in sys.path:
                sys.path.insert(0, p)
        try:
            from agent.core import remediation, schema

            import signals
        except Exception as exc:
            raise RuntimeError(
                "grpo.verl_reward_adapter needs the real Sentinel-agent core modules "
                "(agent.core.remediation / agent.core.schema) and "
                "data/cold_start/signals.py to reconstruct hook verdicts and the "
                f"Diagnosis; import failed from {_REPO_ROOT}: {exc!r}"
            ) from exc
        _REAL_MODULES = (remediation, schema, signals)
        return _REAL_MODULES


def recompute_hook_verdict(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    if tool_name != "Bash":
        return {"decision": "passthrough", "action": None, "target": None}
    remediation, _, _ = _load_real_modules()
    command = str((tool_input or {}).get("command", ""))
    verdict = remediation.classify_command(command)
    decision = verdict.get("decision")
    action = target = None
    if decision in ("allow", "deny"):
        action, target = verdict.get("action"), verdict.get("target")
    return {"decision": decision, "action": action, "target": target}


def _parse_final_diagnosis(final_text: str) -> tuple[Any, bool]:
    _, schema, _ = _load_real_modules()
    obj = extract_last_json_object(final_text)
    if obj is None:
        return (
            schema.Diagnosis.fallback(
                f"final turn contained no parseable Diagnosis JSON; raw tail={final_text[-200:]!r}"
            ),
            True,
        )
    try:
        return schema.Diagnosis.model_validate(obj), False
    except Exception as exc:
        return (
            schema.Diagnosis.fallback(
                f"Diagnosis schema validation failed: {exc}; raw={str(obj)[:200]!r}"
            ),
            True,
        )


def infer_route_optimistic(final_diagnosis: dict[str, Any]) -> str:
    confidence = final_diagnosis.get("confidence")
    if confidence is None or float(confidence) < ROUTE_CONFIDENCE_THRESHOLD:
        return "feishu_low_confidence"
    remediation_type = final_diagnosis.get("remediation_type")
    auto_done = bool(final_diagnosis.get("executed_actions"))
    if remediation_type == "online_op":
        if final_diagnosis.get("also_code_fix"):
            return "auto_remediated_and_code_fix_pr" if auto_done else "online_op_and_code_fix_pr"
        return "auto_remediated" if auto_done else "feishu_online_op"
    if remediation_type == "code_fix":
        return "code_fix_pr"
    return "info_only"


def trajectory_from_solution_str(
    solution_str: str,
    ground_truth: Optional[dict[str, Any]] = None,
    extra_info: Optional[dict[str, Any]] = None,
) -> Trajectory:
    remediation, schema, signals_mod = _load_real_modules()
    parsed = parse_solution_str(solution_str)

    raw_steps: list[dict[str, Any]] = []
    for call, observation in zip(parsed.tool_calls, parsed.observations):
        verdict = recompute_hook_verdict(call["tool_name"], call["tool_input"])
        raw_steps.append(
            {
                "tool_name": call["tool_name"],
                "tool_input": call["tool_input"],
                "hook_decision": verdict["decision"],
                "hook_action": verdict["action"],
                "hook_target": verdict["target"],
                "observation": observation,
            }
        )
    signals_mod.annotate_steps(raw_steps)
    steps = [
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
        for s in raw_steps
    ]

    diag, schema_fallback = _parse_final_diagnosis(parsed.final_text)
    diag.executed_actions = [
        schema.ExecutedAction(
            action=s["hook_action"] or "",
            target=s["hook_target"] or "",
            command=str(s["tool_input"].get("command", "")),
        )
        for s in raw_steps
        if s["hook_decision"] == "allow"
    ]
    final_diagnosis = diag.model_dump(by_alias=True)
    final_diagnosis["schema_fallback_triggered"] = schema_fallback

    extra = extra_info if isinstance(extra_info, dict) else {}
    alert = extra.get("alert") if isinstance(extra.get("alert"), dict) else {}

    return Trajectory(
        alert=alert,
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=dict(ground_truth) if isinstance(ground_truth, dict) else {},
        route=infer_route_optimistic(final_diagnosis),
        post_action_health=None,
    )


class _GroupAccumulator:

    def __init__(self, group_size: int = DEFAULT_GROUP_SIZE) -> None:
        self.group_size = group_size
        self._pending: dict[Any, list[Trajectory]] = defaultdict(list)
        self._last_group_result: dict[Any, dict[str, Any]] = {}
        self.variance_history: dict[str, list[float]] = {}
        self.last_tracker_result: Optional[dict[str, Any]] = None
        self._lock = threading.Lock()

    def add(self, index: Any, traj: Trajectory) -> Optional[dict[str, Any]]:
        with self._lock:
            bucket = self._pending[index]
            bucket.append(traj)
            if len(bucket) < self.group_size:
                return None
            trajectories = self._pending.pop(index)
        return self._flush(index, trajectories)

    def _flush(self, index: Any, trajectories: list[Trajectory]) -> dict[str, Any]:
        result = compute_group_rewards(trajectories)
        with self._lock:
            self._last_group_result[index] = result
            scenario_type = str(trajectories[0].ground_truth.get("kind") or "unknown") if trajectories else "unknown"
            tracker = route_trajectory_reward_to_variance_tracker(
                scenario_type, result["total_rewards"], self.variance_history
            )
            self.variance_history = tracker["updated_variance_history"]
            self.last_tracker_result = tracker
        return result

    def pending(self, index: Any) -> list[Trajectory]:
        with self._lock:
            return list(self._pending.get(index, []))

    def last_group_result(self, index: Any) -> Optional[dict[str, Any]]:
        with self._lock:
            return self._last_group_result.get(index)

    def reset(self) -> None:
        with self._lock:
            self._pending.clear()
            self._last_group_result.clear()
            self.variance_history = {}
            self.last_tracker_result = None


_GROUP_ACCUMULATOR = _GroupAccumulator()


def reset_group_accumulator() -> None:
    _GROUP_ACCUMULATOR.reset()


def compute_score(
    data_source: str = "",
    solution_str: str = "",
    ground_truth: Optional[dict[str, Any]] = None,
    extra_info: Optional[dict[str, Any]] = None,
) -> float:
    traj = trajectory_from_solution_str(solution_str, ground_truth, extra_info)
    breakdown = compute_trajectory_reward(traj)
    index = extra_info.get("index") if isinstance(extra_info, dict) else None
    if index is not None:
        _GROUP_ACCUMULATOR.add(index, traj)
    return float(breakdown["total_reward"])
