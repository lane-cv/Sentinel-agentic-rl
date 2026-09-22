from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


HookDecision = Literal["allow", "deny", "passthrough"]
HOOK_DECISIONS: tuple[str, ...] = ("allow", "deny", "passthrough")

HookAction = Literal["restart_instance", "migrate_instance", "scale_resources"]
HOOK_ACTIONS: tuple[str, ...] = ("restart_instance", "migrate_instance", "scale_resources")

SignalType = Literal[
    "prometheus", "jaeger", "logs", "kubectl_status", "git_history", "rag_history"
]
ALL_SIGNAL_TYPES: tuple[str, ...] = (
    "prometheus", "jaeger", "logs", "kubectl_status", "git_history", "rag_history",
)

Kind = Literal["dependency", "resource", "deploy_regression", "config"]
ALL_KINDS: tuple[str, ...] = ("dependency", "resource", "deploy_regression", "config")

RemediationType = Literal["online_op", "code_fix", "info_only"]
ALL_REMEDIATION_TYPES: tuple[str, ...] = ("online_op", "code_fix", "info_only")

ALL_ROUTES: tuple[str, ...] = (
    "skipped_duplicate",
    "feishu_low_confidence",
    "auto_remediated",
    "auto_remediated_and_code_fix_pr",
    "online_op_and_code_fix_pr",
    "feishu_online_op",
    "code_fix_pr",
    "feishu_fix_unverified",
    "info_only",
)


@dataclass
class Step:

    tool_name: str
    tool_input: dict[str, Any]
    hook_decision: HookDecision
    observation: str = ""
    hook_action: Optional[str] = None
    hook_target: Optional[str] = None
    signal_types_covered: list[str] = field(default_factory=list)
    is_repeat_no_new_info: bool = False
    tool_call_schema_valid: bool = True

    def __post_init__(self) -> None:
        if self.hook_decision not in HOOK_DECISIONS:
            raise ValueError(
                f"Step.hook_decision must be one of {HOOK_DECISIONS}, got {self.hook_decision!r}"
            )
        if self.hook_action is not None and self.hook_action not in HOOK_ACTIONS:
            raise ValueError(
                f"Step.hook_action must be one of {HOOK_ACTIONS} or None, got {self.hook_action!r}"
            )
        bad = [s for s in self.signal_types_covered if s not in ALL_SIGNAL_TYPES]
        if bad:
            raise ValueError(
                f"Step.signal_types_covered contains unknown signal type(s) {bad}; "
                f"must be a subset of {ALL_SIGNAL_TYPES}"
            )


@dataclass
class Trajectory:

    alert: dict[str, Any]
    steps: list[Step]
    final_diagnosis: dict[str, Any]
    ground_truth: dict[str, Any]
    route: str
    post_action_health: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.route not in ALL_ROUTES:
            raise ValueError(f"Trajectory.route must be one of {ALL_ROUTES}, got {self.route!r}")
        gt_route = self.ground_truth.get("route")
        if gt_route is not None and gt_route not in ALL_ROUTES:
            raise ValueError(
                f"Trajectory.ground_truth['route'] must be one of {ALL_ROUTES} or absent, "
                f"got {gt_route!r}"
            )
        gt_kind = self.ground_truth.get("kind")
        if gt_kind is not None and gt_kind not in ALL_KINDS:
            raise ValueError(
                f"Trajectory.ground_truth['kind'] must be one of {ALL_KINDS} or absent, got {gt_kind!r}"
            )
        gt_rt = self.ground_truth.get("remediation_type")
        if gt_rt is not None and gt_rt not in ALL_REMEDIATION_TYPES:
            raise ValueError(
                f"Trajectory.ground_truth['remediation_type'] must be one of "
                f"{ALL_REMEDIATION_TYPES} or absent, got {gt_rt!r}"
            )
        if self.post_action_health is not None:
            missing = {"healthy_at_60s", "new_alert_triggered"} - set(self.post_action_health)
            if missing:
                raise ValueError(
                    f"Trajectory.post_action_health is missing required key(s) {missing}"
                )


    @property
    def schema_fallback_triggered(self) -> bool:
        return bool(self.final_diagnosis.get("schema_fallback_triggered", False))

    def all_observations(self) -> list[str]:
        return [s.observation for s in self.steps]

    def all_signal_types_covered(self) -> set[str]:
        out: set[str] = set()
        for s in self.steps:
            out.update(s.signal_types_covered)
        return out
