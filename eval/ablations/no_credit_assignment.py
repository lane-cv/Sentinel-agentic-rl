from __future__ import annotations

import json
from typing import Any

from reward.anti_hacking import group_reward_variance
from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import ALL_SIGNAL_TYPES, Step, Trajectory

GROUP_SIZE = 6

_GROUND_TRUTH = {
    "suspect_service": "recommendation",
    "kind": "deploy_regression",
    "remediation_type": "code_fix",
    "route": "code_fix_pr",
}
_WRONG_ROUTE = "feishu_low_confidence"
_WRONG_DIAGNOSIS_FIELDS = {
    "suspect_service": "ad",
    "kind": "resource",
    "remediation_type": "info_only",
    "also_code_fix": False,
    "evidence": [],
}


def _build_member(member_index: int, n_signal_types: int) -> Trajectory:
    types = list(ALL_SIGNAL_TYPES)[: max(1, min(n_signal_types, len(ALL_SIGNAL_TYPES)))]
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --breadth"},
            hook_decision="passthrough",
            observation=f"member {member_index} covers {len(types)} signal type(s): {types}",
            signal_types_covered=types,
            is_repeat_no_new_info=False,
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --repeat-1"},
            hook_decision="passthrough",
            observation="repeat probe, no new information",
            signal_types_covered=[],
            is_repeat_no_new_info=True,
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "mock-probe --repeat-2"},
            hook_decision="passthrough",
            observation="repeat probe, no new information",
            signal_types_covered=[],
            is_repeat_no_new_info=True,
        ),
    ]
    final_diagnosis = dict(_WRONG_DIAGNOSIS_FIELDS)
    alert = {
        "alertname": "MockCreditAssignmentAblation",
        "service": "recommendation",
        "labels": {"service": "recommendation", "severity": "critical", "job": "recommendation"},
        "annotations": {"summary": "synthetic group for the no-credit-assignment ablation"},
    }
    return Trajectory(
        alert=alert,
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=dict(_GROUND_TRUTH),
        route=_WRONG_ROUTE,
        post_action_health=None,
    )


def build_synthetic_group(group_size: int = GROUP_SIZE) -> list[Trajectory]:
    return [_build_member(i, n_signal_types=i + 1) for i in range(group_size)]


def run(group_size: int = GROUP_SIZE) -> dict[str, Any]:
    group = build_synthetic_group(group_size)

    step_sums = [sum(compute_step_rewards(t)) for t in group]
    credit_sums = [sum(compute_credit_assignment_rewards(t)) for t in group]
    outcome_totals = [compute_outcome_rewards(t)["total"] for t in group]

    if len(set(round(s, 9) for s in step_sums)) != 1:
        raise AssertionError(f"step-level rewards are not identical across the group: {step_sums}")
    if len(set(round(o, 9) for o in outcome_totals)) != 1:
        raise AssertionError(f"outcome rewards are not identical across the group: {outcome_totals}")

    real_totals = [s + c for s, c in zip(step_sums, credit_sums)]
    naive_totals = [s + o for s, o in zip(step_sums, outcome_totals)]

    real_variance = group_reward_variance(real_totals)
    naive_variance = group_reward_variance(naive_totals)

    if naive_variance != 0.0:
        raise AssertionError(
            f"naive (outcome/T redistribution) variance should be exactly 0.0 by "
            f"construction, got {naive_variance}"
        )
    if not (real_variance > 0.0):
        raise AssertionError(
            f"real (step+credit-assignment) variance should be strictly positive, got {real_variance}"
        )

    return {
        "group_size": group_size,
        "step_level_sum_per_member": step_sums,
        "credit_assignment_sum_per_member": credit_sums,
        "outcome_reward_total_per_member": outcome_totals,
        "real_pipeline_total_per_member": real_totals,
        "naive_pipeline_total_per_member": naive_totals,
        "real_pipeline_group_variance": real_variance,
        "naive_pipeline_group_variance": naive_variance,
        "conclusion": (
            "Every trajectory in this group has the same outcome reward. Dividing that "
            "reward across steps gives zero group variance. Credit assignment gives "
            "different totals as signal coverage varies."
        ),
    }


def main() -> None:
    result = run()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
