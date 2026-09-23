from __future__ import annotations

from typing import Optional

from reward.trajectory import Trajectory

STOP_LOSS_SUCCESS_REWARD = 0.5
STOP_LOSS_NOT_RECOVERED_REWARD = 0.0
STOP_LOSS_NEW_ALERT_PENALTY = -0.3

ROOT_CAUSE_FIELD_REWARD = 0.1
ROOT_CAUSE_FIELDS = ("suspect_service", "kind", "remediation_type")

ROUTE_MATCH_REWARD = 0.2

HANDOFF_FIELD_REWARD = 0.05
HANDOFF_FIELDS = ("suspect_repo", "suspect_commit_hint", "suspect_file_hint")

_FORBIDDEN_FIX_RESULT_KEYS = ("verified", "fix_result", "fixresult", "pr_passed_ci")


def _assert_no_fix_result_leak(traj: Trajectory) -> None:
    for bucket_name, bucket in (
        ("final_diagnosis", traj.final_diagnosis),
        ("ground_truth", traj.ground_truth),
    ):
        leaked = [k for k in _FORBIDDEN_FIX_RESULT_KEYS if k in bucket]
        if leaked:
            raise ValueError(
                f"outcome_reward: {bucket_name} contains fix-agent-result key(s) "
                f"{leaked} — FixResult.verified must NEVER be used as diagnosis-agent "
                f"outcome reward input (doc §7.2). Remove it before scoring."
            )


def stop_loss_reward(traj: Trajectory) -> Optional[float]:
    _assert_no_fix_result_leak(traj)

    if traj.final_diagnosis.get("remediation_type") != "online_op":
        return None

    health = traj.post_action_health
    if health is None:
        return STOP_LOSS_NOT_RECOVERED_REWARD
    if health.get("new_alert_triggered"):
        return STOP_LOSS_NEW_ALERT_PENALTY
    if health.get("healthy_at_60s"):
        return STOP_LOSS_SUCCESS_REWARD
    return STOP_LOSS_NOT_RECOVERED_REWARD


def root_cause_field_rewards(traj: Trajectory) -> dict[str, float]:
    out: dict[str, float] = {}
    for field_name in ROOT_CAUSE_FIELDS:
        predicted = traj.final_diagnosis.get(field_name)
        expected = traj.ground_truth.get(field_name)
        out[field_name] = (
            ROOT_CAUSE_FIELD_REWARD
            if expected is not None and predicted == expected
            else 0.0
        )
    return out


def root_cause_reward(traj: Trajectory) -> float:
    return sum(root_cause_field_rewards(traj).values())


def route_match_reward(traj: Trajectory) -> float:
    expected = traj.ground_truth.get("route")
    if expected is not None and traj.route == expected:
        return ROUTE_MATCH_REWARD
    return 0.0


def _handoff_applicable(traj: Trajectory) -> bool:
    remediation_type = traj.final_diagnosis.get("remediation_type")
    also_code_fix = bool(traj.final_diagnosis.get("also_code_fix"))
    return remediation_type == "code_fix" or also_code_fix


def handoff_field_rewards(traj: Trajectory) -> dict[str, float]:
    if not _handoff_applicable(traj):
        return {f: 0.0 for f in HANDOFF_FIELDS}
    out: dict[str, float] = {}
    for field_name in HANDOFF_FIELDS:
        predicted = traj.final_diagnosis.get(field_name)
        expected = traj.ground_truth.get(field_name)
        out[field_name] = (
            HANDOFF_FIELD_REWARD
            if expected is not None and predicted is not None and predicted == expected
            else 0.0
        )
    return out


def handoff_reward(traj: Trajectory) -> float:
    return sum(handoff_field_rewards(traj).values())


def llm_judge_coherence(traj: Trajectory) -> Optional[float]:
    return None


def compute_outcome_rewards(traj: Trajectory) -> dict[str, object]:
    _assert_no_fix_result_leak(traj)

    stop_loss = stop_loss_reward(traj)
    root_cause = root_cause_reward(traj)
    route = route_match_reward(traj)
    handoff = handoff_reward(traj)

    total = (stop_loss or 0.0) + root_cause + route + handoff

    return {
        "stop_loss_reward": stop_loss,
        "root_cause_field_rewards": root_cause_field_rewards(traj),
        "root_cause_reward": root_cause,
        "route_match_reward": route,
        "handoff_field_rewards": handoff_field_rewards(traj),
        "handoff_reward": handoff,
        "llm_judge_coherence": llm_judge_coherence(traj),
        "total": total,
    }
