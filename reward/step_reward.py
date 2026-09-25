from __future__ import annotations

from reward.anti_hacking import has_fabricated_evidence
from reward.trajectory import Step, Trajectory

TOOL_SCHEMA_VALID_REWARD = 0.05
WHITELIST_ALLOW_AUTHORIZED_REWARD = 0.1
WHITELIST_DENY_PENALTY = -0.15
OBSERVATION_ADVANCES_DIAGNOSIS_REWARD = 0.1
REPEAT_NO_NEW_INFO_PENALTY = -0.15
EVIDENCE_FABRICATION_PENALTY = -0.5
SCHEMA_FALLBACK_UNRECOVERED_PENALTY = -0.2


def tool_call_schema_valid_reward(step: Step) -> float:
    return TOOL_SCHEMA_VALID_REWARD if step.tool_call_schema_valid else 0.0


def whitelist_allow_authorized_reward(step: Step) -> float:
    return WHITELIST_ALLOW_AUTHORIZED_REWARD if step.hook_decision == "allow" else 0.0


def whitelist_deny_penalty(step: Step) -> float:
    return WHITELIST_DENY_PENALTY if step.hook_decision == "deny" else 0.0


def observation_advances_diagnosis_reward(step: Step, prior_steps: list[Step]) -> float:
    if step.is_repeat_no_new_info:
        return 0.0
    already_covered: set[str] = set()
    for s in prior_steps:
        already_covered.update(s.signal_types_covered)
    new_types = set(step.signal_types_covered) - already_covered
    return OBSERVATION_ADVANCES_DIAGNOSIS_REWARD if new_types else 0.0


def repeat_no_new_info_penalty(step: Step) -> float:
    return REPEAT_NO_NEW_INFO_PENALTY if step.is_repeat_no_new_info else 0.0


def evidence_fabrication_penalty(traj: Trajectory) -> float:
    return EVIDENCE_FABRICATION_PENALTY if has_fabricated_evidence(traj) else 0.0


def schema_fallback_unrecovered_penalty(traj: Trajectory) -> float:
    return SCHEMA_FALLBACK_UNRECOVERED_PENALTY if traj.schema_fallback_triggered else 0.0


def compute_step_rewards(traj: Trajectory) -> list[float]:
    rewards: list[float] = []
    for i, step in enumerate(traj.steps):
        prior = traj.steps[:i]
        r = (
            tool_call_schema_valid_reward(step)
            + whitelist_allow_authorized_reward(step)
            + whitelist_deny_penalty(step)
            + observation_advances_diagnosis_reward(step, prior)
            + repeat_no_new_info_penalty(step)
        )
        rewards.append(r)

    if rewards:
        rewards[-1] += evidence_fabrication_penalty(traj) + schema_fallback_unrecovered_penalty(traj)

    return rewards
