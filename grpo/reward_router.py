from __future__ import annotations

from typing import Any

from reward.anti_hacking import (
    dual_source_coverage_penalty,
    group_reward_variance,
    should_downsample,
)
from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Trajectory


TURN_COUNT_PENALTY_THRESHOLD = 8

TURN_COUNT_PENALTY_PER_EXTRA_TURN = -0.02

MAX_TURNS = 12

MAX_TURNS_TRUNCATION_PENALTY = -0.5


def turn_count_penalty(traj: Trajectory) -> float:
    n = len(traj.steps)
    if n <= TURN_COUNT_PENALTY_THRESHOLD:
        return 0.0
    return (n - TURN_COUNT_PENALTY_THRESHOLD) * TURN_COUNT_PENALTY_PER_EXTRA_TURN


def max_turns_truncation_penalty(traj: Trajectory) -> float:
    return MAX_TURNS_TRUNCATION_PENALTY if len(traj.steps) > MAX_TURNS else 0.0


def compute_per_step_rewards(traj: Trajectory) -> list[float]:
    step = compute_step_rewards(traj)
    credit = compute_credit_assignment_rewards(traj)
    assert len(step) == len(credit) == len(traj.steps), (
        "compute_step_rewards and compute_credit_assignment_rewards must both "
        f"return one value per step; got {len(step)} and {len(credit)} for "
        f"{len(traj.steps)} steps"
    )
    return [s + c for s, c in zip(step, credit)]


def compute_trajectory_reward(traj: Trajectory) -> dict[str, Any]:
    per_step = compute_per_step_rewards(traj)

    outcome = compute_outcome_rewards(traj)
    outcome_total = float(outcome["total"])
    dual_source_penalty = dual_source_coverage_penalty(traj)
    turn_penalty = turn_count_penalty(traj)
    truncation_penalty = max_turns_truncation_penalty(traj)

    terminal_addition = outcome_total + dual_source_penalty + turn_penalty + truncation_penalty

    if per_step:
        per_step[-1] += terminal_addition
    else:
        per_step = [terminal_addition]

    total_reward = sum(per_step)

    return {
        "per_step_rewards": per_step,
        "total_reward": total_reward,
        "outcome_breakdown": outcome,
        "dual_source_coverage_penalty": dual_source_penalty,
        "turn_count_penalty": turn_penalty,
        "max_turns_truncation_penalty": truncation_penalty,
    }


def compute_group_rewards(trajectories: list[Trajectory]) -> dict[str, Any]:
    per_traj = [compute_trajectory_reward(t) for t in trajectories]
    total_rewards = [r["total_reward"] for r in per_traj]
    return {
        "trajectory_rewards": per_traj,
        "total_rewards": total_rewards,
        "group_reward_variance": group_reward_variance(total_rewards),
        "group_size": len(trajectories),
    }


def route_trajectory_reward_to_variance_tracker(
    scenario_type: str,
    group_total_rewards: list[float],
    variance_history: dict[str, list[float]],
) -> dict[str, Any]:
    variance = group_reward_variance(group_total_rewards)
    updated_history = dict(variance_history)
    updated_history[scenario_type] = list(variance_history.get(scenario_type, [])) + [variance]
    downsample = should_downsample(updated_history[scenario_type])
    return {
        "scenario_type": scenario_type,
        "variance": variance,
        "updated_variance_history": updated_history,
        "should_downsample": downsample,
    }
