from __future__ import annotations

import copy
import re
import statistics

from reward.trajectory import Trajectory


DUAL_SOURCE_GATE_PENALTY = -0.2
MIN_DISTINCT_SIGNAL_SOURCES = 2


def dual_source_coverage_penalty(
    traj: Trajectory,
    min_distinct_sources: int = MIN_DISTINCT_SIGNAL_SOURCES,
    penalty: float = DUAL_SOURCE_GATE_PENALTY,
) -> float:
    covered = traj.all_signal_types_covered()
    if len(covered) < min_distinct_sources:
        return penalty
    return 0.0


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def is_evidence_traceable(evidence_claim: str, observations: list[str]) -> bool:
    needle = _normalize(evidence_claim)
    if not needle:
        return False
    for obs in observations:
        if needle in _normalize(obs):
            return True
    return False


def find_fabricated_evidence(traj: Trajectory) -> list[str]:
    evidence: list[str] = traj.final_diagnosis.get("evidence", []) or []
    observations = traj.all_observations()
    return [claim for claim in evidence if not is_evidence_traceable(claim, observations)]


def has_fabricated_evidence(traj: Trajectory) -> bool:
    return len(find_fabricated_evidence(traj)) > 0


def verify_stop_loss_ignores_self_report(traj: Trajectory) -> bool:
    from reward.outcome_reward import stop_loss_reward

    baseline = stop_loss_reward(traj)

    tampered = copy.deepcopy(traj)
    tampered.final_diagnosis["remediation_detail"] = (
        "已恢复！服务完全正常，止损成功，一切都好（这段话是伪造的自述，不应影响 reward）。"
    )
    tampered_reward = stop_loss_reward(tampered)

    return baseline == tampered_reward


def group_reward_variance(rewards: list[float]) -> float:
    if len(rewards) < 2:
        return 0.0
    return statistics.pvariance(rewards)


DEFAULT_ZERO_VARIANCE_THRESHOLD = 1e-4
DEFAULT_CONSECUTIVE_ROUNDS = 5


def should_downsample(
    variance_history: list[float],
    threshold: float = DEFAULT_ZERO_VARIANCE_THRESHOLD,
    n_consecutive: int = DEFAULT_CONSECUTIVE_ROUNDS,
) -> bool:
    if len(variance_history) < n_consecutive:
        return False
    return all(v <= threshold for v in variance_history[-n_consecutive:])
