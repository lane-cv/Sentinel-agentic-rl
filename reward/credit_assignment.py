from __future__ import annotations

from reward.trajectory import ALL_SIGNAL_TYPES, Trajectory

SIGNAL_WEIGHTS_BY_KIND: dict[str, dict[str, float]] = {
    "dependency": {
        "jaeger": 1.5,
        "logs": 1.2,
        "prometheus": 1.0,
        "kubectl_status": 0.8,
        "rag_history": 0.8,
        "git_history": 0.6,
    },
    "resource": {
        "kubectl_status": 1.5,
        "prometheus": 1.3,
        "logs": 0.8,
        "rag_history": 0.8,
        "jaeger": 0.6,
        "git_history": 0.5,
    },
    "deploy_regression": {
        "git_history": 1.5,
        "logs": 1.2,
        "kubectl_status": 1.0,
        "prometheus": 0.8,
        "jaeger": 0.8,
        "rag_history": 0.8,
    },
    "config": {
        "logs": 1.5,
        "rag_history": 1.2,
        "kubectl_status": 0.8,
        "prometheus": 0.7,
        "jaeger": 0.7,
        "git_history": 0.6,
    },
}

_DEFAULT_WEIGHTS: dict[str, float] = {t: 1.0 for t in ALL_SIGNAL_TYPES}


def _weights_for_kind(kind: str | None) -> dict[str, float]:
    if kind in SIGNAL_WEIGHTS_BY_KIND:
        return SIGNAL_WEIGHTS_BY_KIND[kind]
    return _DEFAULT_WEIGHTS


def cumulative_signal_types(traj: Trajectory, step_index: int) -> set[str]:
    covered: set[str] = set()
    for step in traj.steps[: step_index + 1]:
        covered.update(step.signal_types_covered)
    return covered


def _newly_covered_signal_types(traj: Trajectory, step_index: int) -> set[str]:
    now = cumulative_signal_types(traj, step_index)
    prior = cumulative_signal_types(traj, step_index - 1) if step_index > 0 else set()
    return now - prior


def evidence_value_reward(traj: Trajectory, step_index: int) -> float:
    kind = traj.final_diagnosis.get("kind")
    weights = _weights_for_kind(kind)
    new_types = _newly_covered_signal_types(traj, step_index)
    return sum((weights.get(t, 0.0) for t in new_types), 0.0)


def compute_credit_assignment_rewards(traj: Trajectory) -> list[float]:
    return [evidence_value_reward(traj, i) for i in range(len(traj.steps))]
