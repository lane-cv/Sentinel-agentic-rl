from __future__ import annotations

from typing import Optional

from reward.anti_hacking import find_fabricated_evidence
from reward.outcome_reward import STOP_LOSS_SUCCESS_REWARD, stop_loss_reward
from reward.trajectory import Trajectory

ROOT_CAUSE_FIELDS: tuple[str, ...] = ("suspect_service", "kind", "remediation_type")


def _safe_div(numerator: float, denominator: float) -> Optional[float]:
    if denominator <= 0:
        return None
    return round(numerator / denominator, 4)


def route_accuracy(trajectories: list[Trajectory]) -> Optional[float]:
    scored = [t for t in trajectories if t.ground_truth.get("route") is not None]
    correct = sum(1 for t in scored if t.route == t.ground_truth["route"])
    return _safe_div(correct, len(scored))


def root_cause_field_accuracy(trajectories: list[Trajectory]) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {}
    for field_name in ROOT_CAUSE_FIELDS:
        scored = [t for t in trajectories if t.ground_truth.get(field_name) is not None]
        correct = sum(1 for t in scored if t.final_diagnosis.get(field_name) == t.ground_truth[field_name])
        out[field_name] = _safe_div(correct, len(scored))
    applicable = [v for v in out.values() if v is not None]
    out["overall"] = round(sum(applicable) / len(applicable), 4) if applicable else None
    return out


def stop_loss_success_rate(trajectories: list[Trajectory]) -> Optional[float]:
    applicable = [t for t in trajectories if t.final_diagnosis.get("remediation_type") == "online_op"]
    successes = sum(1 for t in applicable if stop_loss_reward(t) == STOP_LOSS_SUCCESS_REWARD)
    return _safe_div(successes, len(applicable))


def avg_tool_call_rounds(trajectories: list[Trajectory]) -> Optional[float]:
    if not trajectories:
        return None
    return round(sum(len(t.steps) for t in trajectories) / len(trajectories), 3)


def termination_decision_accuracy(trajectories: list[Trajectory], tolerance: int = 0) -> Optional[float]:
    scored = [t for t in trajectories if t.ground_truth.get("expected_step_count") is not None]
    correct = sum(
        1 for t in scored if abs(len(t.steps) - t.ground_truth["expected_step_count"]) <= tolerance
    )
    return _safe_div(correct, len(scored))


def evidence_traceability_rate(trajectories: list[Trajectory]) -> Optional[float]:
    total = 0
    traceable = 0
    for t in trajectories:
        claims = t.final_diagnosis.get("evidence") or []
        if not claims:
            continue
        fabricated = set(find_fabricated_evidence(t))
        for claim in claims:
            total += 1
            if claim not in fabricated:
                traceable += 1
    return _safe_div(traceable, total)


DEFAULT_OOD_RATIO_KEYS: tuple[str, ...] = ("route_accuracy", "root_cause_field_accuracy_overall")


def ood_generalization_ratio(
    held_out_metrics: dict[str, object],
    ood_metrics: dict[str, object],
    keys: tuple[str, ...] = DEFAULT_OOD_RATIO_KEYS,
) -> Optional[float]:
    ratios: list[float] = []
    for key in keys:
        h = held_out_metrics.get(key)
        o = ood_metrics.get(key)
        if h is None or o is None or not isinstance(h, (int, float)) or h == 0:
            continue
        ratios.append(float(o) / float(h))
    if not ratios:
        return None
    return round(sum(ratios) / len(ratios), 4)


def compute_all(trajectories: list[Trajectory]) -> dict[str, object]:
    root_cause = root_cause_field_accuracy(trajectories)
    return {
        "n": len(trajectories),
        "route_accuracy": route_accuracy(trajectories),
        "root_cause_field_accuracy": root_cause,
        "root_cause_field_accuracy_overall": root_cause.get("overall"),
        "stop_loss_success_rate": stop_loss_success_rate(trajectories),
        "avg_tool_call_rounds": avg_tool_call_rounds(trajectories),
        "termination_decision_accuracy": termination_decision_accuracy(trajectories),
        "evidence_traceability_rate": evidence_traceability_rate(trajectories),
    }
