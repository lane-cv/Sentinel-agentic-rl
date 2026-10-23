from __future__ import annotations

import hashlib
import json
import random
from abc import ABC, abstractmethod
from typing import Any, Optional

from reward.trajectory import (
    ALL_REMEDIATION_TYPES,
    ALL_ROUTES,
    ALL_SIGNAL_TYPES,
    Step,
    Trajectory,
)

SERVICE_POOL: tuple[str, ...] = (
    "product-catalog",
    "recommendation",
    "checkout",
    "ad",
    "currency",
    "payment",
    "cart",
    "frontend",
    "kafka",
    "shipping",
)


def _stable_rng(alert: dict[str, Any], salt: str) -> random.Random:
    key = json.dumps(alert, sort_keys=True, default=str) + "|" + salt
    seed = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % (2**32)
    return random.Random(seed)


class DiagnosisEndpoint(ABC):

    name: str = "base"

    @abstractmethod
    def diagnose(self, alert: dict[str, Any], ground_truth: Optional[dict[str, Any]] = None) -> Trajectory:
        raise NotImplementedError


class ScriptedMockEndpoint(DiagnosisEndpoint):

    def __init__(
        self,
        name: str,
        wrong_field_prob: float,
        n_signal_types: int,
        fabricate_evidence_prob: float,
        padding_steps: int,
    ) -> None:
        self.name = name
        self.wrong_field_prob = wrong_field_prob
        self.n_signal_types = n_signal_types
        self.fabricate_evidence_prob = fabricate_evidence_prob
        self.padding_steps = padding_steps


    def _maybe_corrupt(self, rng: random.Random, truth_value: Any, pool: tuple[str, ...]) -> str:
        if truth_value is None:
            return rng.choice(pool)
        if rng.random() < self.wrong_field_prob:
            candidates = [v for v in pool if v != truth_value]
            return rng.choice(candidates) if candidates else truth_value
        return truth_value

    def diagnose(self, alert: dict[str, Any], ground_truth: Optional[dict[str, Any]] = None) -> Trajectory:
        if ground_truth is None:
            raise ValueError(
                f"{self.name}: scripted mock endpoints require a labeled `ground_truth` "
                "(they can only be used against a labeled eval set, never in production)."
            )
        rng = _stable_rng(alert, self.name)

        service = self._maybe_corrupt(rng, ground_truth.get("suspect_service"), SERVICE_POOL)
        remediation_type = self._maybe_corrupt(rng, ground_truth.get("remediation_type"), ALL_REMEDIATION_TYPES)
        route = self._maybe_corrupt(rng, ground_truth.get("route"), ALL_ROUTES)
        kind = ground_truth.get("kind") or "resource"

        signal_pool = list(ALL_SIGNAL_TYPES)
        rng.shuffle(signal_pool)
        chosen_types = signal_pool[: max(1, min(self.n_signal_types, len(signal_pool)))]

        steps: list[Step] = []
        observations: list[str] = []
        service_label = alert.get("service") or service
        for i, sig in enumerate(chosen_types):
            obs = f"[{self.name}] mock probe #{i} observed {sig} signal for {service_label}"
            observations.append(obs)
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": f"mock-probe --signal {sig}"},
                    hook_decision="passthrough",
                    observation=obs,
                    signal_types_covered=[sig],
                    is_repeat_no_new_info=False,
                )
            )
        for _ in range(self.padding_steps):
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": "mock-probe --signal " + (chosen_types[0] if chosen_types else "kubectl_status")},
                    hook_decision="passthrough",
                    observation=f"[{self.name}] repeat probe, no new information",
                    signal_types_covered=[],
                    is_repeat_no_new_info=True,
                )
            )

        post_action_health = None
        if remediation_type == "online_op":
            restart_obs = f"[{self.name}] {service} restarted, post-action health check pending"
            steps.append(
                Step(
                    tool_name="Bash",
                    tool_input={"command": f"docker restart {service}"},
                    hook_decision="allow",
                    hook_action="restart_instance",
                    hook_target=service,
                    observation=restart_obs,
                )
            )
            observations.append(restart_obs)
            healthy = rng.random() >= self.wrong_field_prob
            post_action_health = {
                "healthy_at_60s": healthy,
                "new_alert_triggered": (not healthy) and rng.random() < 0.3,
            }

        evidence = list(observations[:2])
        if rng.random() < self.fabricate_evidence_prob:
            evidence.append(
                f"[{self.name}] fabricated claim never actually observed in any tool call"
            )

        final_diagnosis: dict[str, Any] = {
            "summary": f"[{self.name} mock] scripted diagnosis for {alert.get('alertname', 'unknown-alert')}",
            "kind": kind,
            "suspect_service": service,
            "suspect_repo": ground_truth.get("suspect_repo"),
            "suspect_commit_hint": ground_truth.get("suspect_commit_hint"),
            "suspect_file_hint": ground_truth.get("suspect_file_hint"),
            "remediation_type": remediation_type,
            "remediation_detail": f"[{self.name} mock] scripted remediation_detail text "
            "(never read by any reward function — see reward/outcome_reward.py's "
            "anti-self-report guard).",
            "confidence": round(max(0.0, 1.0 - self.wrong_field_prob), 2),
            "evidence": evidence,
            "time_window": None,
            "also_code_fix": False,
            "executed_actions": [],
        }

        return Trajectory(
            alert=alert,
            steps=steps,
            final_diagnosis=final_diagnosis,
            ground_truth=dict(ground_truth),
            route=route,
            post_action_health=post_action_health,
        )


class PerfectDiagnosisEndpoint(ScriptedMockEndpoint):

    def __init__(self) -> None:
        super().__init__(
            name="perfect",
            wrong_field_prob=0.0,
            n_signal_types=4,
            fabricate_evidence_prob=0.0,
            padding_steps=0,
        )


class ZeroShotLikeEndpoint(ScriptedMockEndpoint):

    def __init__(self) -> None:
        super().__init__(
            name="zero_shot_like",
            wrong_field_prob=0.75,
            n_signal_types=1,
            fabricate_evidence_prob=0.55,
            padding_steps=6,
        )


class PartiallyGoodEndpoint(ScriptedMockEndpoint):

    def __init__(self) -> None:
        super().__init__(
            name="sft_like",
            wrong_field_prob=0.35,
            n_signal_types=2,
            fabricate_evidence_prob=0.15,
            padding_steps=3,
        )


class SFTGRPOEndpoint(ScriptedMockEndpoint):

    def __init__(self) -> None:
        super().__init__(
            name="sft_grpo_like",
            wrong_field_prob=0.15,
            n_signal_types=3,
            fabricate_evidence_prob=0.05,
            padding_steps=1,
        )


class OpusReferencePlaceholderEndpoint(ScriptedMockEndpoint):

    def __init__(self) -> None:
        super().__init__(
            name="opus_reference_placeholder",
            wrong_field_prob=0.08,
            n_signal_types=4,
            fabricate_evidence_prob=0.02,
            padding_steps=0,
        )


ENDPOINT_REGISTRY: dict[str, type] = {
    "perfect": PerfectDiagnosisEndpoint,
    "zero_shot_like": ZeroShotLikeEndpoint,
    "sft_like": PartiallyGoodEndpoint,
    "sft_grpo_like": SFTGRPOEndpoint,
    "opus_reference_placeholder": OpusReferencePlaceholderEndpoint,
}


def build_endpoint(key: str) -> DiagnosisEndpoint:
    if key not in ENDPOINT_REGISTRY:
        raise KeyError(f"unknown endpoint {key!r}; choices: {sorted(ENDPOINT_REGISTRY)}")
    return ENDPOINT_REGISTRY[key]()
