from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

Kind = Literal["dependency", "resource", "deploy_regression", "config"]
RemediationType = Literal["online_op", "code_fix", "info_only"]


class TimeWindow(BaseModel):
    from_: str = Field(alias="from")
    to: str

    model_config = {"populate_by_name": True}


class ExecutedAction(BaseModel):

    action: str
    target: str
    command: str
    ts: Optional[str] = None


class Diagnosis(BaseModel):

    summary: str
    kind: Kind
    suspect_service: str
    suspect_repo: Optional[str] = None
    suspect_commit_hint: Optional[str] = None
    suspect_file_hint: Optional[str] = None
    remediation_type: RemediationType
    remediation_detail: str
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[str] = Field(min_length=1)
    time_window: Optional[TimeWindow] = None
    also_code_fix: bool = False
    executed_actions: list[ExecutedAction] = Field(default_factory=list)

    @staticmethod
    def fallback(reason: str) -> "Diagnosis":
        return Diagnosis(
            summary=f"诊断未能产出可信结果: {reason}",
            kind="config",
            suspect_service="unknown",
            remediation_type="info_only",
            remediation_detail="自动诊断失败/降级，需人工介入排查。",
            confidence=0.0,
            evidence=[f"degraded: {reason}"],
        )


DIAGNOSIS_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "summary",
        "kind",
        "suspect_service",
        "remediation_type",
        "remediation_detail",
        "confidence",
        "evidence",
    ],
    "properties": {
        "summary": {"type": "string"},
        "kind": {
            "type": "string",
            "enum": ["dependency", "resource", "deploy_regression", "config"],
        },
        "suspect_service": {"type": "string"},
        "suspect_repo": {"type": ["string", "null"]},
        "suspect_commit_hint": {"type": ["string", "null"]},
        "suspect_file_hint": {"type": ["string", "null"]},
        "remediation_type": {
            "type": "string",
            "enum": ["online_op", "code_fix", "info_only"],
        },
        "remediation_detail": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
        },
        "time_window": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "from": {"type": "string"},
                "to": {"type": "string"},
            },
        },
        "also_code_fix": {"type": "boolean"},
    },
}
