from __future__ import annotations

import json
from typing import Any, Optional

from pydantic import ValidationError

from agent import config
from agent.agents import prompts
from agent.core import remediation
from agent.core.hooks import guard_diagnose_ops
from agent.core.schema import DIAGNOSIS_JSON_SCHEMA, Diagnosis, ExecutedAction
from agent.core.sdk_runner import run_sdk
from agent.integrations import rag_tool


def _coerce(structured: Any, raw: Optional[str]) -> Optional[Diagnosis]:
    candidate = structured
    if candidate is None and raw:
        text = raw.strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{") : text.rfind("}") + 1]
        try:
            candidate = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return None
    if candidate is None:
        return None
    try:
        return Diagnosis.model_validate(candidate)
    except ValidationError:
        return None


async def diagnose(alert: dict) -> dict[str, Any]:
    prompt = prompts.diagnose_prompt(alert)
    last_errors = ""
    meta: dict[str, Any] = {"attempts": 0, "usage": {}, "cost_usd": None, "degraded": None, "latency_s": 0.0}

    remediation.reset_ledger()

    for attempt in range(config.SCHEMA_RETRY_MAX + 1):
        meta["attempts"] = attempt + 1

        allowed = ["Bash", "Read", "Grep", "Glob"]
        mcp_servers = None
        if config.RAG_ENABLED:
            allowed.append(rag_tool.TOOL_NAME)
            mcp_servers = {"aiops_rag": rag_tool.server}

        out = await run_sdk(
            prompt,
            cwd=str(config.REPO_ROOT),
            allowed_tools=allowed,
            append_prompt=prompts.diagnose_append(),
            hook=guard_diagnose_ops,
            max_turns=config.DIAGNOSE_MAX_TURNS,
            timeout_s=config.DIAGNOSE_TIMEOUT_S,
            json_schema=DIAGNOSIS_JSON_SCHEMA,
            mcp_servers=mcp_servers,
            skills="all",
        )

        for k, v in out["usage"].items():
            meta["usage"][k] = meta["usage"].get(k, 0) + v
        if out["cost_usd"] is not None:
            meta["cost_usd"] = (meta["cost_usd"] or 0.0) + out["cost_usd"]
        meta["latency_s"] += out.get("latency_s", 0.0)
        meta["degraded"] = out["degraded"]

        diag = _coerce(out["structured"], out["raw"])
        if diag is not None:
            _attach_executed_actions(diag)
            return {"diagnosis": diag, "meta": meta}

        last_errors = _describe_failure(out)
        prompt = prompts.diagnose_retry_prompt(last_errors)

    meta["degraded"] = meta["degraded"] or "schema_validation_failed"
    fb = Diagnosis.fallback(f"schema 校验重试 {config.SCHEMA_RETRY_MAX} 次仍失败")
    _attach_executed_actions(fb)
    return {"diagnosis": fb, "meta": meta}


def _attach_executed_actions(diag: Diagnosis) -> None:
    diag.executed_actions = [ExecutedAction(**e) for e in remediation.get_ledger()]


def _describe_failure(out: dict[str, Any]) -> str:
    if out["degraded"]:
        return f"上次运行降级({out['degraded']})，未产出合法 JSON。"
    raw = (out["raw"] or "")[:500]
    return f"上次输出无法解析为合法 Diagnosis JSON。原始片段: {raw!r}"
