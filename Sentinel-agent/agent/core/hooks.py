from __future__ import annotations

import re
from typing import Any

from agent.core import remediation

_READONLY_DENY = [
    r"\bkubectl\s+(apply|delete|scale|edit|patch|create|replace|rollout\s+undo|rollout\s+restart|cordon|drain|set|exec|cp|attach|port-forward)\b",
    r"\bhelm\s+(install|upgrade|uninstall|rollback|delete)\b",
    r"\bdocker\s+(rm|stop|kill|rmi|exec|run)\b",
    r"\bgit\s+(push|commit|merge|reset|rebase|checkout\s+-b|tag|clone)\b",
    r"\bgit\s+branch\s+(-[dDmM]\b|(?!-)\S)",
    r"\brm\s+-rf\b",
    r">\s*(?!/dev/(?:null|stdout|stderr)\b)/",
    r"\bredis-cli\b",
    r"\b(mysql|psql|mongo|mongosh)\b",
    r"\bflushall\b|\bflushdb\b|\bdrop\s+(table|database)\b|\bdelete\s+from\b|\btruncate\b",
]

_WRITE_FALLBACK_DENY = [
    r"\bdocker\s+(restart|update|pause|unpause)\b",
    r"\bdocker(?:\s+compose|-compose)\b(?![.*])",
]

_ONLINE_OP_DENY = [
    r"\bkubectl\s+(apply|delete|scale|edit|patch|rollout\s+undo|rollout\s+restart|cordon|drain|set|exec|cp|attach|port-forward)\b",
    r"\bhelm\s+(install|upgrade|uninstall|rollback|delete)\b",
    r"\bdocker\s+(rm|stop|restart|kill|rmi|update)\b",
    r"\bdocker(?:\s+compose|-compose)\s+(up|down|restart|stop)\b",
    r"\brm\s+-rf\b",
    r"\bgit\s+push\s+.*--force\b|\bgit\s+push\s+.*\bmaster\b|\bgit\s+push\s+.*\bmain\b",
]


def _extract_command(input_data: dict[str, Any]) -> str:
    if input_data.get("tool_name") != "Bash":
        return ""
    return str(input_data.get("tool_input", {}).get("command", ""))


def _deny(input_data: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": input_data.get("hook_event_name", "PreToolUse"),
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }


def _allow(input_data: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": input_data.get("hook_event_name", "PreToolUse"),
            "permissionDecision": "allow",
            "permissionDecisionReason": reason,
        }
    }


def _match(command: str, patterns: list[str]) -> str | None:
    for pat in patterns:
        if re.search(pat, command):
            return pat
    return None


async def guard_diagnose_ops(input_data, tool_use_id, context):
    command = _extract_command(input_data)
    if not command:
        return {}

    verdict = remediation.classify_command(command)
    if verdict["decision"] == "allow":
        remediation.record_execution(verdict["action"], verdict["target"], command.strip())
        return _allow(
            input_data,
            f"低风险处置已授权自动执行：{verdict['action']} → {verdict['target']}。",
        )
    if verdict["decision"] == "deny":
        return _deny(input_data, verdict["reason"])

    hit = _match(command, _READONLY_DENY)
    if hit:
        return _deny(
            input_data,
            f"诊断处置 Agent 禁止此写/变更命令（命中: {hit}）——高风险操作只发飞书卡片交人工。",
        )
    hit = _match(command, _WRITE_FALLBACK_DENY)
    if hit:
        return _deny(
            input_data,
            f"变更类命令不在低风险白名单内（命中: {hit}），拒绝执行。可自动执行的操作见白名单。",
        )
    return {}


async def deny_online_ops(input_data, tool_use_id, context):
    command = _extract_command(input_data)
    if not command:
        return {}
    hit = _match(command, _ONLINE_OP_DENY)
    if hit:
        return _deny(
            input_data,
            f"修复 Agent 禁止线上变更/强推/推 master（命中: {hit}）。只能改代码 + 推特性分支 + 提 PR。",
        )
    return {}
