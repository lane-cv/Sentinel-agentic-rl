from __future__ import annotations

import json
from typing import Any


def stringify_tool_result_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


def pack_mcp_result(mcp_result: dict[str, Any] | None) -> str:
    return stringify_tool_result_content((mcp_result or {}).get("content", []))


def pack_bash_exec_result(stdout: str, stderr: str = "", exit_code: int = 0) -> str:
    parts = [p for p in (stdout.rstrip("\n") if stdout else "", stderr.rstrip("\n") if stderr else "") if p]
    if not parts:
        return f"(no output; exit_code={exit_code})"
    text = "\n".join(parts)
    if exit_code != 0:
        text += f"\n(exit_code={exit_code})"
    return text


def pack_hook_denied_result(reason: str) -> str:
    return reason
