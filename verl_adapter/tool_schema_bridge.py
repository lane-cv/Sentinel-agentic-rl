from __future__ import annotations

from typing import Any

AIOPS_AGENT_BUILTIN_TOOLS: tuple[str, ...] = ("Bash", "Read", "Grep", "Glob")

RAG_TOOL_NAME = "mcp__aiops_rag__search_past_incidents"


def _tool_schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
            "strict": False,
        },
    }


def bash_tool_schema() -> dict[str, Any]:
    return _tool_schema(
        "Bash",
        "Execute a shell command. Read-only observability commands (kubectl get/"
        "describe/top/logs, docker ps/stats/inspect/logs, curl GET, git log/show/"
        "diff/blame, rg/grep/cat/ls) are unrestricted. Low-risk remediation "
        "commands are auto-executed ONLY if they match the Sentinel-agent "
        "remediation.py whitelist AND target an authorized instance — every "
        "other command (including a whitelisted action against an "
        "unauthorized target) is denied by the real PreToolUse hook before it "
        "ever runs.",
        {"command": {"type": "string", "description": "The shell command to run."}},
        ["command"],
    )


def read_tool_schema() -> dict[str, Any]:
    return _tool_schema(
        "Read",
        "Read a file from the local filesystem (read-only).",
        {"file_path": {"type": "string", "description": "Absolute path to the file to read."}},
        ["file_path"],
    )


def grep_tool_schema() -> dict[str, Any]:
    return _tool_schema(
        "Grep",
        "Search file contents with a regex pattern (read-only).",
        {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "File or directory to search (optional)."},
        },
        ["pattern"],
    )


def glob_tool_schema() -> dict[str, Any]:
    return _tool_schema(
        "Glob",
        "Find files matching a glob pattern (read-only).",
        {"pattern": {"type": "string", "description": "Glob pattern, e.g. '**/*.py'."}},
        ["pattern"],
    )


def search_past_incidents_tool_schema() -> dict[str, Any]:
    return _tool_schema(
        RAG_TOOL_NAME,
        "语义检索历史相似的诊断工单（根因/处置经验）。当你想参考过去是否处理过类似故障、"
        "或想借鉴历史根因与处置方式时调用。传入描述当前故障的自然语言查询（如告警症状、"
        "可疑服务、现象），返回最相似的若干历史工单摘要。",
        {
            "query": {"type": "string", "description": "描述当前故障的自然语言查询。"},
            "top_k": {"type": "integer", "description": "返回的历史工单条数。"},
        },
        ["query"],
    )


_SCHEMA_BUILDERS = {
    "Bash": bash_tool_schema,
    "Read": read_tool_schema,
    "Grep": grep_tool_schema,
    "Glob": glob_tool_schema,
    RAG_TOOL_NAME: search_past_incidents_tool_schema,
}


def build_verl_tool_schemas(include_rag: bool = True) -> list[dict[str, Any]]:
    names = list(AIOPS_AGENT_BUILTIN_TOOLS) + ([RAG_TOOL_NAME] if include_rag else [])
    return [_SCHEMA_BUILDERS[n]() for n in names]


def build_tool_config_entries(
    include_rag: bool = True,
    class_name: str = "verl_adapter.rollout_worker.RealHookRoutedTool",
) -> list[dict[str, Any]]:
    return [
        {
            "class_name": class_name,
            "config": {"type": "native", "tool_name": schema["function"]["name"]},
            "tool_schema": schema,
        }
        for schema in build_verl_tool_schemas(include_rag=include_rag)
    ]
