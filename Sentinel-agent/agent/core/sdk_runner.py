from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Optional

from claude_agent_sdk import (
    ClaudeAgentOptions,
    HookMatcher,
    ResultMessage,
    query,
)

from agent import config


def _accumulate_usage(dst: dict[str, int], usage: Optional[dict]) -> None:
    if not usage:
        return
    for k in (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ):
        v = usage.get(k)
        if isinstance(v, (int, float)):
            dst[k] = dst.get(k, 0) + int(v)


def _llm_env_overrides() -> dict[str, str]:
    return {
        "ANTHROPIC_BASE_URL": config.LLM_BASE_URL,
        "ANTHROPIC_API_KEY": config.LLM_API_KEY,
        "ANTHROPIC_AUTH_TOKEN": config.LLM_AUTH_TOKEN,
        "ANTHROPIC_MODEL": config.MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": config.MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": config.MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": config.MODEL,
        "ANTHROPIC_SMALL_FAST_MODEL": config.MODEL,
        "ANTHROPIC_CUSTOM_HEADERS": "",
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


async def run_sdk(
    prompt: str,
    *,
    cwd: Optional[str],
    allowed_tools: list[str],
    append_prompt: str,
    hook,
    max_turns: int,
    timeout_s: int,
    json_schema: Optional[dict] = None,
    max_budget_usd: Optional[float] = None,
    mcp_servers: Optional[dict] = None,
    skills: Optional[Any] = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"

    setting_sources = ["project"] if skills is not None else []

    options_kwargs: dict[str, Any] = dict(
        cwd=cwd,
        model=config.MODEL,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": append_prompt,
        },
        allowed_tools=allowed_tools,
        permission_mode="acceptEdits",
        disallowed_tools=["Bash(rm -rf *)", "Bash(git push --force *)"],
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[hook])]},
        setting_sources=setting_sources,
        max_turns=max_turns,
        env=_llm_env_overrides(),
    )
    if json_schema is not None:
        options_kwargs["output_format"] = {"type": "json_schema", "schema": json_schema}
    if max_budget_usd is not None:
        options_kwargs["max_budget_usd"] = max_budget_usd
    if mcp_servers is not None:
        options_kwargs["mcp_servers"] = mcp_servers
    if skills is not None:
        options_kwargs["skills"] = skills

    opts = ClaudeAgentOptions(**options_kwargs)

    result_text: Optional[str] = None
    structured: Any = None
    usage: dict[str, int] = {}
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    degraded: Optional[str] = None

    try:
        async with asyncio.timeout(timeout_s):
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, ResultMessage):
                    _accumulate_usage(usage, msg.usage)
                    if msg.total_cost_usd is not None:
                        cost_usd = msg.total_cost_usd
                    num_turns = msg.num_turns
                    if msg.subtype == "success":
                        result_text = msg.result
                        structured = msg.structured_output
                    else:
                        degraded = msg.subtype
                        result_text = msg.result
                        structured = msg.structured_output
    except asyncio.TimeoutError:
        degraded = "timeout"
    except Exception as exc:
        msg = str(exc)
        if "maximum number of turns" in msg:
            degraded = "error_max_turns"
        elif "budget" in msg.lower():
            degraded = "error_max_budget"
        else:
            degraded = f"sdk_error: {msg[:200]}"

    return {
        "raw": result_text,
        "structured": structured,
        "degraded": degraded,
        "usage": usage,
        "cost_usd": cost_usd,
        "num_turns": num_turns,
        "latency_s": time.perf_counter() - t0,
    }
