#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "Sentinel-agent"


def _load_search_fn() -> Callable[[dict], Any]:
    if AIOPS_AGENT_DIR.is_dir():
        p = str(AIOPS_AGENT_DIR)
        if p not in sys.path:
            sys.path.insert(0, p)
    from agent.integrations import rag_tool

    fn = rag_tool.search_past_incidents
    if callable(fn):
        return fn
    handler = getattr(fn, "handler", None)
    if callable(handler):
        return handler
    raise RuntimeError(
        "rag_bridge: rag_tool.search_past_incidents is neither callable nor exposes a callable .handler "
        f"(type={type(fn).__name__}) — claude_agent_sdk 的 tool 装饰器行为可能变了，请核对 SDK 版本"
    )


async def _main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} '<json args dict, e.g. {{\"query\": \"...\"}}>'", file=sys.stderr)
        return 2
    try:
        args = json.loads(argv[1])
        if not isinstance(args, dict):
            raise ValueError(f"args must be a JSON object, got {type(args).__name__}")
    except (json.JSONDecodeError, ValueError) as exc:
        print(f"rag_bridge: bad args JSON: {exc}", file=sys.stderr)
        return 2

    search_fn = _load_search_fn()
    try:
        result = await search_fn(args)
    except Exception as exc:
        print(f"rag_bridge: search_past_incidents failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(_main(sys.argv)))
