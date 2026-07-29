from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from agent import config
from agent.agents import prompts
from agent.core.hooks import deny_online_ops
from agent.core.schema import FIX_JSON_SCHEMA, FixResult
from agent.core.sdk_runner import run_sdk


def clone_service_repo(service: str, branch: Optional[str] = None) -> Path:
    config.validate_github_repository()
    config.WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
    dest = config.WORKSPACE_DIR / config.GITHUB_REPO
    if dest.exists():
        shutil.rmtree(dest)

    if config.GITHUB_TOKEN:
        auth = f"{config.GITHUB_TOKEN}@github.com"
    else:
        auth = "github.com"
    clone_url = f"https://{auth}/{config.GITHUB_FORK_OWNER}/{config.GITHUB_REPO}.git"

    cmd = ["git", "clone", "--depth", "50"]
    if branch:
        cmd += ["--branch", branch]
    cmd += [clone_url, str(dest)]
    subprocess_run(cmd)
    return dest


def subprocess_run(cmd: list[str]) -> None:
    import subprocess

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_fix(out: dict[str, Any]) -> FixResult:
    candidate = out.get("structured")
    if candidate is None and out.get("raw"):
        text = out["raw"].strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{") : text.rfind("}") + 1]
        try:
            candidate = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            candidate = None

    if candidate is not None:
        try:
            fr = FixResult.model_validate(candidate)
            if out.get("degraded") and not fr.degraded:
                fr.degraded = out["degraded"]
            return fr
        except ValidationError:
            pass

    return FixResult(
        verified=False,
        degraded=out.get("degraded") or "unparseable_fix_result",
        verify_log=(out.get("raw") or "")[:500],
    )


async def code_fix(rc: dict, cwd: Optional[str] = None, branch: Optional[str] = None) -> dict[str, Any]:
    try:
        if cwd is None:
            repo = clone_service_repo(rc.get("suspect_service", config.GITHUB_REPO), branch=branch)
            cwd = str(repo)
        else:
            config.validate_github_repository()
    except ValueError as exc:
        fix = FixResult(verified=False, degraded="github_config_error", verify_log=str(exc))
        meta = {"usage": {}, "cost_usd": None, "num_turns": 0, "degraded": fix.degraded, "latency_s": 0.0}
        return {"fix": fix, "meta": meta}

    out = await run_sdk(
        prompts.fix_prompt(rc, branch=branch),
        cwd=cwd,
        allowed_tools=["Bash", "Read", "Write", "Edit", "Grep", "Glob"],
        append_prompt=prompts.fix_append(),
        hook=deny_online_ops,
        max_turns=config.FIX_MAX_TURNS,
        timeout_s=config.FIX_TIMEOUT_S,
        json_schema=FIX_JSON_SCHEMA,
    )
    fix = parse_fix(out)
    meta = {
        "usage": out["usage"],
        "cost_usd": out["cost_usd"],
        "num_turns": out["num_turns"],
        "degraded": out["degraded"],
        "latency_s": out.get("latency_s", 0.0),
    }
    return {"fix": fix, "meta": meta}
