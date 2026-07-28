from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from agent import config


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
