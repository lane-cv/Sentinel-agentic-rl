from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
AIOPS_AGENT_DIR = REPO_ROOT / "Sentinel-agent"
COLD_START_DIR = REPO_ROOT / "data" / "cold_start"


def ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def ensure_cold_start_on_path() -> None:
    p = str(COLD_START_DIR)
    if COLD_START_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def load_real_hook_modules() -> tuple[Any, Any]:
    ensure_aiops_agent_on_path()
    from agent.core import remediation
    from agent.core.hooks import guard_diagnose_ops

    return remediation, guard_diagnose_ops


def load_signals_module() -> Any:
    ensure_cold_start_on_path()
    import signals

    return signals
