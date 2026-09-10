from __future__ import annotations

import json
import re
from typing import Any

SIGNAL_KEYWORDS: dict[str, list[str]] = {
    "prometheus": [r"prometheus", r":9090\b", r"/api/v1/query"],
    "jaeger": [r"jaeger", r":16686\b", r"/api/traces"],
    "logs": [r"kubectl\s+logs", r"docker\s+logs", r"\.log\b", r"deploys\.log"],
    "kubectl_status": [
        r"kubectl\s+(get|describe|top)\b",
        r"docker\s+(ps|stats|inspect)\b",
    ],
    "git_history": [r"git\s+(log|show|diff|blame)\b"],
    "rag_history": [r"search_past_incidents", r"历史工单", r"past_incidents"],
}


def classify_signal_types(tool_name: str, tool_input: dict[str, Any], observation: str) -> list[str]:
    text = f"{tool_name} {json.dumps(tool_input, ensure_ascii=False)} {observation}".lower()
    covered = []
    for sig, patterns in SIGNAL_KEYWORDS.items():
        if any(re.search(pat, text, re.IGNORECASE) for pat in patterns):
            covered.append(sig)
    return covered


def normalize_call(tool_name: str, tool_input: dict[str, Any]) -> str:
    if tool_name == "Bash":
        command = str(tool_input.get("command", "")).strip()
        command = re.sub(r"\s+", " ", command)
        return f"Bash:{command}"
    return f"{tool_name}:{json.dumps(tool_input, sort_keys=True, ensure_ascii=False)}"


def annotate_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_max_signals: dict[str, set[str]] = {}
    for step in steps:
        tool_name = step.get("tool_name", "")
        tool_input = step.get("tool_input", {}) or {}
        observation = step.get("observation", "") or ""

        covered = classify_signal_types(tool_name, tool_input, observation)
        step["signal_types_covered"] = covered

        key = normalize_call(tool_name, tool_input)
        prev_signals = seen_max_signals.get(key)
        if prev_signals is not None and set(covered) <= prev_signals:
            step["is_repeat_no_new_info"] = True
        else:
            step["is_repeat_no_new_info"] = False
        seen_max_signals[key] = prev_signals | set(covered) if prev_signals else set(covered)
    return steps
