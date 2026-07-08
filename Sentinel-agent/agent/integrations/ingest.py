from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from agent import config

_SEEN: dict[str, float] = {}


def load_alert(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def alert_fingerprint(alert: dict) -> str:
    name = alert.get("alertname") or alert.get("name") or ""
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    severity = labels.get("severity", "")
    key = f"{name}|{service}|{severity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def seen_recently(fingerprint: str) -> bool:
    now = time.time()
    for fp in [k for k, ts in _SEEN.items() if now - ts > config.DEDUP_TTL_SECONDS]:
        _SEEN.pop(fp, None)
    last = _SEEN.get(fingerprint)
    if last is not None and now - last <= config.DEDUP_TTL_SECONDS:
        return True
    _SEEN[fingerprint] = now
    return False
