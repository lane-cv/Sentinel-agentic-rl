from __future__ import annotations

import json
import re
from pathlib import Path

KIND_VALUES = ("dependency", "resource", "deploy_regression", "config")

KNOWN_FIXTURE_SERVICES = {
    "recommendation",
    "ad",
    "adservice",
    "frontend",
    "cart",
    "checkout",
    "currency",
    "payment",
    "shipping",
    "quote",
    "email",
    "product-catalog",
    "productcatalog",
    "kafka",
}


def resolve_service(alert: dict) -> str:
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    return service.strip()


def alert_text(alert: dict) -> str:
    ann = alert.get("annotations", {}) or {}
    summary = (ann.get("summary") or "").strip()
    description = (ann.get("description") or "").strip()
    return f"{summary} {description}".strip()


_HYBRID_PATTERNS = [
    r"混合根因",
    r"also_code_fix",
    r"既要.*也要",
    r"先.*(?:止血|止损).*(?:再|同时).*(?:根治|改代码|code_fix)",
    r"同时(?:触发|需要).*(?:代码修复|code_fix)",
]

_LOW_CONFIDENCE_PATTERNS = [
    r"证据不足",
    r"没有明确",
    r"没有指名",
    r"不清楚具体",
    r"无法定位",
    r"感觉有点",
    r"大约.*左右",
    r"间歇性.*但",
]

_KIND_KEYWORD_MAP: dict[str, list[str]] = {
    "resource": [r"cpu", r"内存", r"memory", r"oom", r"扩容", r"资源(?:不足|打满)"],
    "deploy_regression": [r"发版", r"上线", r"部署", r"commit", r"镜像", r"回归", r"deploy"],
    "dependency": [r"依赖", r"上游", r"下游", r"调用失败", r"5xx", r"错误率", r"flagd", r"队列", r"积压", r"kafka"],
    "config": [r"配置(?:开关|中心|变更)", r"flag(?:d)?\s*(?:开关|key)", r"功能开关"],
}


def infer_hybrid(alert: dict) -> bool:
    explicit = alert.get("hybrid")
    if isinstance(explicit, bool):
        return explicit
    text = f"{alert_text(alert)} {alert.get('scenario_hint', '')}"
    return any(re.search(p, text, re.IGNORECASE) for p in _HYBRID_PATTERNS)


def infer_low_confidence(alert: dict) -> bool:
    explicit = alert.get("low_confidence")
    if isinstance(explicit, bool):
        return explicit
    service = resolve_service(alert)
    if not service or service.lower() == "unknown":
        return True
    text = alert_text(alert)
    return any(re.search(p, text) for p in _LOW_CONFIDENCE_PATTERNS)


def infer_kind(alert: dict) -> str:
    explicit = alert.get("kind")
    if explicit in KIND_VALUES:
        return explicit

    haystack = " ".join(
        [
            alert_text(alert),
            str(alert.get("alertname", "")),
            str(alert.get("scenario_hint", "")),
        ]
    ).lower()
    scores = {k: sum(1 for p in pats if re.search(p, haystack)) for k, pats in _KIND_KEYWORD_MAP.items()}
    best_kind, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score == 0:
        return "dependency"
    return best_kind


def stratum_key(alert: dict) -> str:
    if infer_low_confidence(alert):
        return "low_confidence"
    if infer_hybrid(alert):
        return "hybrid"
    return f"kind:{infer_kind(alert)}"


def read_alerts_dir(in_dir: str | Path) -> list[tuple[str, dict]]:
    in_dir = Path(in_dir)
    return [
        (p.name, json.loads(p.read_text(encoding="utf-8")))
        for p in sorted(in_dir.glob("seed_*.json"))
    ]


def write_alerts_dir(pairs: list[tuple[str, dict]], out_dir: str | Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {fname for fname, _ in pairs}
    for stale in out_dir.glob("seed_*.json"):
        if stale.name not in keep_names:
            stale.unlink()
    for fname, alert in pairs:
        with (out_dir / fname).open("w", encoding="utf-8") as f:
            json.dump(alert, f, ensure_ascii=False, indent=2)
            f.write("\n")
