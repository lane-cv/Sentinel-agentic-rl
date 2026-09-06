from __future__ import annotations

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import Callable, Optional

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import common

logger = logging.getLogger(__name__)


EXTERNAL_DEP_PATTERNS = [
    r"外部网络",
    r"公网",
    r"第三方(?:服务|接口|API)",
    r"运营商",
    r"卫星链路",
    r"跨机房专线",
    r"互联网(?:出口|链路)",
]
NONDETERMINISTIC_TIMING_PATTERNS = [
    r"偶发",
    r"随机时间",
    r"不定期",
    r"时不时",
    r"看运气",
    r"时好时坏",
]
FIXTURE_HOOK_PATTERNS = [r"flagd", r"flag", r"故障注入", r"inject", r"commit", r"发版", r"部署", r"镜像"]


def is_likely_unreproducible(alert: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    service = common.resolve_service(alert)
    if not service or service.lower() == "unknown":
        reasons.append("缺失可解析的 service，没有可挂的注入钩子")
    elif service.lower() not in common.KNOWN_FIXTURE_SERVICES:
        reasons.append(f"service={service!r} 不在已知 fixture 服务集合内")

    text = common.alert_text(alert)
    has_hook = any(re.search(p, text, re.IGNORECASE) for p in FIXTURE_HOOK_PATTERNS)
    if not has_hook:
        if any(re.search(p, text) for p in EXTERNAL_DEP_PATTERNS):
            reasons.append("描述涉及外部网络依赖，且未见明确注入钩子（flagd flag / commit / 发版）")
        if any(re.search(p, text) for p in NONDETERMINISTIC_TIMING_PATTERNS):
            reasons.append("描述涉及非确定性时序/偶发现象，且未见明确注入钩子")

    return (len(reasons) > 0, reasons)


def filter_unreproducible(alerts: list[dict]) -> tuple[list[dict], list[tuple[dict, list[str]]]]:
    kept: list[dict] = []
    dropped: list[tuple[dict, list[str]]] = []
    for a in alerts:
        flagged, reasons = is_likely_unreproducible(a)
        if flagged:
            dropped.append((a, reasons))
        else:
            kept.append(a)
    return kept, dropped


def _load_real_fingerprint_fn() -> Optional[Callable[[dict], str]]:
    repo_root = Path(__file__).resolve().parents[2]
    aiops_agent_dir = repo_root / "Sentinel-agent"
    if not aiops_agent_dir.is_dir():
        logger.warning("找不到 Sentinel-agent 子模块目录 (%s)，回退到本地重实现的指纹函数", aiops_agent_dir)
        return None

    path_str = str(aiops_agent_dir)
    inserted = path_str not in sys.path
    if inserted:
        sys.path.insert(0, path_str)
    try:
        from agent.integrations.ingest import alert_fingerprint

        return alert_fingerprint
    except Exception as exc:
        logger.warning(
            "无法从 Sentinel-agent 子模块导入真实 alert_fingerprint()（%s: %s），"
            "回退到本地重实现（_fallback_alert_fingerprint，逻辑需与 ingest.py 保持字节级一致）",
            type(exc).__name__,
            exc,
        )
        if inserted:
            try:
                sys.path.remove(path_str)
            except ValueError:
                pass
        return None


def _fallback_alert_fingerprint(alert: dict) -> str:
    name = alert.get("alertname") or alert.get("name") or ""
    labels = alert.get("labels", {}) or {}
    service = alert.get("service") or labels.get("service") or labels.get("job") or ""
    severity = labels.get("severity", "")
    key = f"{name}|{service}|{severity}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


_fingerprint_fn: Optional[Callable[[dict], str]] = None
_fingerprint_source = "uninitialized"


def get_fingerprint_fn() -> Callable[[dict], str]:
    global _fingerprint_fn, _fingerprint_source
    if _fingerprint_fn is None:
        real = _load_real_fingerprint_fn()
        if real is not None:
            _fingerprint_fn = real
            _fingerprint_source = "Sentinel-agent.agent.integrations.ingest.alert_fingerprint (imported)"
        else:
            _fingerprint_fn = _fallback_alert_fingerprint
            _fingerprint_source = "data.clean.filter._fallback_alert_fingerprint (reimplemented)"
        logger.info("指纹函数来源：%s", _fingerprint_source)
    return _fingerprint_fn


_DEFAULT_DEDUP_TTL_SECONDS = 30 * 60


def _load_dedup_ttl_seconds() -> int:
    try:
        from agent import config

        return int(config.DEDUP_TTL_SECONDS)
    except Exception:
        return _DEFAULT_DEDUP_TTL_SECONDS


def _parse_starts_at_epoch(alert: dict) -> Optional[float]:
    raw = alert.get("startsAt")
    if not raw:
        return None
    try:
        import datetime

        dt = datetime.datetime.strptime(raw, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc
        )
        return dt.timestamp()
    except ValueError:
        return None


def filter_fingerprint_collisions(
    alerts: list[dict], fingerprint_fn: Optional[Callable[[dict], str]] = None
) -> tuple[list[dict], list[dict]]:
    fp_fn = fingerprint_fn or get_fingerprint_fn()
    ttl = _load_dedup_ttl_seconds()
    last_seen_at: dict[str, float] = {}
    kept: list[dict] = []
    dropped: list[dict] = []
    for a in alerts:
        fp = fp_fn(a)
        ts = _parse_starts_at_epoch(a)
        last = last_seen_at.get(fp)
        if last is not None and (ts is None or abs(ts - last) <= ttl):
            dropped.append(a)
        else:
            if ts is not None:
                last_seen_at[fp] = ts
            elif fp not in last_seen_at:
                last_seen_at[fp] = 0.0
            kept.append(a)
    return kept, dropped


def quality_filter_pipeline(alerts: list[dict]) -> dict:
    kept1, dropped_unrepro = filter_unreproducible(alerts)
    kept2, dropped_fp = filter_fingerprint_collisions(kept1)
    return {
        "kept": kept2,
        "dropped_unreproducible": dropped_unrepro,
        "dropped_fingerprint_collision": dropped_fp,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="过滤无法复现和指纹重复的告警")
    parser.add_argument("--in-dir", required=True, help="输入目录，读取其中的 seed_*.json")
    parser.add_argument("--out-dir", required=True, help="输出目录，写入幸存的 seed_*.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    get_fingerprint_fn()
    logger.info("fingerprint fn source in this environment: %s", _fingerprint_source)

    pairs = common.read_alerts_dir(args.in_dir)
    logger.info("读取 %d 条种子告警自 %s", len(pairs), args.in_dir)

    alerts = [a for _, a in pairs]
    result = quality_filter_pipeline(alerts)

    kept_ids = {id(a) for a in result["kept"]}
    kept_pairs = [(fname, a) for fname, a in pairs if id(a) in kept_ids]
    common.write_alerts_dir(kept_pairs, args.out_dir)

    logger.info(
        "质量过滤完成：保留 %d 条，砍掉不可复现 %d 条、指纹碰撞 %d 条 -> 写入 %s",
        len(kept_pairs),
        len(result["dropped_unreproducible"]),
        len(result["dropped_fingerprint_collision"]),
        args.out_dir,
    )


if __name__ == "__main__":
    main()
