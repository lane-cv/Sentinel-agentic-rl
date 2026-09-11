from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

_FLAG_RE = re.compile(r"\bflags?=([\w+]+)")


def parse_flags(scenario_hint: str) -> list[str]:
    m = _FLAG_RE.search(scenario_hint or "")
    if not m:
        return []
    return [name for name in m.group(1).split("+") if name]


def discover_flagd_config() -> Optional[Path]:
    try:
        out = subprocess.run(
            ["docker", "inspect", "flagd", "--format", "{{json .Mounts}}"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return None
    for mount in json.loads(out or "[]"):
        if mount.get("Destination") == "/etc/flagd":
            candidate = Path(mount["Source"]) / "demo.flagd.json"
            if candidate.exists():
                return candidate
    return None


def _load(config_path: Path) -> dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _save(config_path: Path, cfg: dict[str, Any]) -> None:
    config_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")


def _most_severe_variant(flag_def: dict[str, Any]) -> Optional[str]:
    variants = flag_def.get("variants", {})
    if "on" in variants:
        return "on"
    for name in variants:
        if name != "off":
            return name
    return None


def activate_flags(config_path: Path, flag_names: list[str]) -> list[str]:
    cfg = _load(config_path)
    flags = cfg.get("flags", {})
    touched = []
    for name in flag_names:
        flag_def = flags.get(name)
        if flag_def is None:
            continue
        variant = _most_severe_variant(flag_def)
        if variant is not None:
            flag_def["defaultVariant"] = variant
            touched.append(name)
    if touched:
        _save(config_path, cfg)
    return touched


def reset_all_flags(config_path: Path) -> None:
    cfg = _load(config_path)
    for flag in cfg.get("flags", {}).values():
        if "off" in flag.get("variants", {}):
            flag["defaultVariant"] = "off"
    _save(config_path, cfg)


_RESOURCE_CPU_TAG = "[参数化扩增/resource_cpu]"
_QUEUE_BACKLOG_TAG = "[参数化扩增/queue_backlog]"
_DEPLOY_REGRESSION_TAG = "[参数化扩增/deploy_regression]"
_PURE_CODE_FIX_RANKING_TAG = "[参数化扩增/pure_code_fix_ranking]"
_PURE_INFO_ONLY_TAG = "[参数化扩增/pure_info_only]"
_PURE_CODE_FIX_DEDUPE_TAG = "[参数化扩增/pure_code_fix_dedupe]"
_HISTORICAL_TAG_PREFIX = "[历史工单反演/"

RESOURCE_CPU_FLAG = "adHighCpu"

QUEUE_BACKLOG_FLAG = "kafkaQueueProblems"

RECOMMENDATION_LEAK_IMAGE = "recommendation-leak-fixture:test"
RECOMMENDATION_CONTAINER = "recommendation"
RECOMMENDATION_BACKUP_CONTAINER = "recommendation-original-backup"
RECOMMENDATION_NETWORK = "opentelemetry-demo"

RECOMMENDATION_RANKING_IMAGE = "recommendation-ranking-fixture:test"
RECOMMENDATION_RANKING_HOST_PORT = 18080

RECOMMENDATION_DEDUPE_IMAGE = "recommendation-dedupe-fixture:test"
RECOMMENDATION_DEDUPE_HOST_PORT = 18081

_V9_AUG_TAG_PREFIX = "[v9增强/"

_HIST_CERT_KEYWORDS = ("证书",)
_HIST_LEAK_KEYWORDS = ("内存", "OOM")
_HIST_CPU_KEYWORDS = ("CPU",)
_HIST_QUEUE_KEYWORDS = ("消费", "lag", "堆积", "consumer")
_HIST_TIMEOUT_KEYWORDS = ("超时", "线程池", "级联")

_HIST_CPU_FLAGS = {"ad": RESOURCE_CPU_FLAG}
_HIST_DEPENDENCY_FLAGS = {
    "product-catalog": "productCatalogFailure",
    "cart": "cartFailure",
    "recommendation": "recommendationCacheFailure",
}


def _start_recommendation_fixture(
    image: str,
    env: dict[str, str],
    *,
    extra_run_args: Optional[list[str]] = None,
    publish_port: Optional[int] = None,
) -> list[str]:
    check = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=10,
    )
    if check.returncode != 0:
        return []
    backup_exists = subprocess.run(
        ["docker", "inspect", RECOMMENDATION_BACKUP_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    if backup_exists.returncode == 0:
        return []
    try:
        subprocess.run(
            ["docker", "stop", RECOMMENDATION_CONTAINER],
            capture_output=True,
            timeout=20,
            check=True,
        )
        subprocess.run(
            ["docker", "rename", RECOMMENDATION_CONTAINER, RECOMMENDATION_BACKUP_CONTAINER],
            capture_output=True,
            timeout=10,
            check=True,
        )
        env_args = []
        for k, v in env.items():
            env_args += ["-e", f"{k}={v}"]
        publish_args = ["-p", f"{publish_port}:8080"] if publish_port else []
        subprocess.run(
            [
                "docker", "run", "-d", "--name", RECOMMENDATION_CONTAINER,
                "--network", RECOMMENDATION_NETWORK,
                "--network-alias", RECOMMENDATION_CONTAINER,
                *(extra_run_args or []),
                *publish_args,
                *env_args,
                image,
            ],
            capture_output=True,
            timeout=20,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return []
    return [RECOMMENDATION_CONTAINER]


def start_recommendation_leak_fixture() -> list[str]:
    return _start_recommendation_fixture(
        RECOMMENDATION_LEAK_IMAGE,
        {"PORT": "9001", "LOAD_RPS": "400"},
        extra_run_args=["--memory=1073741824", "--memory-swap=1073741824"],
    )


def start_recommendation_ranking_fixture() -> list[str]:
    return _start_recommendation_fixture(
        RECOMMENDATION_RANKING_IMAGE,
        {"PORT": "8080", "LOAD_RPS": "0"},
        publish_port=RECOMMENDATION_RANKING_HOST_PORT,
    )


def start_recommendation_dedupe_fixture() -> list[str]:
    return _start_recommendation_fixture(
        RECOMMENDATION_DEDUPE_IMAGE,
        {"PORT": "8080", "LOAD_RPS": "0"},
        publish_port=RECOMMENDATION_DEDUPE_HOST_PORT,
    )


def stop_recommendation_leak_fixture() -> None:
    backup_exists = subprocess.run(
        ["docker", "inspect", RECOMMENDATION_BACKUP_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    if backup_exists.returncode != 0:
        return
    subprocess.run(["docker", "stop", RECOMMENDATION_CONTAINER], capture_output=True, timeout=20)
    subprocess.run(["docker", "rm", RECOMMENDATION_CONTAINER], capture_output=True, timeout=10)
    subprocess.run(
        ["docker", "rename", RECOMMENDATION_BACKUP_CONTAINER, RECOMMENDATION_CONTAINER],
        capture_output=True,
        timeout=10,
    )
    subprocess.run(["docker", "start", RECOMMENDATION_CONTAINER], capture_output=True, timeout=20)


def _activate_and_wait(config_path: Path, flag_names: list[str]) -> list[str]:
    touched = activate_flags(config_path, flag_names)
    if touched:
        time.sleep(1.5)
    return touched


def _inject_historical_fault(alert: dict[str, Any], config_path: Path, service: str) -> list[str]:
    annotations = alert.get("annotations", {}) or {}
    text = f"{annotations.get('summary', '')} {annotations.get('description', '')}"
    if any(k in text for k in _HIST_CERT_KEYWORDS):
        return []
    if any(k in text for k in _HIST_LEAK_KEYWORDS):
        if service == "recommendation":
            return start_recommendation_leak_fixture()
        return []
    if any(k in text for k in _HIST_CPU_KEYWORDS):
        flag = _HIST_CPU_FLAGS.get(service)
        if flag is None:
            return []
        return _activate_and_wait(config_path, [flag])
    if any(k in text for k in _HIST_QUEUE_KEYWORDS):
        return _activate_and_wait(config_path, [QUEUE_BACKLOG_FLAG])
    if any(k in text for k in _HIST_TIMEOUT_KEYWORDS):
        flag = _HIST_DEPENDENCY_FLAGS.get(service)
        if flag is None:
            return []
        return _activate_and_wait(config_path, [flag])
    return []


def inject_alert_fault(alert: dict[str, Any], config_path: Optional[Path]) -> list[str]:
    if config_path is None:
        return []
    hint = alert.get("scenario_hint", "") or ""

    flag_names = parse_flags(hint)
    if flag_names:
        return _activate_and_wait(config_path, flag_names)

    if _RESOURCE_CPU_TAG in hint:
        return _activate_and_wait(config_path, [RESOURCE_CPU_FLAG])

    if _QUEUE_BACKLOG_TAG in hint:
        return _activate_and_wait(config_path, [QUEUE_BACKLOG_FLAG])

    if _DEPLOY_REGRESSION_TAG in hint:
        return start_recommendation_leak_fixture()

    if _PURE_CODE_FIX_RANKING_TAG in hint:
        return start_recommendation_ranking_fixture()

    if _PURE_CODE_FIX_DEDUPE_TAG in hint:
        return start_recommendation_dedupe_fixture()

    if _PURE_INFO_ONLY_TAG in hint:
        return []

    if _V9_AUG_TAG_PREFIX in hint:
        return []

    if _HISTORICAL_TAG_PREFIX in hint:
        service = alert.get("service") or alert.get("labels", {}).get("service", "")
        return _inject_historical_fault(alert, config_path, service)

    return []


def reset_all(config_path: Path) -> None:
    reset_all_flags(config_path)
    stop_recommendation_leak_fixture()
