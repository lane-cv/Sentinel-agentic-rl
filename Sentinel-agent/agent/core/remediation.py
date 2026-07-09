from __future__ import annotations

import re
import time
from typing import Any, Optional

from agent import config


ALLOWED_TARGETS: set[str] = set(
    t.strip()
    for t in config.REMEDIATION_ALLOWED_TARGETS.split(",")
    if t.strip()
)

_TARGET = r"[a-zA-Z0-9][a-zA-Z0-9._-]*"
_NS = r"[a-z0-9][a-z0-9-]*"


class LowRiskAction:

    def __init__(self, action: str, pattern: str, description: str) -> None:
        self.action = action
        self.description = description
        self._re = re.compile(pattern)

    def match(self, command: str) -> Optional[str]:
        m = self._re.match(command.strip())
        if not m:
            return None
        return m.group("target")


_NS_OPT = rf"(?:-n\s+{_NS}\s+|--namespace\s+{_NS}\s+)?"
_K8S_ACTIONS: list[LowRiskAction] = [
    LowRiskAction(
        "restart_instance",
        rf"^kubectl\s+{_NS_OPT}rollout\s+restart\s+(?:deployment|deploy)/(?P<target>{_TARGET})$",
        "滚动重启工作负载：kubectl rollout restart deploy/<svc>（逐个换 Pod 止血，如清掉泄漏进程的内存、重置卡死状态）。",
    ),
    LowRiskAction(
        "migrate_instance",
        rf"^kubectl\s+{_NS_OPT}delete\s+pod\s+(?:-l|--selector)\s+app=(?P<target>{_TARGET})$",
        "迁移/重建工作负载 Pod：kubectl delete pod -l app=<svc>（控制器立即重建，等价于重新调度到新节点）。",
    ),
    LowRiskAction(
        "scale_resources",
        rf"^kubectl\s+{_NS_OPT}set\s+resources\s+(?:deployment|deploy)/(?P<target>{_TARGET})\s+--limits=(?:cpu=[0-9.]+m?|memory=\d+[EPTGMK]i?)(?:,(?:cpu=[0-9.]+m?|memory=\d+[EPTGMK]i?))*$",
        "抬高工作负载资源上限：kubectl set resources deploy/<svc> --limits=cpu=..,memory=..（缓解 CPU/内存打满）。",
    ),
]

_DOCKER_ACTIONS: list[LowRiskAction] = [
    LowRiskAction(
        "restart_instance",
        rf"^docker\s+restart\s+(?:-t\s+\d+\s+)?(?P<target>{_TARGET})$",
        "重启单实例：docker restart <实例>（临时止血，如清掉泄漏进程的内存、重置卡死状态）。",
    ),
    LowRiskAction(
        "migrate_instance",
        rf"^docker(?:\s+compose|-compose)\s+up\s+-d\s+--force-recreate\s+(?P<target>{_TARGET})$",
        "迁移/重建单实例：docker compose up -d --force-recreate <实例>（或 docker-compose，视本机装的是插件还是独立二进制）（把实例重新拉起，等价于重新调度）。",
    ),
    LowRiskAction(
        "scale_resources",
        rf"^docker\s+update\s+(?:--cpus\s+[0-9.]+\s+|--memory\s+\d+[bkmgBKMG]*\s+|--memory-swap\s+\d+[bkmgBKMG]*\s+)+(?P<target>{_TARGET})$",
        "扩容单实例资源上限：docker update --cpus <n> --memory <m> <实例>（缓解 CPU/内存打满，副本级弹性）。",
    ),
]

LOW_RISK_ACTIONS: list[LowRiskAction] = (
    _K8S_ACTIONS if config.REMEDIATION_BACKEND == "k8s" else _DOCKER_ACTIONS
)


def classify_command(command: str) -> dict[str, Any]:
    if not config.REMEDIATION_ENABLED:
        return {"decision": "passthrough"}

    cmd = command.strip()
    for act in LOW_RISK_ACTIONS:
        target = act.match(cmd)
        if target is None:
            continue
        if target not in ALLOWED_TARGETS:
            return {
                "decision": "deny",
                "action": act.action,
                "target": target,
                "reason": (
                    f"低风险操作 {act.action} 命中，但目标实例 '{target}' 不在可自动处置白名单内"
                    f"（允许：{sorted(ALLOWED_TARGETS)}）。有状态组件/未授权目标一律交人工。"
                ),
            }
        return {"decision": "allow", "action": act.action, "target": target}
    return {"decision": "passthrough"}


_LEDGER: list[dict[str, Any]] = []


def reset_ledger() -> None:
    _LEDGER.clear()


def record_execution(action: str, target: str, command: str) -> None:
    _LEDGER.append(
        {
            "action": action,
            "target": target,
            "command": command,
            "ts": _now(),
        }
    )


def get_ledger() -> list[dict[str, Any]]:
    return [dict(x) for x in _LEDGER]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def actions_catalog_md() -> str:
    backend = "k8s（kubectl）" if config.REMEDIATION_BACKEND == "k8s" else "docker"
    lines = [f"- `{a.action}`：{a.description}" for a in LOW_RISK_ACTIONS]
    targets = ", ".join(sorted(ALLOWED_TARGETS)) or "（未配置任何可处置目标）"
    return (
        f"当前处置后端：**{backend}**（仅下列命令形态会被放行；换后端由 `AIOPS_BACKEND` 控制）。\n"
        + "\n".join(lines)
        + f"\n\n可自动处置的目标（白名单，仅限这些工作负载）：{targets}"
    )
