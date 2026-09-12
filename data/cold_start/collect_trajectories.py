from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import signals
import fault_injection

REPO_ROOT = HERE.parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "Sentinel-agent"
DEFAULT_ALERTS_DIR = AIOPS_AGENT_DIR / "alerts"
DEFAULT_TRAJ_DIR = HERE / "trajectories"
DEFAULT_MOCK_DIR = DEFAULT_TRAJ_DIR / "mock"


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_aiops_core():
    _ensure_aiops_agent_on_path()
    from agent import config
    from agent.core import remediation
    from agent.core import schema
    from agent.core.hooks import guard_diagnose_ops

    return config, remediation, schema, guard_diagnose_ops


def check_env() -> dict[str, Any]:
    _ensure_aiops_agent_on_path()
    result: dict[str, Any] = {
        "aiops_agent_dir": str(AIOPS_AGENT_DIR),
        "aiops_agent_dir_exists": AIOPS_AGENT_DIR.is_dir(),
        "core_importable": False,
        "claude_agent_sdk_importable": False,
        "sdk_runner_importable": False,
        "errors": [],
    }
    try:
        _load_aiops_core()
        result["core_importable"] = True
    except Exception as exc:
        result["errors"].append(f"core import failed: {exc!r}")

    try:
        import claude_agent_sdk

        result["claude_agent_sdk_importable"] = True
    except Exception as exc:
        result["errors"].append(f"claude_agent_sdk import failed: {exc!r}")

    try:
        from agent.core import sdk_runner

        result["sdk_runner_importable"] = True
    except Exception as exc:
        result["errors"].append(f"sdk_runner import failed: {exc!r}")
    return result


def _stringify_tool_result(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(json.dumps(item, ensure_ascii=False))
        return "\n".join(parts)
    return json.dumps(content, ensure_ascii=False) if content is not None else ""


_KIND_TAXONOMY_GUIDANCE = """

## kind 字段分类口径（按故障的症状机制归类，不是按注入/触发手段归类）
- 队列/消息积压、消费延迟、CPU/内存/容量耗尽类症状 → `resource`（即使该故障是功能开关注入的，积压本身就是资源/容量类症状）
- 服务间依赖调用失败/超时/不可达 → `dependency`
- 发版引入的代码回归：内存泄漏、无界数据结构、业务逻辑 bug（排序/去重/计数/分页/计价等）→ `deploy_regression`
- 纯配置项/功能开关取值错误（开关只改变行为参数，不落入上述症状机制，如图片加载限速档位）→ `config`

## 矛盾信号的交叉验证纪律
若某个信号源的读数与告警或其它证据矛盾（例如 Prometheus 指标缺失、冻结为 0 或显示正常，但业务侧症状确实存在），不要单凭这一个读数否定告警——换第二信源交叉验证（日志 / 调用链 / OFREP 实时开关值），把两路证据都写进 evidence 再下结论。
"""


def _llm_env_overrides_real_teacher(config_mod) -> dict[str, str]:
    import os

    return {
        "ANTHROPIC_BASE_URL": config_mod.LLM_BASE_URL,
        "ANTHROPIC_API_KEY": config_mod.LLM_API_KEY,
        "ANTHROPIC_AUTH_TOKEN": config_mod.LLM_AUTH_TOKEN,
        "ANTHROPIC_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": config_mod.MODEL,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": config_mod.MODEL,
        "ANTHROPIC_SMALL_FAST_MODEL": config_mod.MODEL,
        "ANTHROPIC_CUSTOM_HEADERS": os.environ.get("ANTHROPIC_CUSTOM_HEADERS", ""),
        "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    }


def _write_teacher_settings_file(env: dict[str, str]) -> str:
    import json
    import tempfile

    d = tempfile.mkdtemp(prefix="aiops_teacher_settings_")
    p = Path(d) / "settings.json"
    p.write_text(json.dumps({"env": env}, indent=2), encoding="utf-8")
    return str(p)


def _make_recording_hook(inner_hook, remediation_mod, sink: dict[str, dict[str, Any]]):

    async def _hook(input_data, tool_use_id, context):
        result = await inner_hook(input_data, tool_use_id, context)
        decision_raw = (result.get("hookSpecificOutput") or {}).get("permissionDecision")
        decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
        action = target = None
        if input_data.get("tool_name") == "Bash":
            verdict = remediation_mod.classify_command(str(input_data.get("tool_input", {}).get("command", "")))
            if verdict.get("decision") in ("allow", "deny"):
                action, target = verdict.get("action"), verdict.get("target")
        sink[tool_use_id] = {"decision": decision, "action": action, "target": target}
        return result

    return _hook


async def _run_diagnose_capturing_steps(alert: dict[str, Any]) -> dict[str, Any]:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        HookMatcher,
        ResultMessage,
        ToolResultBlock,
        ToolUseBlock,
        UserMessage,
        query,
    )

    config, remediation, schema, guard_diagnose_ops = _load_aiops_core()
    from agent.agents import prompts
    from agent.integrations import rag_tool

    remediation.reset_ledger()
    alert_for_prompt = {k: v for k, v in alert.items() if k != "scenario_hint"}
    prompt = prompts.diagnose_prompt(alert_for_prompt)

    allowed = ["Bash", "Read", "Grep", "Glob"]
    mcp_servers = None
    if config.RAG_ENABLED:
        allowed.append(rag_tool.TOOL_NAME)
        mcp_servers = {"aiops_rag": rag_tool.server}

    hook_sink: dict[str, dict[str, Any]] = {}
    recording_hook = _make_recording_hook(guard_diagnose_ops, remediation, hook_sink)

    teacher_env = _llm_env_overrides_real_teacher(config)
    teacher_settings_path = _write_teacher_settings_file(teacher_env)

    opts = ClaudeAgentOptions(
        cwd=str(config.REPO_ROOT),
        model=config.MODEL,
        system_prompt={
            "type": "preset",
            "preset": "claude_code",
            "append": prompts.diagnose_append() + _KIND_TAXONOMY_GUIDANCE,
        },
        allowed_tools=allowed,
        permission_mode="acceptEdits",
        disallowed_tools=["Bash(rm -rf *)", "Bash(git push --force *)"],
        hooks={"PreToolUse": [HookMatcher(matcher="Bash", hooks=[recording_hook])]},
        setting_sources=["project"],
        max_turns=config.DIAGNOSE_MAX_TURNS,
        output_format={"type": "json_schema", "schema": schema.DIAGNOSIS_JSON_SCHEMA},
        mcp_servers=mcp_servers,
        skills="all",
        env=teacher_env,
        settings=teacher_settings_path,
    )

    pending_calls: dict[str, dict[str, Any]] = {}
    steps: list[dict[str, Any]] = []
    result_text: Optional[str] = None
    structured: Any = None
    degraded: Optional[str] = None
    usage: dict[str, int] = {}
    cost_usd: Optional[float] = None
    num_turns: Optional[int] = None
    model_usage: Any = None
    duration_ms: Optional[int] = None

    def _accumulate_usage(u: Optional[dict]) -> None:
        if not u:
            return
        for k in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        ):
            v = u.get(k)
            if isinstance(v, (int, float)):
                usage[k] = usage.get(k, 0) + int(v)

    try:
        async with asyncio.timeout(config.DIAGNOSE_TIMEOUT_S):
            async for msg in query(prompt=prompt, options=opts):
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        if isinstance(block, ToolUseBlock):
                            pending_calls[block.id] = {"tool_name": block.name, "tool_input": block.input}
                elif isinstance(msg, UserMessage):
                    for block in msg.content:
                        if isinstance(block, ToolResultBlock):
                            call = pending_calls.pop(block.tool_use_id, None)
                            if call is None:
                                continue
                            hook_info = hook_sink.get(block.tool_use_id, {})
                            steps.append(
                                {
                                    "tool_name": call["tool_name"],
                                    "tool_input": call["tool_input"],
                                    "hook_decision": hook_info.get("decision", "passthrough"),
                                    "hook_action": hook_info.get("action"),
                                    "hook_target": hook_info.get("target"),
                                    "observation": _stringify_tool_result(block.content),
                                }
                            )
                elif isinstance(msg, ResultMessage):
                    _accumulate_usage(msg.usage)
                    if msg.total_cost_usd is not None:
                        cost_usd = msg.total_cost_usd
                    num_turns = msg.num_turns
                    model_usage = msg.model_usage
                    duration_ms = msg.duration_ms
                    if msg.subtype == "success":
                        result_text = msg.result
                        structured = msg.structured_output
                    else:
                        degraded = msg.subtype
                        result_text = msg.result
                        structured = msg.structured_output
    except asyncio.TimeoutError:
        degraded = "timeout"
    except Exception as exc:
        degraded = f"sdk_error:{type(exc).__name__}"
        result_text = str(exc)[:500]

    signals.annotate_steps(steps)

    final_diagnosis: Optional[dict[str, Any]] = None
    diag_obj = None
    if structured is not None:
        try:
            diag_obj = schema.Diagnosis.model_validate(structured)
            diag_obj.executed_actions = [schema.ExecutedAction(**e) for e in remediation.get_ledger()]
            final_diagnosis = diag_obj.model_dump(by_alias=True)
        except Exception:
            diag_obj = schema.Diagnosis.fallback(
                f"structured_output 校验失败，raw={str(result_text)[:200]!r}"
            )
            final_diagnosis = diag_obj.model_dump(by_alias=True)
            degraded = degraded or "schema_validation_failed"

    route = _infer_route_real(diag_obj) if diag_obj is not None else None

    return {
        "alert": alert,
        "alert_id": alert.get("alertname") or alert.get("id") or "unknown",
        "steps": steps,
        "final_diagnosis": final_diagnosis,
        "ground_truth": None,
        "route": route,
        "post_action_health": None,
        "meta": {
            "source": "real",
            "degraded": degraded,
            "raw": result_text,
            "teacher_model": config.MODEL,
            "usage": usage,
            "cost_usd": cost_usd,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "model_usage": model_usage,
        },
    }


def real_mode(alerts_dir: Path, out_dir: Path, sample: Optional[int] = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    alert_files = sorted(alerts_dir.glob("*.json"))
    if sample is not None:
        alert_files = alert_files[:sample]

    flagd_config = fault_injection.discover_flagd_config()
    if flagd_config is None:
        print("[real] 没找到活的 flagd 容器/配置，本轮不注入 scenario_hint 里的 flag（跑的是环境当前的实际状态）。")

    for f in alert_files:
        alert_id = f.stem
        try:
            alert = json.loads(f.read_text(encoding="utf-8"))
            injected = fault_injection.inject_alert_fault(alert, flagd_config)
            if injected:
                print(f"[real] {alert_id} 注入故障：{injected}")
            try:
                traj = asyncio.run(_run_diagnose_capturing_steps(alert))
            finally:
                if flagd_config is not None:
                    fault_injection.reset_all(flagd_config)
        except Exception as exc:
            print(f"[real] {alert_id} 失败，跳过继续下一条：{type(exc).__name__}: {exc}")
            continue
        traj["alert_id"] = alert_id
        out_path = out_dir / f"{alert_id}.json"
        out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)
        print(f"[real] {alert_id} 完成（{len(traj.get('steps') or [])} 步，degraded={traj.get('meta', {}).get('degraded')}）")
    return written


def _step(
    tool_name: str,
    tool_input: dict[str, Any],
    observation: str,
    *,
    hook_decision: str = "passthrough",
    hook_action: Optional[str] = None,
    hook_target: Optional[str] = None,
) -> dict[str, Any]:
    return {
        "tool_name": tool_name,
        "tool_input": tool_input,
        "hook_decision": hook_decision,
        "hook_action": hook_action,
        "hook_target": hook_target,
        "observation": observation,
    }


def _hook_verdict_for_bash(guard_diagnose_ops, remediation_mod, command: str) -> dict[str, Any]:
    input_data = {"tool_name": "Bash", "tool_input": {"command": command}, "hook_event_name": "PreToolUse"}
    result = asyncio.run(guard_diagnose_ops(input_data, "mock-tool-use-id", {}))
    decision_raw = (result.get("hookSpecificOutput") or {}).get("permissionDecision")
    decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
    verdict = remediation_mod.classify_command(command)
    action = target = None
    if verdict.get("decision") in ("allow", "deny"):
        action, target = verdict.get("action"), verdict.get("target")
    return {"hook_decision": decision, "hook_action": action, "hook_target": target}


def build_mock_trajectories() -> list[dict[str, Any]]:
    _, remediation, schema, guard_diagnose_ops = _load_aiops_core()

    trajectories: list[dict[str, Any]] = []

    alert1 = {
        "alertname": "MockHighErrorRate",
        "service": "product-catalog",
        "labels": {"service": "product-catalog", "severity": "critical", "job": "product-catalog"},
        "annotations": {
            "summary": "product-catalog GetProduct 5xx 飙升",
            "description": "flagd productCatalogFailure 开关翻转后，product-catalog 依赖调用失败率飙升，"
            "上游 recommendation/frontend 大量报错。",
        },
        "startsAt": "2026-06-20T10:00:00Z",
    }
    restart_cmd = "docker restart product-catalog"
    steps1 = [
        _step("Bash", {"command": "docker ps"}, "product-catalog   Up 2 hours (unhealthy)\nrecommendation    Up 3 hours"),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:9090/api/v1/query?query=up{job=\"product-catalog\"}"},
            'Prometheus: up{job="product-catalog"} = 0，过去 5 分钟内错误率从 1% 跳升到 62%。',
        ),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:16686/api/traces?service=product-catalog&limit=5"},
            "Jaeger: 最近 5 条 trace 全部在 GetProduct span 返回 UNAVAILABLE，"
            "下游未再往前传播，故障止在 product-catalog 自身。",
        ),
        _step("Bash", {"command": "docker logs --tail 50 product-catalog"}, "ERROR flagd: feature flag "
              "'productCatalogFailure' evaluated to true, injecting synthetic 5xx"),
        _step(
            "Bash",
            {"command": restart_cmd},
            "PreToolUse hook 拒绝执行：低风险操作 restart_instance 命中，但目标 product-catalog "
            "不在可自动处置白名单内，需改发飞书卡片交人工重启。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, restart_cmd),
        ),
    ]
    diag1 = {
        "summary": "flagd productCatalogFailure 开关注入的合成依赖故障，需重启 product-catalog 清除注入状态。",
        "kind": "dependency",
        "suspect_service": "product-catalog",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "建议重启 product-catalog 清除注入状态；product-catalog 不在自动处置白名单内，已改发飞书卡片交人工执行。",
        "confidence": 0.88,
        "evidence": [
            'Prometheus up{job="product-catalog"}=0，错误率从 1% 跳升到 62%',
            "Jaeger 最近 5 条 trace 全部在 GetProduct span 返回 UNAVAILABLE",
            "docker logs 显示 flagd productCatalogFailure 开关命中，注入合成 5xx",
        ],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_dep_clean", alert1, steps1, diag1, schema, remediation))

    alert2 = {
        "alertname": "MockAdCpuThrottled",
        "service": "ad",
        "labels": {"service": "ad", "severity": "warning", "job": "ad"},
        "annotations": {
            "summary": "ad 服务 CPU 打满，p99 延迟飙升",
            "description": "ad 服务容器 CPU throttling 持续 100%，AdRequest p99 从 80ms 涨到 1.2s。",
        },
        "startsAt": "2026-06-20T11:00:00Z",
    }
    scale_cmd_bad = "docker exec -it ad top"
    scale_cmd_good = "docker update --cpus 2 --memory 1024m ad"
    steps2 = [
        _step("Bash", {"command": "docker stats --no-stream ad"}, "ad   CPU% 99.8   MEM 480MiB / 512MiB"),
        _step(
            "Bash",
            {"command": "curl -s http://localhost:9090/api/v1/query?query=rate(container_cpu_usage_seconds_total{name=\"ad\"}[1m])"},
            "Prometheus: ad 容器 CPU throttling 比例过去 10 分钟持续 100%。",
        ),
        _step(
            "Bash",
            {"command": scale_cmd_bad},
            "PreToolUse hook 拒绝执行：docker exec 属于进容器高风险操作，不在低风险白名单内。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, scale_cmd_bad),
        ),
        _step(
            "Bash",
            {"command": scale_cmd_good},
            "ad 容器资源上限已提升为 --cpus 2 --memory 1024m。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, scale_cmd_good),
        ),
    ]
    diag2 = {
        "summary": "ad 服务 CPU 打满导致延迟飙升，抬高资源上限后恢复。",
        "kind": "resource",
        "suspect_service": "ad",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "已临时抬高 ad 容器 CPU/内存上限缓解打满。",
        "confidence": 0.82,
        "evidence": [
            "docker stats 显示 ad 容器 CPU% 99.8",
            "Prometheus container_cpu_usage_seconds_total 显示 ad 持续 100% throttling",
        ],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_hookdeny_res", alert2, steps2, diag2, schema, remediation))

    alert3 = {
        "alertname": "MockRecommendationMemoryLeak",
        "service": "recommendation",
        "labels": {"service": "recommendation", "severity": "critical", "job": "recommendation"},
        "annotations": {
            "summary": "recommendation 内存单调上升且需先止血再改代码根治",
            "description": "recommendation 容器内存持续上涨、OOMKilled 次数飙升，怀疑最近一次发版引入了"
            "无界缓存；业务要求先扩容/重启止血，再触发代码修复 Agent 提 PR 根治。",
        },
        "startsAt": "2026-06-20T12:00:00Z",
    }
    restart3 = "docker restart recommendation"
    steps3 = [
        _step("Bash", {"command": "docker stats --no-stream recommendation"}, "recommendation   MEM 940MiB / 1024MiB（持续上涨）"),
        _step("Bash", {"command": "git -C /repo/recommendation log --oneline -5"},
              "abc1234 feat: cache popular items in-memory for latency\n"
              "d4e5f67 chore: bump grpc\n..."),
        _step("Bash", {"command": "git -C /repo/recommendation show abc1234 --stat"},
              "abc1234 修改 src/recommendation_server.py：新增全局 dict `_POPULAR_CACHE`，"
              "无过期/无容量上限，每次请求都会写入新 key。"),
        _step("Bash", {"command": "docker logs --tail 30 recommendation"}, "OOMKilled 事件 x3（过去 1 小时），"
              "MemoryError: cannot allocate memory"),
        _step(
            "Bash",
            {"command": restart3},
            "recommendation 已重启，内存回落到 120MiB。",
            **_hook_verdict_for_bash(guard_diagnose_ops, remediation, restart3),
        ),
    ]
    diag3 = {
        "summary": "commit abc1234 引入的无界内存缓存导致 OOM，已重启止血，需代码修复根治。",
        "kind": "deploy_regression",
        "suspect_service": "recommendation",
        "suspect_repo": "recommendation",
        "suspect_commit_hint": "abc1234",
        "suspect_file_hint": "src/recommendation_server.py",
        "remediation_type": "online_op",
        "remediation_detail": "已滚动重启 recommendation 清空内存，同时需要修复 abc1234 引入的无界缓存。",
        "confidence": 0.9,
        "evidence": [
            "docker stats 显示 recommendation 内存持续上涨至 940MiB/1024MiB",
            "git show abc1234 显示新增无过期/无容量上限的全局 dict 缓存",
            "docker logs 显示过去 1 小时 OOMKilled 事件 x3",
        ],
        "also_code_fix": True,
    }
    trajectories.append(_assemble("mock_hybrid_leak", alert3, steps3, diag3, schema, remediation))

    alert4 = {
        "alertname": "MockFabricatedEvidence",
        "service": "checkout",
        "labels": {"service": "checkout", "severity": "critical", "job": "checkout"},
        "annotations": {"summary": "checkout 错误率上升", "description": "checkout 服务错误率上升。"},
        "startsAt": "2026-06-20T13:00:00Z",
    }
    steps4 = [
        _step("Bash", {"command": "docker ps"}, "checkout   Up 1 hour (healthy)"),
    ]
    diag4 = {
        "summary": "checkout 数据库连接池耗尽。",
        "kind": "resource",
        "suspect_service": "checkout",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "已重启 checkout。",
        "confidence": 0.7,
        "evidence": ["kubectl top 显示 checkout 数据库连接池占用 100%，触发连接超时"],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_fail_evidence", alert4, steps4, diag4, schema, remediation))

    alert5 = {
        "alertname": "MockMismatchedConclusion",
        "service": "currency",
        "labels": {"service": "currency", "severity": "warning", "job": "currency"},
        "annotations": {"summary": "currency 延迟上升", "description": "currency 服务延迟上升。"},
        "startsAt": "2026-06-20T14:00:00Z",
    }
    steps5 = [
        _step("Bash", {"command": "docker stats --no-stream currency"}, "currency   CPU% 97.0"),
    ]
    diag5 = {
        "summary": "currency 服务判定为 info_only，无需处置。",
        "kind": "resource",
        "suspect_service": "payment",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "info_only",
        "remediation_detail": "无需处置。",
        "confidence": 0.75,
        "evidence": ["docker stats 显示 currency CPU% 97.0"],
        "also_code_fix": False,
    }
    trajectories.append(_assemble("mock_fail_mismatch", alert5, steps5, diag5, schema, remediation))

    return trajectories


def _assemble(
    alert_id: str,
    alert: dict[str, Any],
    steps: list[dict[str, Any]],
    diag_fields: dict[str, Any],
    schema_mod,
    remediation_mod,
) -> dict[str, Any]:
    signals.annotate_steps(steps)

    executed_actions = [
        {
            "action": s["hook_action"],
            "target": s["hook_target"],
            "command": s["tool_input"].get("command", ""),
            "ts": "2026-06-20T00:00:00Z",
        }
        for s in steps
        if s["hook_decision"] == "allow"
    ]
    diag = schema_mod.Diagnosis.model_validate(diag_fields)
    diag.executed_actions = [schema_mod.ExecutedAction(**e) for e in executed_actions]

    route = _infer_route(diag)
    post_health = None
    if diag.remediation_type == "online_op" or executed_actions:
        post_health = {
            "checked": True,
            "healthy": True,
            "metric": "container_memory_usage_bytes" if diag.kind == "deploy_regression" else "http_error_rate",
            "window_s": 60,
        }

    return {
        "alert": alert,
        "alert_id": alert_id,
        "steps": steps,
        "final_diagnosis": diag.model_dump(by_alias=True),
        "ground_truth": None,
        "route": route,
        "post_action_health": post_health,
        "meta": {"source": "mock", "degraded": None, "raw": None},
    }


def _infer_route(diag) -> str:
    if diag.confidence < 0.6:
        return "feishu_low_confidence"
    if diag.remediation_type == "online_op":
        auto_done = bool(diag.executed_actions)
        if diag.also_code_fix:
            return "auto_remediated_and_code_fix_pr" if auto_done else "online_op_and_code_fix_pr"
        return "auto_remediated" if auto_done else "feishu_online_op"
    if diag.remediation_type == "code_fix":
        return "code_fix_pr"
    return "info_only"


def _infer_route_real(diag) -> Optional[str]:
    if diag.confidence < 0.6:
        return _infer_route(diag)
    if diag.remediation_type == "code_fix":
        return None
    if diag.remediation_type == "online_op" and diag.also_code_fix:
        return None
    return _infer_route(diag)


def mock_mode(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for traj in build_mock_trajectories():
        out_path = out_dir / f"{traj['alert_id']}.json"
        out_path.write_text(json.dumps(traj, ensure_ascii=False, indent=2), encoding="utf-8")
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="收集冷启轨迹（真实模式 / --mock 合成模式）")
    parser.add_argument("--mock", action="store_true", help="生成合成轨迹而不是真实调用 Agent")
    parser.add_argument("--check-env", action="store_true", help="只检查 Sentinel-agent 环境是否可 import，不发起任何调用")
    parser.add_argument("--alerts-dir", type=Path, default=DEFAULT_ALERTS_DIR, help="真实模式：告警 JSON 所在目录")
    parser.add_argument("--out-dir", type=Path, default=None, help="输出目录（默认真实模式写 trajectories/，mock 模式写 trajectories/mock/）")
    parser.add_argument("--sample", type=int, default=None, help="真实模式：只跑目录下前 N 条告警（不传则跑全部）")
    args = parser.parse_args()

    if args.check_env:
        print(json.dumps(check_env(), ensure_ascii=False, indent=2))
        return

    if args.mock:
        out_dir = args.out_dir or DEFAULT_MOCK_DIR
        written = mock_mode(out_dir)
        print(f"[mock] 写入 {len(written)} 条合成轨迹到 {out_dir}")
        for p in written:
            print(f"  - {p}")
        return

    out_dir = args.out_dir or DEFAULT_TRAJ_DIR
    scope = f"前 {args.sample} 条" if args.sample is not None else "全部"
    print(
        "[real] 即将对 "
        f"{args.alerts_dir} 下的{scope}告警发起真实 Agent 调用（需要真实 API key + docker/k8s 环境）。"
    )
    written = real_mode(args.alerts_dir, out_dir, sample=args.sample)
    print(f"[real] 写入 {len(written)} 条轨迹到 {out_dir}")


if __name__ == "__main__":
    main()
