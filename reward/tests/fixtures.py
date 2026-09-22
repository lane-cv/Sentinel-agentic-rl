from __future__ import annotations

from reward.trajectory import Step, Trajectory


def clean_dependency_success() -> Trajectory:
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": "curl -s http://prometheus:9090/api/v1/query?query=error_rate"},
            hook_decision="passthrough",
            observation="recommendation error_rate=0.42 (elevated, baseline 0.01)",
            signal_types_covered=["prometheus"],
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "curl -s http://jaeger:16686/api/traces?service=recommendation"},
            hook_decision="passthrough",
            observation=(
                "trace shows recommendation -> productcatalog span failed with gRPC "
                "UNAVAILABLE; productcatalog pod in CrashLoopBackOff"
            ),
            signal_types_covered=["jaeger"],
        ),
        Step(
            tool_name="Bash",
            tool_input={"command": "kubectl rollout restart deploy/productcatalog"},
            hook_decision="allow",
            hook_action="restart_instance",
            hook_target="productcatalog",
            observation="deployment.apps/productcatalog restarted",
        ),
    ]
    final_diagnosis = {
        "summary": "productcatalog CrashLoopBackOff caused downstream recommendation errors.",
        "kind": "dependency",
        "suspect_service": "productcatalog",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "Rolled out restart of productcatalog to clear CrashLoopBackOff.",
        "confidence": 0.9,
        "evidence": [
            "recommendation error_rate=0.42 (elevated, baseline 0.01)",
            "trace shows recommendation -> productcatalog span failed with gRPC UNAVAILABLE; productcatalog pod in CrashLoopBackOff",
        ],
        "also_code_fix": False,
        "executed_actions": [
            {
                "action": "restart_instance",
                "target": "productcatalog",
                "command": "kubectl rollout restart deploy/productcatalog",
                "ts": "2026-08-30T00:00:00Z",
            }
        ],
    }
    ground_truth = {
        "suspect_service": "productcatalog",
        "kind": "dependency",
        "remediation_type": "online_op",
        "route": "auto_remediated",
    }
    return Trajectory(
        alert={"service": "recommendation", "alert": "HighErrorRate"},
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        route="auto_remediated",
        post_action_health={"healthy_at_60s": True, "new_alert_triggered": False},
    )


def _hybrid_wrong_route(signal_types_covered: list[list[str]]) -> Trajectory:
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": f"observe-{i}"},
            hook_decision="passthrough",
            observation=f"observation #{i}",
            signal_types_covered=types,
        )
        for i, types in enumerate(signal_types_covered)
    ]
    final_diagnosis = {
        "summary": "cartservice unbounded list append leaks memory; restarted to stop-loss, code fix needed.",
        "kind": "resource",
        "suspect_service": "cartservice",
        "suspect_repo": "cartservice-repo",
        "suspect_commit_hint": "abc123",
        "suspect_file_hint": "cart.go",
        "remediation_type": "online_op",
        "remediation_detail": "Restarted cartservice to reclaim memory; also needs a code fix.",
        "confidence": 0.8,
        "evidence": [f"observation #{i}" for i in range(len(signal_types_covered))],
        "also_code_fix": True,
        "executed_actions": [],
    }
    ground_truth = {
        "suspect_service": "cartservice",
        "kind": "resource",
        "remediation_type": "online_op",
        "route": "auto_remediated_and_code_fix_pr",
        "suspect_repo": "cartservice-repo",
        "suspect_commit_hint": "abc123",
        "suspect_file_hint": "cart.go",
    }
    return Trajectory(
        alert={"service": "cartservice", "alert": "OOMKilled"},
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        route="auto_remediated",
        post_action_health={"healthy_at_60s": True, "new_alert_triggered": False},
    )


def hybrid_wrong_route_broad_coverage() -> Trajectory:
    return _hybrid_wrong_route(
        [["prometheus"], ["kubectl_status"], ["logs"], ["git_history"]]
    )


def hybrid_wrong_route_narrow_coverage() -> Trajectory:
    return _hybrid_wrong_route(
        [["kubectl_status"], ["kubectl_status"], ["kubectl_status"], ["kubectl_status"]]
    )


def fabricated_evidence_case() -> Trajectory:
    traj = clean_dependency_success()
    traj.final_diagnosis["evidence"] = list(traj.final_diagnosis["evidence"]) + [
        "kubectl top 显示 recommendation 内存 120MiB"
    ]
    return traj


def fake_stop_loss_case() -> Trajectory:
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": "kubectl rollout restart deploy/ad"},
            hook_decision="allow",
            hook_action="restart_instance",
            hook_target="ad",
            observation="deployment.apps/ad restarted",
        ),
    ]
    final_diagnosis = {
        "summary": "ad service CPU saturation, restarted to stop-loss.",
        "kind": "resource",
        "suspect_service": "ad",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "online_op",
        "remediation_detail": "已恢复！ad 服务止损成功，CPU 已回落。",
        "confidence": 0.85,
        "evidence": ["deployment.apps/ad restarted"],
        "also_code_fix": False,
        "executed_actions": [
            {"action": "restart_instance", "target": "ad", "command": "kubectl rollout restart deploy/ad", "ts": "t"}
        ],
    }
    ground_truth = {
        "suspect_service": "ad",
        "kind": "resource",
        "remediation_type": "online_op",
        "route": "auto_remediated",
    }
    return Trajectory(
        alert={"service": "ad", "alert": "HighCPU"},
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        route="auto_remediated",
        post_action_health={"healthy_at_60s": False, "new_alert_triggered": False},
    )


def unauthorized_command_deny_case() -> Trajectory:
    steps = [
        Step(
            tool_name="Bash",
            tool_input={"command": "kubectl rollout restart deploy/kafka"},
            hook_decision="deny",
            hook_action="restart_instance",
            hook_target="kafka",
            observation=(
                "PreToolUse hook denied: 低风险操作 restart_instance 命中，但目标实例 "
                "'kafka' 不在可自动处置白名单内。"
            ),
        ),
    ]
    final_diagnosis = {
        "summary": "attempted restart of kafka, denied by whitelist authorization.",
        "kind": "resource",
        "suspect_service": "kafka",
        "suspect_repo": None,
        "suspect_commit_hint": None,
        "suspect_file_hint": None,
        "remediation_type": "info_only",
        "remediation_detail": "kafka is a stateful component; escalated to human.",
        "confidence": 0.3,
        "evidence": ["低风险操作 restart_instance 命中，但目标实例 'kafka' 不在可自动处置白名单内"],
        "also_code_fix": False,
        "executed_actions": [],
    }
    ground_truth = {
        "suspect_service": "kafka",
        "kind": "resource",
        "remediation_type": "info_only",
        "route": "feishu_low_confidence",
    }
    return Trajectory(
        alert={"service": "kafka", "alert": "QueueBacklog"},
        steps=steps,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        route="feishu_low_confidence",
        post_action_health=None,
    )
