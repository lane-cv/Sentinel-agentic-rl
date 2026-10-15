from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
from typing import Any

import pytest

from verl_adapter import observation_packer
from verl_adapter.rollout_worker import (
    RealHookRoutedTool,
    SampledToolCall,
    ToolResponseShim,
    execute_tool_call,
    route_through_real_hook,
    ssh_tunnel_bash_executor,
    tunnel_ssh_argv,
)


def _cfg(tool_name: str, **extra: Any) -> dict[str, Any]:
    return {"type": "native", "tool_name": tool_name, **extra}


class _FakeProc:

    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def _install_fake_subprocess_run(monkeypatch: pytest.MonkeyPatch, handler) -> list[tuple[Any, dict]]:

    calls: list[tuple[Any, dict]] = []

    def fake_run(args, **kwargs):
        calls.append((args, kwargs))
        return handler(args, kwargs)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return calls


def _recording_executor(results: list[tuple[str, str, int]], seen: list[str]):

    def executor(command: str) -> tuple[str, str, int]:
        seen.append(command)
        return results.pop(0) if results else ("", "", 0)

    return executor


@pytest.fixture(autouse=True)
def _no_mac_agent_dir(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("AIOPS_MAC_AGENT_DIR", raising=False)


class _FakeAsyncProc:

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0) -> None:
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, self._stderr

    def kill(self) -> None:
        self.killed = True

    async def wait(self) -> int:
        return self.returncode


def _install_fake_async_exec(monkeypatch: pytest.MonkeyPatch, handler) -> list[tuple[Any, ...]]:

    calls: list[tuple[Any, ...]] = []

    async def fake_exec(*argv, **kwargs):
        calls.append(argv)
        return handler(argv)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    return calls


def test_tunnel_ssh_argv_defaults(monkeypatch: pytest.MonkeyPatch):
    for var in ("AIOPS_TUNNEL_PORT", "AIOPS_TUNNEL_USER", "AIOPS_TUNNEL_HOST"):
        monkeypatch.delenv(var, raising=False)
    assert tunnel_ssh_argv() == ["ssh", "-p", "2222", "--", "changeme@localhost"]


def test_tunnel_ssh_argv_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIOPS_TUNNEL_PORT", "22022")
    monkeypatch.setenv("AIOPS_TUNNEL_USER", "ops")
    monkeypatch.setenv("AIOPS_TUNNEL_HOST", "env-host.internal")
    assert tunnel_ssh_argv() == ["ssh", "-p", "22022", "--", "ops@env-host.internal"]


def test_ssh_executor_sends_command_as_one_byte_exact_argv_element(monkeypatch: pytest.MonkeyPatch):
    command = """echo "it's a 'mixed' \"quote\" $(echo not-expanded-here)" | tr a-z A-Z"""
    calls = _install_fake_subprocess_run(monkeypatch, lambda a, k: _FakeProc("out", "err", 0))

    stdout, stderr, exit_code = ssh_tunnel_bash_executor(command, timeout_s=42.5)

    args, kwargs = calls[0]
    assert args == ["ssh", "-p", "2222", "--", "changeme@localhost", command]
    assert args[-1] == command
    assert not kwargs.get("shell")
    assert kwargs["capture_output"] is True and kwargs["text"] is True
    assert kwargs["timeout"] == 42.5
    assert (stdout, stderr, exit_code) == ("out", "err", 0)


def test_ssh_executor_mixed_quotes_survive_a_real_shell_round_trip(monkeypatch: pytest.MonkeyPatch):
    real_run = subprocess.run
    command = """echo "it's O'Reilly's \\"book\\" and 'single' too" """
    calls = _install_fake_subprocess_run(
        monkeypatch,
        lambda a, k: real_run(a[-1], shell=True, capture_output=True, text=True, timeout=10),
    )

    stdout, stderr, exit_code = ssh_tunnel_bash_executor(command)

    assert exit_code == 0, stderr
    assert stdout == 'it\'s O\'Reilly\'s "book" and \'single\' too\n'
    assert len(calls) == 1


def test_ssh_executor_cwd_becomes_quoted_cd_prefix(monkeypatch: pytest.MonkeyPatch):
    cwd = "/Users/testuser/work dir/with'quote"
    calls = _install_fake_subprocess_run(monkeypatch, lambda a, k: _FakeProc("", "", 0))

    ssh_tunnel_bash_executor("pwd", cwd=cwd)

    args, _ = calls[0]
    assert args[-1] == f"cd {shlex.quote(cwd)} && pwd"


def test_ssh_executor_cwd_actually_changes_directory_round_trip(
    monkeypatch: pytest.MonkeyPatch, tmp_path
):
    spicy = tmp_path / "work dir"
    spicy.mkdir()
    real_run = subprocess.run
    _install_fake_subprocess_run(
        monkeypatch,
        lambda a, k: real_run(a[-1], shell=True, capture_output=True, text=True, timeout=10),
    )

    stdout, stderr, exit_code = ssh_tunnel_bash_executor("pwd", cwd=str(spicy))

    assert exit_code == 0, stderr
    assert stdout.strip() == str(spicy)


def test_ssh_executor_returns_proc_triple_in_order(monkeypatch: pytest.MonkeyPatch):
    _install_fake_subprocess_run(monkeypatch, lambda a, k: _FakeProc("so", "se", 3))
    assert ssh_tunnel_bash_executor("anything") == ("so", "se", 3)


def test_ssh_executor_without_uses_default_remote_cwd(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_subprocess_run(monkeypatch, lambda a, k: _FakeProc())
    ssh_tunnel_bash_executor("docker ps")
    assert calls[0][0][-1] == "docker ps"


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected_command"),
    [
        ("Read", {"file_path": "deploys.log"}, "cat deploys.log"),
        ("Read", {"file_path": "workspace/payment/main.py"}, "cat workspace/payment/main.py"),
        ("Read", {"file_path": "my deploys.log"}, "cat 'my deploys.log'"),
        ("Grep", {"pattern": "RETRY_LIMIT", "path": "workspace/payment/"}, "grep -rE -- RETRY_LIMIT workspace/payment/"),
        ("Grep", {"pattern": "OOMKilled"}, "grep -rE -- OOMKilled ."),
        ("Grep", {"pattern": "foo bar"}, "grep -rE -- 'foo bar' ."),
        ("Glob", {"pattern": "workspace/*/deploys.log"}, "find . -path './workspace/*/deploys.log' -print"),
        ("Glob", {"pattern": "**/*.py"}, "find . -path './**/*.py' -print"),
        ("Glob", {"pattern": "./**/*.py"}, "find . -path './**/*.py' -print"),
    ],
)
def test_read_grep_glob_map_to_remote_commands(tool_name, tool_input, expected_command):
    seen: list[str] = []
    executor = _recording_executor([("line1\nline2\n", "", 0)], seen)

    result, verdict = asyncio.run(
        execute_tool_call(SampledToolCall("tu-1", tool_name, tool_input), bash_executor=executor)
    )

    assert seen == [expected_command]
    assert result.content == observation_packer.pack_bash_exec_result("line1\nline2\n", "", 0)
    assert result.content == "line1\nline2"
    assert result.is_error is False
    assert verdict["decision"] == "passthrough"


def test_read_grep_glob_failing_exit_code_is_surfaced_not_swallowed():
    seen: list[str] = []
    executor = _recording_executor([("", "cat: nope: No such file or directory", 1)], seen)

    result, verdict = asyncio.run(
        execute_tool_call(SampledToolCall("tu-2", "Read", {"file_path": "nope"}), bash_executor=executor)
    )

    assert seen == ["cat nope"]
    assert result.is_error is True
    assert "No such file or directory" in result.content
    assert "(exit_code=1)" in result.content


def test_read_grep_glob_go_through_the_ssh_executor_seam_when_wired(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_subprocess_run(monkeypatch, lambda a, k: _FakeProc("log-line", "", 0))

    result, _ = asyncio.run(
        execute_tool_call(
            SampledToolCall("tu-3", "Read", {"file_path": "deploys.log"}),
            bash_executor=ssh_tunnel_bash_executor,
        )
    )

    args, kwargs = calls[0]
    assert args == ["ssh", "-p", "2222", "--", "changeme@localhost", "cat deploys.log"]
    assert not kwargs.get("shell")
    assert result.content == "log-line"


def test_execute_tool_call_threads_cwd_to_the_executor():

    def spying_executor(command: str, cwd: str = None) -> tuple[str, str, int]:
        spying_executor.seen.append((command, cwd))
        return ("", "", 0)

    spying_executor.seen = []

    asyncio.run(
        execute_tool_call(
            SampledToolCall("c1", "Glob", {"pattern": "**/*.py"}),
            bash_executor=spying_executor,
            cwd="/agent-workspace",
        )
    )
    asyncio.run(
        execute_tool_call(
            SampledToolCall("c2", "Bash", {"command": "cat deploys.log"}),
            bash_executor=spying_executor,
            cwd="/agent-workspace",
        )
    )

    assert spying_executor.seen == [
        ("find . -path './**/*.py' -print", "/agent-workspace"),
        ("cat deploys.log", "/agent-workspace"),
    ]


def test_execute_tool_call_without_cwd_keeps_one_arg_executor_shape():

    def strict_one_arg_executor(command: str) -> tuple[str, str, int]:
        strict_one_arg_executor.seen.append(command)
        return ("", "", 0)

    strict_one_arg_executor.seen = []

    asyncio.run(
        execute_tool_call(
            SampledToolCall("c3", "Read", {"file_path": "deploys.log"}),
            bash_executor=strict_one_arg_executor,
        )
    )

    assert strict_one_arg_executor.seen == ["cat deploys.log"]


def test_realhookroutedtool_cwd_threads_through_to_the_executor():
    seen: list[tuple[str, str]] = []

    def executor(command: str, cwd: str = None) -> tuple[str, str, int]:
        seen.append((command, cwd))
        return ("", "", 0)

    tool = RealHookRoutedTool(_cfg("Glob"), bash_executor=executor, cwd="/agent-workspace")
    instance_id, _ = asyncio.run(tool.create(create_kwargs={}))
    asyncio.run(tool.execute(instance_id, {"pattern": "*.log"}, agent_data=object()))

    assert seen == [("find . -path './*.log' -print", "/agent-workspace")]


def test_create_is_called_the_way_tool_agent_loop_calls_it():
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=lambda c: ("", "", 0))

    instance_id, creation_response = asyncio.run(tool.create(create_kwargs={}))

    assert isinstance(instance_id, str) and instance_id
    assert isinstance(creation_response, ToolResponseShim) and creation_response.text == ""


def test_create_generates_fresh_ids_for_parallel_instances():
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=lambda c: ("", "", 0))
    first, _ = asyncio.run(tool.create(create_kwargs={}))
    second, _ = asyncio.run(tool.create(create_kwargs={}))
    assert first != second


def test_create_passthrough_explicit_instance_id():
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=lambda c: ("", "", 0))
    assert asyncio.run(tool.create(instance_id="rollout-7")) == ("rollout-7", ToolResponseShim(text=""))


def test_full_dispatch_flow_exactly_as_tool_agent_loop_runs_it():
    seen: list[str] = []
    executor = _recording_executor([("Healthy\n", "", 0)], seen)
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=executor)
    agent_data = object()
    tools_kwargs: dict[str, dict[str, Any]] = {"Bash": {"create_kwargs": {}}}

    instance_id, _ = asyncio.run(tool.create(create_kwargs=tools_kwargs.get("create_kwargs", {})))
    tool_response, tool_reward, tool_metrics = asyncio.run(
        tool.execute(instance_id, {"command": "curl -s http://localhost:9090/-/healthy"}, agent_data=agent_data)
    )
    asyncio.run(tool.release(instance_id))

    assert seen == ["curl -s http://localhost:9090/-/healthy"]
    assert tool_response == ToolResponseShim(text="Healthy")
    assert tool_reward == 0.0
    assert tool_metrics["hook_decision"] == "passthrough"


def test_every_execute_routes_through_real_hook_deny_never_runs():
    seen: list[str] = []
    executor = _recording_executor([], seen)
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=executor)

    instance_id, _ = asyncio.run(tool.create(create_kwargs={}))
    tool_response, _, tool_metrics = asyncio.run(
        tool.execute(instance_id, {"command": "docker exec -it ad top"}, agent_data=object())
    )

    assert seen == []
    assert tool_metrics["hook_decision"] == "deny"
    real_verdict = asyncio.run(
        route_through_real_hook(SampledToolCall("x", "Bash", {"command": "docker exec -it ad top"}))
    )
    assert tool_response.text == real_verdict["reason"]
    assert "禁止" in tool_response.text


def test_every_execute_routes_through_real_hook_allow_reaches_executor():
    seen: list[str] = []
    executor = _recording_executor([("ad\n", "", 0)], seen)
    tool = RealHookRoutedTool(_cfg("Bash"), bash_executor=executor)

    instance_id, _ = asyncio.run(tool.create(create_kwargs={}))
    tool_response, _, tool_metrics = asyncio.run(
        tool.execute(
            instance_id, {"command": "docker update --cpus 2 --memory 512m ad"}, agent_data=object()
        )
    )

    assert seen == ["docker update --cpus 2 --memory 512m ad"]
    assert tool_metrics == {"hook_decision": "allow", "hook_action": "scale_resources", "hook_target": "ad"}
    assert tool_response == ToolResponseShim(text="ad")


def test_read_tool_through_realhookrouted_lifecycle():
    seen: list[str] = []
    executor = _recording_executor([("deploy log line\n", "", 0)], seen)
    tool = RealHookRoutedTool(_cfg("Read"), bash_executor=executor)

    instance_id, _ = asyncio.run(tool.create(create_kwargs={}))
    tool_response, _, tool_metrics = asyncio.run(
        tool.execute(instance_id, {"file_path": "deploys.log"}, agent_data=object())
    )
    asyncio.run(tool.release(instance_id))

    assert seen == ["cat deploys.log"]
    assert tool_response == ToolResponseShim(text="deploy log line")
    assert tool_metrics["hook_decision"] == "passthrough"


from verl_adapter.rollout_worker import (
    AIOPS_MAC_AGENT_DIR_ENV,
    RAG_TOOL_NAME,
    ssh_tunnel_mcp_executor,
)

_GOOD_RESULT = {"content": [{"type": "text", "text": "检索到以下历史相似诊断工单……"}]}


def test_mcp_executor_primary_tier_curls_the_rag_service(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_async_exec(
        monkeypatch, lambda argv: _FakeAsyncProc(stdout=json.dumps(_GOOD_RESULT).encode())
    )
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "kafka 消费'积压\""}))
    assert result == _GOOD_RESULT
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:4] == ("ssh", "-p", "2222", "--")
    assert argv[4] == "changeme@localhost"
    remote = argv[5]
    assert remote.startswith("curl ")
    assert "http://localhost:18765/search" in remote
    assert shlex.quote('{"query": "kafka 消费\'积压\\""}') in remote


def test_mcp_executor_service_port_env_override(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIOPS_RAG_SERVICE_PORT", "29999")
    calls = _install_fake_async_exec(
        monkeypatch, lambda argv: _FakeAsyncProc(stdout=json.dumps(_GOOD_RESULT).encode())
    )
    asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {}))
    assert "http://localhost:29999/search" in calls[0][5]


def test_mcp_executor_falls_back_to_bridge_when_service_unreachable(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_async_exec(
        monkeypatch,
        lambda argv: (
            _FakeAsyncProc(stderr=b"curl: (7) Failed to connect", returncode=7)
            if "curl" in argv[5]
            else _FakeAsyncProc(stdout=json.dumps(_GOOD_RESULT).encode())
        ),
    )
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "x"}))
    assert result == _GOOD_RESULT
    assert len(calls) == 2
    assert "curl" in calls[0][5]
    assert "rag_bridge.py" in calls[1][5] and ".venv/bin/python" in calls[1][5]


def test_mcp_executor_fallback_uses_mac_env_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AIOPS_MAC_PYTHON", "/opt/other/python")
    monkeypatch.setenv("AIOPS_MAC_REPO_ROOT", "/somewhere/else")
    monkeypatch.setenv("AIOPS_MAC_BRIDGE_PATH", "bridge/rag.py")
    calls = _install_fake_async_exec(
        monkeypatch, lambda argv: _FakeAsyncProc(stderr=b"refused", returncode=7)
    )
    asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {}))
    fallback_remote = calls[1][5]
    assert "/opt/other/python" in fallback_remote and "/somewhere/else/bridge/rag.py" in fallback_remote


def test_mcp_executor_both_tiers_failing_returns_error_not_raise(monkeypatch: pytest.MonkeyPatch):
    _install_fake_async_exec(
        monkeypatch, lambda argv: _FakeAsyncProc(stderr=b"ssh: connect refused", returncode=255)
    )
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "x"}))
    assert result["is_error"] is True
    assert "rag_service unreachable" in result["content"][0]["text"]


def test_mcp_executor_non_json_stdout_returns_error_without_fallback(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_async_exec(monkeypatch, lambda argv: _FakeAsyncProc(stdout=b"garbage not json"))
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "x"}))
    assert result["is_error"] is True
    assert "not JSON" in result["content"][0]["text"]
    assert len(calls) == 1


def test_mcp_executor_unknown_tool_returns_error(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_async_exec(monkeypatch, lambda argv: _FakeAsyncProc())
    result = asyncio.run(ssh_tunnel_mcp_executor("mcp__other__tool", {}))
    assert result["is_error"] is True
    assert calls == []


def test_mcp_executor_timeout_on_service_falls_back_to_bridge(monkeypatch: pytest.MonkeyPatch):
    class _HangingProc(_FakeAsyncProc):
        async def communicate(self) -> tuple[bytes, bytes]:
            await asyncio.sleep(30)

    hanging = _HangingProc()
    good = _FakeAsyncProc(stdout=json.dumps(_GOOD_RESULT).encode())
    state = {"n": 0}

    async def fake_exec(*argv, **kwargs):
        state["n"] += 1
        return hanging if state["n"] == 1 else good

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_exec)
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "x"}, timeout_s=0.05))
    assert result == _GOOD_RESULT
    assert hanging.killed is True


def test_mcp_executor_success_preserves_rag_result_shape(monkeypatch: pytest.MonkeyPatch):
    rag_result = {"content": [{"type": "text", "text": "结果"}], "is_error": False}
    _install_fake_async_exec(
        monkeypatch, lambda argv: _FakeAsyncProc(stdout=json.dumps(rag_result).encode())
    )
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "x"}))
    assert result == rag_result


def test_init_matches_verl_registry_call_convention():
    tool = RealHookRoutedTool(_cfg("Bash"), tool_schema={"function": {"name": "Bash"}})
    assert tool.name == "Bash"
    assert tool.tool_schema == {"function": {"name": "Bash"}}
    assert tool._bash_executor is ssh_tunnel_bash_executor
    assert tool._mcp_executor is ssh_tunnel_mcp_executor
    assert tool._cwd is None


def test_init_cwd_resolution_order(monkeypatch: pytest.MonkeyPatch):
    assert RealHookRoutedTool(_cfg("Bash"))._cwd is None
    monkeypatch.setenv(AIOPS_MAC_AGENT_DIR_ENV, "/from/env")
    assert RealHookRoutedTool(_cfg("Bash"))._cwd == "/from/env"
    assert RealHookRoutedTool(_cfg("Bash", cwd="/from/config"))._cwd == "/from/config"
    assert RealHookRoutedTool(_cfg("Bash"), cwd="/explicit")._cwd == "/explicit"


def test_init_missing_tool_name_raises():
    with pytest.raises(ValueError, match="tool_name"):
        RealHookRoutedTool({"type": "native"})


_REQUIRES_LIVE_TUNNEL = pytest.mark.skipif(
    os.environ.get("AIOPS_TUNNEL_TEST") != "1",
    reason=(
        "needs the live reverse SSH tunnel (GPU box's localhost:2222 -> the Mac's sshd); "
        "run on the GPU training box with AIOPS_TUNNEL_TEST=1 (e.g. via `ssh aiops-gpu`). "
        "On the Mac itself localhost:2222 is colima's ssh mux, NOT this tunnel."
    ),
)


@_REQUIRES_LIVE_TUNNEL
def test_live_tunnel_executes_on_the_environment_host():
    stdout, stderr, exit_code = ssh_tunnel_bash_executor(
        "echo TUNNEL-OK && uname -s && docker ps --format '{{.Names}}'", timeout_s=60.0
    )
    assert exit_code == 0, stderr
    assert "TUNNEL-OK" in stdout
    assert "Darwin" in stdout
    assert stdout.strip()


@_REQUIRES_LIVE_TUNNEL
def test_live_tunnel_cwd_prefix_executes_in_the_given_directory():
    stdout, stderr, exit_code = ssh_tunnel_bash_executor("pwd", cwd="/tmp", timeout_s=60.0)
    assert exit_code == 0, stderr
    assert stdout.strip() == "/tmp"


@_REQUIRES_LIVE_TUNNEL
def test_live_tunnel_mixed_quotes_arrive_byte_exact():
    command = """echo "it's a 'mixed' \\"quote\\"" 'and $LITERAL stays'"""
    stdout, stderr, exit_code = ssh_tunnel_bash_executor(command, timeout_s=60.0)
    assert exit_code == 0, stderr
    assert stdout == 'it\'s a \'mixed\' "quote" and $LITERAL stays\n'


@_REQUIRES_LIVE_TUNNEL
def test_live_mcp_executor_rag_end_to_end_through_tunnel():
    result = asyncio.run(ssh_tunnel_mcp_executor(RAG_TOOL_NAME, {"query": "kafka 消费积压"}))
    assert result.get("is_error") is not True, result
    text = result["content"][0]["text"]
    assert "历史相似诊断工单" in text
