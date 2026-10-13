from __future__ import annotations

import asyncio
import json
import os
import shlex
import subprocess
import uuid
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from reward.trajectory import Step, Trajectory
from verl_adapter import _repo_paths, observation_packer
from verl_adapter.tool_schema_bridge import RAG_TOOL_NAME


@dataclass
class SampledToolCall:
    tool_use_id: str
    tool_name: str
    tool_input: dict[str, Any]


@dataclass
class ToolResult:
    tool_use_id: str
    content: Any
    is_error: bool = False


def default_subprocess_bash_executor(
    command: str, cwd: Optional[str] = None, timeout_s: float = 30.0
) -> tuple[str, str, int]:
    proc = subprocess.run(command, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, proc.returncode


AIOPS_TUNNEL_PORT_ENV = "AIOPS_TUNNEL_PORT"
AIOPS_TUNNEL_USER_ENV = "AIOPS_TUNNEL_USER"
AIOPS_TUNNEL_HOST_ENV = "AIOPS_TUNNEL_HOST"
_TUNNEL_DEFAULT_PORT = "2222"
_TUNNEL_DEFAULT_USER = "changeme"
_TUNNEL_DEFAULT_HOST = "localhost"

AIOPS_MAC_PYTHON_ENV = "AIOPS_MAC_PYTHON"
AIOPS_MAC_REPO_ROOT_ENV = "AIOPS_MAC_REPO_ROOT"
AIOPS_MAC_BRIDGE_PATH_ENV = "AIOPS_MAC_BRIDGE_PATH"
AIOPS_RAG_SERVICE_PORT_ENV = "AIOPS_RAG_SERVICE_PORT"
_RAG_SERVICE_DEFAULT_PORT = "18765"
AIOPS_MAC_AGENT_DIR_ENV = "AIOPS_MAC_AGENT_DIR"
_MAC_DEFAULT_REPO_ROOT = "Desktop/Sentinel-agentic rl"
_MAC_DEFAULT_BRIDGE_PATH = "verl_adapter/rag_bridge.py"


def tunnel_ssh_argv() -> list[str]:
    port = os.environ.get(AIOPS_TUNNEL_PORT_ENV) or _TUNNEL_DEFAULT_PORT
    user = os.environ.get(AIOPS_TUNNEL_USER_ENV) or _TUNNEL_DEFAULT_USER
    host = os.environ.get(AIOPS_TUNNEL_HOST_ENV) or _TUNNEL_DEFAULT_HOST
    return ["ssh", "-p", port, "--", f"{user}@{host}"]


def ssh_tunnel_bash_executor(
    command: str, cwd: Optional[str] = None, timeout_s: float = 30.0
) -> tuple[str, str, int]:
    remote_command = f"cd {shlex.quote(cwd)} && {command}" if cwd else command
    argv = [*tunnel_ssh_argv(), remote_command]
    proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s)
    return proc.stdout, proc.stderr, proc.returncode


def _mcp_error_observation(text: str) -> dict[str, Any]:
    return {"is_error": True, "content": [{"type": "text", "text": text}]}


async def ssh_tunnel_mcp_executor(
    tool_name: str,
    tool_input: dict[str, Any],
    timeout_s: float = 120.0,
) -> dict[str, Any]:
    if tool_name != RAG_TOOL_NAME:
        return _mcp_error_observation(
            f"ssh_tunnel_mcp_executor: no bridge wired for {tool_name!r} (only {RAG_TOOL_NAME})"
        )

    args_json = json.dumps(tool_input or {}, ensure_ascii=False)
    port = os.environ.get(AIOPS_RAG_SERVICE_PORT_ENV) or _RAG_SERVICE_DEFAULT_PORT

    curl_command = (
        f"curl -s -m {int(timeout_s)} -X POST http://localhost:{port}/search"
        f" -H {shlex.quote('Content-Type: application/json')} -d {shlex.quote(args_json)}"
    )
    result = await _ssh_run_remote_json(curl_command, timeout_s=timeout_s, what="rag_service curl")
    if result is not None:
        return result

    repo_root = os.environ.get(AIOPS_MAC_REPO_ROOT_ENV) or _MAC_DEFAULT_REPO_ROOT
    python = os.environ.get(AIOPS_MAC_PYTHON_ENV) or os.path.join(repo_root, "Sentinel-agent/.venv/bin/python")
    bridge_rel = os.environ.get(AIOPS_MAC_BRIDGE_PATH_ENV) or _MAC_DEFAULT_BRIDGE_PATH
    bridge_abs = os.path.join(repo_root, bridge_rel)
    bridge_command = f"{shlex.quote(python)} {shlex.quote(bridge_abs)} {shlex.quote(args_json)}"
    result = await _ssh_run_remote_json(bridge_command, timeout_s=timeout_s, what="rag_bridge fallback")
    if result is not None:
        return result
    return _mcp_error_observation(
        "rag_service unreachable AND rag_bridge fallback failed "
        f"(see stderr details in the two preceding observations' logs; tunnel port/host: "
        f"{os.environ.get(AIOPS_TUNNEL_PORT_ENV) or _TUNNEL_DEFAULT_PORT})"
    )


async def _ssh_run_remote_json(
    remote_command: str, timeout_s: float, what: str
) -> Optional[dict[str, Any]]:
    argv = [*tunnel_ssh_argv(), remote_command]
    proc = await asyncio.create_subprocess_exec(
        *argv, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        print(f"[ssh_tunnel_mcp_executor] {what}: timed out after {timeout_s}s", flush=True)
        return None
    stdout, stderr = out_b.decode("utf-8", errors="replace").strip(), err_b.decode("utf-8", errors="replace").strip()
    if proc.returncode != 0:
        print(f"[ssh_tunnel_mcp_executor] {what}: exited {proc.returncode}: {stderr[:200]}", flush=True)
        return None
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError:
        return _mcp_error_observation(f"{what}: stdout is not JSON: {stdout[:200]!r}")
    if not isinstance(parsed, dict) or "content" not in parsed:
        return _mcp_error_observation(f"{what}: result has unexpected shape: {str(parsed)[:200]!r}")
    return parsed


BashExecutor = Callable[[str], tuple[str, str, int]]


def _run_executor(
    bash_executor: BashExecutor, command: str, cwd: Optional[str] = None
) -> tuple[str, str, int]:
    if cwd:
        return bash_executor(command, cwd=cwd)
    return bash_executor(command)

McpExecutor = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


async def route_through_real_hook(tool_call: SampledToolCall) -> dict[str, Any]:
    remediation, guard_diagnose_ops = _repo_paths.load_real_hook_modules()
    input_data = {
        "tool_name": tool_call.tool_name,
        "tool_input": tool_call.tool_input,
        "hook_event_name": "PreToolUse",
    }
    result = await guard_diagnose_ops(input_data, tool_call.tool_use_id, {})
    hook_output = result.get("hookSpecificOutput") or {}
    decision_raw = hook_output.get("permissionDecision")
    decision = decision_raw if decision_raw in ("allow", "deny") else "passthrough"
    action = target = None
    if tool_call.tool_name == "Bash":
        verdict = remediation.classify_command(str(tool_call.tool_input.get("command", "")))
        if verdict.get("decision") in ("allow", "deny"):
            action, target = verdict.get("action"), verdict.get("target")
    return {
        "decision": decision,
        "action": action,
        "target": target,
        "reason": hook_output.get("permissionDecisionReason"),
    }


def _read_remote_command(tool_input: dict[str, Any]) -> str:
    file_path = str(tool_input.get("file_path", ""))
    return f"cat {shlex.quote(file_path)}"


def _grep_remote_command(tool_input: dict[str, Any]) -> str:
    pattern = str(tool_input.get("pattern", ""))
    path = tool_input.get("path")
    target = shlex.quote(str(path)) if path else "."
    return f"grep -rE -- {shlex.quote(pattern)} {target}"


def _glob_remote_command(tool_input: dict[str, Any]) -> str:
    pattern = str(tool_input.get("pattern", ""))
    normalized = pattern[2:] if pattern.startswith("./") else pattern
    return f"find . -path {shlex.quote('./' + normalized)} -print"


_READ_GREP_GLOB_COMMAND_BUILDERS: dict[str, Callable[[dict[str, Any]], str]] = {
    "Read": _read_remote_command,
    "Grep": _grep_remote_command,
    "Glob": _glob_remote_command,
}


async def execute_tool_call(
    tool_call: SampledToolCall,
    bash_executor: BashExecutor = default_subprocess_bash_executor,
    mcp_executor: Optional[McpExecutor] = None,
    cwd: Optional[str] = None,
) -> tuple[ToolResult, dict[str, Any]]:
    verdict = await route_through_real_hook(tool_call)

    if tool_call.tool_name == "Bash" and verdict["decision"] == "deny":
        content = observation_packer.pack_hook_denied_result(verdict["reason"] or "denied by PreToolUse hook")
        return ToolResult(tool_call.tool_use_id, content, is_error=True), verdict

    if tool_call.tool_name == "Bash":
        stdout, stderr, exit_code = _run_executor(
            bash_executor, str(tool_call.tool_input.get("command", "")), cwd=cwd
        )
        content = observation_packer.pack_bash_exec_result(stdout, stderr, exit_code)
        return ToolResult(tool_call.tool_use_id, content, is_error=exit_code != 0), verdict

    if tool_call.tool_name.startswith("mcp__"):
        if mcp_executor is None:
            raise RuntimeError(f"execute_tool_call: no mcp_executor configured for {tool_call.tool_name!r}")
        raw = await mcp_executor(tool_call.tool_name, tool_call.tool_input)
        content = observation_packer.pack_mcp_result(raw)
        return ToolResult(tool_call.tool_use_id, content, is_error=bool((raw or {}).get("is_error"))), verdict

    command_builder = _READ_GREP_GLOB_COMMAND_BUILDERS.get(tool_call.tool_name)
    if command_builder is not None:
        stdout, stderr, exit_code = _run_executor(bash_executor, command_builder(tool_call.tool_input), cwd=cwd)
        content = observation_packer.pack_bash_exec_result(stdout, stderr, exit_code)
        return ToolResult(tool_call.tool_use_id, content, is_error=exit_code != 0), verdict

    raise NotImplementedError(
        f"execute_tool_call: no real executor wired for tool_name={tool_call.tool_name!r} "
        "(supported: Bash, mcp__*, Read, Grep, Glob)."
    )


class StepAccumulator:

    def __init__(self) -> None:
        self._raw_steps: list[dict[str, Any]] = []

    def record(self, tool_call: SampledToolCall, result: ToolResult, hook_verdict: dict[str, Any]) -> None:
        self._raw_steps.append(
            {
                "tool_name": tool_call.tool_name,
                "tool_input": tool_call.tool_input,
                "hook_decision": hook_verdict["decision"],
                "hook_action": hook_verdict["action"],
                "hook_target": hook_verdict["target"],
                "observation": observation_packer.stringify_tool_result_content(result.content),
            }
        )

    def build_trajectory(
        self,
        alert: dict[str, Any],
        final_diagnosis: dict[str, Any],
        ground_truth: dict[str, Any],
        route: str,
        post_action_health: Optional[dict[str, Any]] = None,
    ) -> Trajectory:
        signals_mod = _repo_paths.load_signals_module()
        signals_mod.annotate_steps(self._raw_steps)
        steps = [
            Step(
                tool_name=s["tool_name"],
                tool_input=s["tool_input"],
                hook_decision=s["hook_decision"],
                observation=s["observation"],
                hook_action=s["hook_action"],
                hook_target=s["hook_target"],
                signal_types_covered=s["signal_types_covered"],
                is_repeat_no_new_info=s["is_repeat_no_new_info"],
            )
            for s in self._raw_steps
        ]
        return Trajectory(
            alert=alert,
            steps=steps,
            final_diagnosis=final_diagnosis,
            ground_truth=ground_truth,
            route=route,
            post_action_health=post_action_health,
        )


ToolCallSource = Callable[[list[dict[str, Any]]], Awaitable[Optional[SampledToolCall]]]


async def run_real_rollout(
    alert: dict[str, Any],
    next_tool_call: ToolCallSource,
    final_diagnosis_reader: Callable[[], Awaitable[dict[str, Any]]],
    ground_truth: dict[str, Any],
    route: str,
    post_action_health_reader: Optional[Callable[[], Awaitable[Optional[dict[str, Any]]]]] = None,
    bash_executor: BashExecutor = default_subprocess_bash_executor,
    mcp_executor: Optional[McpExecutor] = None,
    cwd: Optional[str] = None,
    max_turns: int = 12,
) -> Trajectory:
    accumulator = StepAccumulator()
    history: list[dict[str, Any]] = []
    for _ in range(max_turns):
        tool_call = await next_tool_call(history)
        if tool_call is None:
            break
        result, verdict = await execute_tool_call(
            tool_call, bash_executor=bash_executor, mcp_executor=mcp_executor, cwd=cwd
        )
        accumulator.record(tool_call, result, verdict)
        history.append({"tool_call": tool_call, "result": result})

    final_diagnosis = await final_diagnosis_reader()
    post_action_health = await post_action_health_reader() if post_action_health_reader is not None else None

    return accumulator.build_trajectory(alert, final_diagnosis, ground_truth, route, post_action_health)


class ToolResponseShim:

    __slots__ = ("text", "image", "video")

    def __init__(
        self,
        text: Optional[str] = None,
        image: Optional[list[Any]] = None,
        video: Optional[list[Any]] = None,
    ) -> None:
        self.text = text
        self.image = image
        self.video = video

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, ToolResponseShim):
            return (self.text, self.image, self.video) == (other.text, other.image, other.video)
        return NotImplemented

    def __repr__(self) -> str:
        return f"ToolResponseShim(text={self.text!r}, image={self.image!r}, video={self.video!r})"


class RealHookRoutedTool:

    def __init__(
        self,
        config: dict[str, Any],
        tool_schema: Any = None,
        *,
        bash_executor: BashExecutor = ssh_tunnel_bash_executor,
        mcp_executor: Optional[McpExecutor] = ssh_tunnel_mcp_executor,
        cwd: Optional[str] = None,
    ) -> None:
        tool_name = (config or {}).get("tool_name")
        if not tool_name:
            raise ValueError(
                f"RealHookRoutedTool: config dict must carry a 'tool_name' entry (got {config!r}) "
                "— this is how verl's tool_config.yaml passes the tool's name"
            )
        self.name = tool_name
        self.tool_schema = tool_schema
        self._bash_executor = bash_executor
        self._mcp_executor = mcp_executor
        self._cwd = cwd or (config or {}).get("cwd") or os.environ.get(AIOPS_MAC_AGENT_DIR_ENV) or None

    async def create(
        self, instance_id: Optional[str] = None, **kwargs: Any
    ) -> tuple[str, ToolResponseShim]:
        resolved = instance_id if instance_id is not None else str(uuid.uuid4())
        return resolved, ToolResponseShim(text="")

    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        agent_data: Any = None,
        **kwargs: Any,
    ) -> tuple[ToolResponseShim, float, dict[str, Any]]:
        tool_call = SampledToolCall(tool_use_id=instance_id, tool_name=self.name, tool_input=parameters)
        result, verdict = await execute_tool_call(
            tool_call, bash_executor=self._bash_executor, mcp_executor=self._mcp_executor, cwd=self._cwd
        )
        text = observation_packer.stringify_tool_result_content(result.content)
        tool_response = ToolResponseShim(text=text)
        tool_metrics = {
            "hook_decision": verdict["decision"],
            "hook_action": verdict["action"],
            "hook_target": verdict["target"],
        }
        return tool_response, 0.0, tool_metrics

    async def calc_reward(self, instance_id: str, **kwargs: Any) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs: Any) -> None:
        return None
