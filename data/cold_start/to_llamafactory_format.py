from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
AIOPS_AGENT_DIR = REPO_ROOT / "Sentinel-agent"

DEFAULT_IN_PATH = HERE / "prefix_subsamples.mock.json"
DEFAULT_OUT_JSONL = HERE / "aiops_cold_start_sft.mock.jsonl"
DEFAULT_DATASET_INFO_SNIPPET = HERE / "dataset_info_snippet.json"
DATASET_NAME = "aiops_cold_start_prefix_split"

_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "Bash",
        "description": "执行一条只读观测/低风险处置 Bash 命令（受 PreToolUse hook 白名单硬约束）。",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "要执行的 shell 命令"}},
            "required": ["command"],
        },
    },
    {
        "name": "Read",
        "description": "读取一个文件的内容（如 deploys.log）。",
        "parameters": {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        },
    },
    {
        "name": "Grep",
        "description": "在文件/目录里按正则搜索内容。",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
            "required": ["pattern"],
        },
    },
    {
        "name": "Glob",
        "description": "按 glob 模式匹配文件路径。",
        "parameters": {
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
    },
]
TOOLS_JSON_STR = json.dumps(_TOOLS_SCHEMA, ensure_ascii=False)


def _ensure_aiops_agent_on_path() -> None:
    p = str(AIOPS_AGENT_DIR)
    if AIOPS_AGENT_DIR.is_dir() and p not in sys.path:
        sys.path.insert(0, p)


def _load_prompt_fns():
    try:
        _ensure_aiops_agent_on_path()
        from agent.agents import prompts

        return prompts.diagnose_prompt, prompts.diagnose_append
    except Exception:

        def _fallback_prompt(alert: dict[str, Any]) -> str:
            return f"请诊断以下告警：\n```json\n{json.dumps(alert, ensure_ascii=False, indent=2)}\n```\n"

        def _fallback_append() -> str:
            return "你是 AIOps 故障诊断处置 Agent，只读排查后输出结构化 Diagnosis。"

        return _fallback_prompt, _fallback_append


_diagnose_prompt, _diagnose_append = _load_prompt_fns()
SYSTEM_PROMPT = _diagnose_append()


def _render_prior_steps_block(prefix_steps: list[dict[str, Any]]) -> str:
    if not prefix_steps:
        return "（尚未执行任何工具调用。）"
    lines = ["已执行步骤："]
    for idx, step in enumerate(prefix_steps, start=1):
        hook_note = ""
        if step.get("hook_decision") == "deny":
            hook_note = "  [被 hook 拦截：未真正执行]"
        elif step.get("hook_decision") == "allow":
            hook_note = f"  [hook 已放行自动处置：{step.get('hook_action')}→{step.get('hook_target')}]"
        lines.append(
            f"{idx}. 工具: {step.get('tool_name')}  参数: {json.dumps(step.get('tool_input', {}), ensure_ascii=False)}{hook_note}\n"
            f"   观测: {step.get('observation', '')}"
        )
    return "\n".join(lines)


def _render_human(record: dict[str, Any]) -> str:
    base = _diagnose_prompt(record.get("alert") or {})
    prior = _render_prior_steps_block(record.get("prefix_steps") or [])
    if record["target_type"] == "tool_call":
        ask = "请给出下一步要调用的工具（只输出一个 JSON 对象：{\"tool_name\": ..., \"tool_input\": {...}}，不要多余文字）。"
    else:
        ask = "现有观测是否已经足够收敛？请输出最终结构化 Diagnosis JSON（严格符合 Diagnosis schema）。"
    return f"{base}\n{prior}\n\n{ask}"


def _render_target_turn(record: dict[str, Any]) -> dict[str, str]:
    if record["target_type"] == "tool_call":
        target = record["target"]
        value = json.dumps({"name": target["tool_name"], "arguments": target["tool_input"]}, ensure_ascii=False)
        return {"from": "function_call", "value": value}
    value = json.dumps(record["target"], ensure_ascii=False)
    return {"from": "gpt", "value": value}


def convert_record(record: dict[str, Any]) -> dict[str, Any]:
    human_turn = {"from": "human", "value": _render_human(record)}
    target_turn = _render_target_turn(record)
    return {
        "conversations": [human_turn, target_turn],
        "system": SYSTEM_PROMPT,
        "tools": TOOLS_JSON_STR,
    }


def convert_all(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [convert_record(r) for r in records]


DATASET_INFO_SNIPPET: dict[str, Any] = {
    DATASET_NAME: {
        "file_name": DEFAULT_OUT_JSONL.name,
        "formatting": "sharegpt",
        "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
    }
}


def main() -> None:
    parser = argparse.ArgumentParser(description="把 prefix_split.py 的输出转成 LLaMA-Factory sharegpt jsonl")
    parser.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN_PATH)
    parser.add_argument("--out", dest="out_path", type=Path, default=DEFAULT_OUT_JSONL)
    parser.add_argument("--dataset-info-out", type=Path, default=DEFAULT_DATASET_INFO_SNIPPET)
    args = parser.parse_args()

    records = json.loads(args.in_path.read_text(encoding="utf-8"))
    converted = convert_all(records)

    with args.out_path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    snippet = dict(DATASET_INFO_SNIPPET)
    snippet[DATASET_NAME]["file_name"] = args.out_path.name
    args.dataset_info_out.write_text(json.dumps(snippet, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[to_llamafactory_format] {len(converted)} 条样本 -> {args.out_path}")
    print(f"[to_llamafactory_format] dataset_info.json 注册片段 -> {args.dataset_info_out}")
    print(f"[to_llamafactory_format] 把下面这段合并进 LLaMA-Factory 的 data/dataset_info.json：")
    print(json.dumps(snippet, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
