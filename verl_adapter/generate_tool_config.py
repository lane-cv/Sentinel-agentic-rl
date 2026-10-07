from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml

from verl_adapter.tool_schema_bridge import build_tool_config_entries

HERE = Path(__file__).resolve().parent
DEFAULT_OUT_PATH = HERE / "tool_config.yaml"


def build_tool_config_document(include_rag: bool = True) -> dict[str, Any]:
    return {"tools": build_tool_config_entries(include_rag=include_rag)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="生成 verl_adapter/tool_config.yaml（veRL 的 multi_turn.tool_config_path 指向的真实文件）"
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument(
        "--no-rag",
        dest="include_rag",
        action="store_false",
        default=True,
        help="不含 search_past_incidents RAG 工具（对应真实部署里 AIOPS_RAG_ENABLED=0 的情形）。默认包含，"
        "因为 Sentinel-agent/agent/config.py::RAG_ENABLED 默认就是开的。",
    )
    args = parser.parse_args()

    document = build_tool_config_document(include_rag=args.include_rag)

    with args.out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(document, f, allow_unicode=True, sort_keys=False)

    tool_names = [entry["config"]["tool_name"] for entry in document["tools"]]
    print(f"[generate_tool_config] {len(tool_names)} 条工具 -> {args.out}")
    print(f"[generate_tool_config] 工具: {tool_names}")


if __name__ == "__main__":
    main()
