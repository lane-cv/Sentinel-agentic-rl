from __future__ import annotations

from claude_agent_sdk import create_sdk_mcp_server, tool

from agent import config
from agent.integrations import memory


def _format_hits(hits: list[dict]) -> str:
    if not hits:
        return "没有检索到相似的历史工单（可能是首次遇到此类问题，或历史库为空）。"
    lines = ["检索到以下历史相似诊断工单（按相似度降序）：\n"]
    for i, h in enumerate(hits, 1):
        lines.append(
            f"[{i}] 相似度={h.get('score')} | 服务={h.get('suspect_service')} | "
            f"类型={h.get('kind')} | 当时走向={h.get('route')}\n"
            f"    根因摘要: {h.get('summary')}\n"
            f"    处置: {h.get('remediation_detail')}"
        )
    lines.append(
        "\n注意：历史工单仅供参考，请结合当前实时观测证据独立判断，不要盲目照搬。"
    )
    return "\n".join(lines)


@tool(
    "search_past_incidents",
    "语义检索历史相似的诊断工单（根因/处置经验）。当你想参考过去是否处理过类似故障、"
    "或想借鉴历史根因与处置方式时调用。传入描述当前故障的自然语言查询（如告警症状、"
    "可疑服务、现象），返回最相似的若干历史工单摘要。",
    {"query": str, "top_k": int},
)
async def search_past_incidents(args: dict) -> dict:
    query = (args or {}).get("query", "")
    top_k = (args or {}).get("top_k") or config.RAG_TOP_K
    if not query:
        return {"content": [{"type": "text", "text": "查询为空，请提供描述当前故障的查询文本。"}],
                "is_error": True}
    hits = memory.search_tickets(query, top_k=top_k)
    return {"content": [{"type": "text", "text": _format_hits(hits)}]}


server = create_sdk_mcp_server(
    name="aiops_rag",
    version="1.0.0",
    tools=[search_past_incidents],
)

TOOL_NAME = "mcp__aiops_rag__search_past_incidents"
