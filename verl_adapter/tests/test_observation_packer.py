from __future__ import annotations

from verl_adapter import observation_packer as pk


def test_stringify_tool_result_content_passthrough_str():
    assert pk.stringify_tool_result_content("plain text output") == "plain text output"


def test_stringify_tool_result_content_text_blocks_joined_with_newline():
    content = [{"type": "text", "text": "line one"}, {"type": "text", "text": "line two"}]
    assert pk.stringify_tool_result_content(content) == "line one\nline two"


def test_stringify_tool_result_content_non_text_block_json_dumped():
    content = [{"type": "image", "data": "abc"}]
    out = pk.stringify_tool_result_content(content)
    assert "image" in out and "abc" in out


def test_stringify_tool_result_content_none_is_empty_string():
    assert pk.stringify_tool_result_content(None) == ""


def test_stringify_tool_result_content_dict_json_dumped():
    out = pk.stringify_tool_result_content({"a": 1})
    assert out == '{"a": 1}'


def test_pack_mcp_result_matches_rag_tool_shape():
    mcp_result = {"content": [{"type": "text", "text": "检索到以下历史相似诊断工单"}]}
    assert pk.pack_mcp_result(mcp_result) == "检索到以下历史相似诊断工单"


def test_pack_mcp_result_none_is_empty_string():
    assert pk.pack_mcp_result(None) == ""


def test_pack_bash_exec_result_plain_stdout_no_wrapper():
    out = pk.pack_bash_exec_result("product-catalog   Up 2 hours (unhealthy)\n", exit_code=0)
    assert out == "product-catalog   Up 2 hours (unhealthy)"
    assert "exit_code" not in out


def test_pack_bash_exec_result_includes_stderr_when_present():
    out = pk.pack_bash_exec_result("", "command not found", exit_code=127)
    assert "command not found" in out
    assert "exit_code=127" in out


def test_pack_bash_exec_result_no_output_marker():
    out = pk.pack_bash_exec_result("", "", exit_code=0)
    assert out == "(no output; exit_code=0)"


def test_pack_hook_denied_result_returns_reason_verbatim():
    reason = "低风险操作 restart_instance 命中，但目标 product-catalog 不在可自动处置白名单内"
    assert pk.pack_hook_denied_result(reason) == reason
