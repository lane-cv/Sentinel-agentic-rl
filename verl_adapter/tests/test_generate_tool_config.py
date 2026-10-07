from __future__ import annotations

from pathlib import Path

import yaml

from verl_adapter.generate_tool_config import DEFAULT_OUT_PATH, build_tool_config_document

REPO_TOOL_CONFIG_PATH = Path(__file__).resolve().parents[1] / "tool_config.yaml"


def _assert_well_formed_document(document: dict) -> None:
    assert isinstance(document, dict)
    assert "tools" in document
    tools = document["tools"]
    assert isinstance(tools, list)
    assert len(tools) > 0
    for entry in tools:
        assert set(entry.keys()) == {"class_name", "config", "tool_schema"}
        assert entry["class_name"] == "verl_adapter.rollout_worker.RealHookRoutedTool"

        config = entry["config"]
        assert config["type"] == "native"
        assert isinstance(config["tool_name"], str) and config["tool_name"]

        schema = entry["tool_schema"]
        assert schema["type"] == "function"
        function = schema["function"]
        assert isinstance(function["name"], str) and function["name"]
        assert isinstance(function["description"], str) and function["description"]
        parameters = function["parameters"]
        assert parameters["type"] == "object"
        assert isinstance(parameters["properties"], dict)
        assert isinstance(parameters["required"], list)
        assert isinstance(function["strict"], bool)


def test_build_tool_config_document_default_includes_rag_and_is_well_formed():
    document = build_tool_config_document()
    _assert_well_formed_document(document)
    tool_names = [entry["config"]["tool_name"] for entry in document["tools"]]
    assert tool_names == ["Bash", "Read", "Grep", "Glob", "mcp__aiops_rag__search_past_incidents"]


def test_build_tool_config_document_no_rag_has_four_builtin_tools_only():
    document = build_tool_config_document(include_rag=False)
    _assert_well_formed_document(document)
    tool_names = [entry["config"]["tool_name"] for entry in document["tools"]]
    assert tool_names == ["Bash", "Read", "Grep", "Glob"]


def test_default_out_path_is_tool_config_yaml_next_to_this_module():
    assert DEFAULT_OUT_PATH.name == "tool_config.yaml"
    assert DEFAULT_OUT_PATH.parent == REPO_TOOL_CONFIG_PATH.parent


def test_checked_in_tool_config_yaml_exists_and_matches_generator_output():
    assert REPO_TOOL_CONFIG_PATH.is_file()
    with REPO_TOOL_CONFIG_PATH.open(encoding="utf-8") as f:
        on_disk = yaml.safe_load(f)
    _assert_well_formed_document(on_disk)
    assert on_disk == build_tool_config_document()
