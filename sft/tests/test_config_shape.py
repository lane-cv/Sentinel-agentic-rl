from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "llamafactory_config" / "qwen3_5_9b_lora_sft.yaml"
)


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_yaml_parses():
    config = _load_config()
    assert isinstance(config, dict)
    assert config, "配置文件不应为空"


def test_lora_hyperparams_match_design_doc():
    config = _load_config()
    assert config["finetuning_type"] == "lora"
    assert config["lora_rank"] == 16
    assert config["lora_alpha"] == 32

    target_modules = {m.strip() for m in config["lora_target"].split(",")}
    expected_modules = {
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "out_proj",
    }
    assert target_modules == expected_modules


def test_precision_and_memory_optimizations_match_design_doc():
    config = _load_config()
    assert config["bf16"] is True
    assert config["disable_gradient_checkpointing"] is False
    assert config["flash_attn"] == "sdpa"


def test_dataset_matches_sibling_task_registration():
    config = _load_config()
    assert config["dataset"] == "aiops_cold_start_prefix_split"
    assert config["stage"] == "sft"
    assert config["do_train"] is True


def test_model_path_is_placeholder_for_text_only_checkpoint():
    config = _load_config()
    model_path = config["model_name_or_path"]
    assert isinstance(model_path, str) and model_path
    assert "text-only" in model_path or "text_only" in model_path


def test_dataset_info_snippet_registers_same_dataset_name():
    import json

    snippet_path = (
        CONFIG_PATH.resolve().parent.parent.parent
        / "data"
        / "cold_start"
        / "dataset_info_snippet_v9.json"
    )
    snippet = json.loads(snippet_path.read_text(encoding="utf-8"))

    config = _load_config()
    assert config["dataset"] in snippet
    assert snippet[config["dataset"]]["formatting"] == "sharegpt"
