from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

_DEPLOY_DIR = Path(__file__).resolve().parent.parent
if str(_DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(_DEPLOY_DIR))

import merge_lora


def test_parser_requires_all_three_paths():
    parser = merge_lora._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_parser_defaults_match_design_doc():
    parser = merge_lora._build_parser()
    args = parser.parse_args(
        [
            "--base-checkpoint",
            "/data/models/qwen3.5-9b-text-only",
            "--adapter-dir",
            "/data/saves/lora",
            "--output-dir",
            "/data/models/merged",
        ]
    )
    assert args.dtype == "bf16"
    assert args.attn_implementation == "flash_attention_2"
    assert args.trust_remote_code is True
    assert args.base_checkpoint == "/data/models/qwen3.5-9b-text-only"
    assert args.adapter_dir == Path("/data/saves/lora")
    assert args.output_dir == Path("/data/models/merged")


def test_parser_rejects_unsupported_dtype():
    parser = merge_lora._build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--base-checkpoint",
                "x",
                "--adapter-dir",
                "y",
                "--output-dir",
                "z",
                "--dtype",
                "int8",
            ]
        )


def _make_valid_adapter_dir(tmp_path: Path) -> Path:
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    (adapter_dir / "adapter_config.json").write_text("{}", encoding="utf-8")
    return adapter_dir


def test_validate_paths_rejects_missing_adapter_dir(tmp_path):
    with pytest.raises(SystemExit, match="adapter-dir 指向的目录不存在"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=tmp_path / "does_not_exist",
            output_dir=tmp_path / "out",
        )


def test_validate_paths_rejects_adapter_dir_without_config(tmp_path):
    adapter_dir = tmp_path / "lora_adapter"
    adapter_dir.mkdir()
    with pytest.raises(SystemExit, match="没有 adapter_config.json"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "out",
        )


def test_validate_paths_accepts_hub_id_base_checkpoint_without_requiring_it_on_disk(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    merge_lora._validate_paths(
        base_checkpoint="Qwen/Qwen3.5-9B",
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "brand_new_output_dir",
    )


def test_validate_paths_rejects_missing_local_base_checkpoint(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    missing_local_path = str(tmp_path / "no_such_checkpoint_dir")
    with pytest.raises(SystemExit, match="base-checkpoint 看起来是本地路径但不存在"):
        merge_lora._validate_paths(
            base_checkpoint=missing_local_path,
            adapter_dir=adapter_dir,
            output_dir=tmp_path / "out",
        )


def test_validate_paths_accepts_existing_local_base_checkpoint(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    base_checkpoint_dir = tmp_path / "text_only_base"
    base_checkpoint_dir.mkdir()
    merge_lora._validate_paths(
        base_checkpoint=str(base_checkpoint_dir),
        adapter_dir=adapter_dir,
        output_dir=tmp_path / "out",
    )


def test_validate_paths_rejects_nonempty_output_dir(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "leftover_file.bin").write_text("stale", encoding="utf-8")
    with pytest.raises(SystemExit, match="已存在且非空"):
        merge_lora._validate_paths(
            base_checkpoint="Qwen/Qwen3.5-9B",
            adapter_dir=adapter_dir,
            output_dir=output_dir,
        )


def test_validate_paths_accepts_empty_existing_output_dir(tmp_path):
    adapter_dir = _make_valid_adapter_dir(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    merge_lora._validate_paths(
        base_checkpoint="Qwen/Qwen3.5-9B",
        adapter_dir=adapter_dir,
        output_dir=output_dir,
    )


@pytest.fixture
def clean_sys_modules(monkeypatch):
    for name in ("transformers", "peft"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    yield monkeypatch


def test_missing_transformers_gives_actionable_error(clean_sys_modules):
    clean_sys_modules.setitem(sys.modules, "transformers", None)
    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert "pip install -U transformers" in msg


def test_transformers_without_causal_lm_class_gives_actionable_error(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.0.0-fake"
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)

    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert merge_lora.EXPECTED_CAUSAL_LM_CLASS in msg
    assert "4.0.0-fake" in msg


def test_transformers_ok_but_peft_missing_gives_actionable_error(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.99.0-fake"
    setattr(fake_transformers, merge_lora.EXPECTED_CAUSAL_LM_CLASS, object())
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)
    clean_sys_modules.setitem(sys.modules, "peft", None)

    with pytest.raises(SystemExit) as excinfo:
        merge_lora._check_merge_dependencies()
    msg = str(excinfo.value)
    assert "pip install -U peft" in msg


def test_all_dependencies_present_returns_modules_and_class(clean_sys_modules):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.__version__ = "4.99.0-fake"
    fake_causal_lm_cls = object()
    setattr(fake_transformers, merge_lora.EXPECTED_CAUSAL_LM_CLASS, fake_causal_lm_cls)
    clean_sys_modules.setitem(sys.modules, "transformers", fake_transformers)

    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = object()
    clean_sys_modules.setitem(sys.modules, "peft", fake_peft)

    transformers_mod, causal_lm_cls, peft_mod = merge_lora._check_merge_dependencies()
    assert transformers_mod is fake_transformers
    assert causal_lm_cls is fake_causal_lm_cls
    assert peft_mod is fake_peft
