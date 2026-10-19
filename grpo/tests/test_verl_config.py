from __future__ import annotations

from pathlib import Path

import yaml

from grpo.reward_router import MAX_TURNS

CONFIG_PATH = Path(__file__).resolve().parents[1] / "verl_config" / "aiops_grpo.yaml"


def _load_config() -> dict:
    with CONFIG_PATH.open() as f:
        return yaml.safe_load(f)


def test_file_exists_and_parses_as_yaml():
    config = _load_config()
    assert isinstance(config, dict)
    assert config


def test_group_size_is_six():
    config = _load_config()
    assert config["actor_rollout_ref"]["rollout"]["n"] == 6


def test_max_turns_is_twelve_and_matches_reward_router():
    config = _load_config()
    multi_turn = config["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert multi_turn["max_assistant_turns"] == 12
    assert multi_turn["max_assistant_turns"] == MAX_TURNS


def test_kl_penalty_against_sft_policy_is_enabled():
    config = _load_config()
    actor = config["actor_rollout_ref"]["actor"]
    assert actor["use_kl_loss"] is True
    assert isinstance(actor["kl_loss_coef"], (int, float))
    assert actor["kl_loss_coef"] > 0


def test_adv_estimator_is_grpo():
    config = _load_config()
    assert config["algorithm"]["adv_estimator"] == "grpo"


def test_reward_function_points_at_verl_reward_adapter():
    config = _load_config()
    custom_fn = config["reward"]["custom_reward_function"]
    assert custom_fn["path"] == "grpo/verl_reward_adapter.py"
    assert custom_fn["name"] == "compute_score"


def test_adapter_group_size_matches_rollout_n():
    from grpo.verl_reward_adapter import DEFAULT_GROUP_SIZE

    config = _load_config()
    assert config["actor_rollout_ref"]["rollout"]["n"] == DEFAULT_GROUP_SIZE


def test_lora_hyperparameters_match_doc_5_1():
    config = _load_config()
    model = config["actor_rollout_ref"]["model"]
    assert model["lora_rank"] == 16
    assert model["lora_alpha"] == 32
