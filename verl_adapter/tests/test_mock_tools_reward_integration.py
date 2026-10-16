from __future__ import annotations

import pytest

from reward.credit_assignment import compute_credit_assignment_rewards
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Trajectory
from verl_adapter import mock_tools


@pytest.fixture(scope="module")
def mock_trajectories() -> list[Trajectory]:
    trajs = mock_tools.load_all_mock_trajectories()
    assert len(trajs) == 5
    return trajs


def _by_alertname(trajs: list[Trajectory], alertname: str) -> Trajectory:
    return next(t for t in trajs if t.alert.get("alertname") == alertname)


def test_load_all_mock_trajectories_are_valid_trajectory_instances(mock_trajectories):
    for traj in mock_trajectories:
        assert isinstance(traj, Trajectory)
        assert len(traj.steps) >= 1
        assert all(step.hook_decision in ("allow", "deny", "passthrough") for step in traj.steps)


def test_step_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        rewards = compute_step_rewards(traj)
        assert len(rewards) == len(traj.steps)
        assert all(isinstance(r, float) for r in rewards)


def test_outcome_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        out = compute_outcome_rewards(traj)
        assert "total" in out
        assert isinstance(out["total"], float)


def test_credit_assignment_rewards_compute_without_error(mock_trajectories):
    for traj in mock_trajectories:
        rewards = compute_credit_assignment_rewards(traj)
        assert len(rewards) == len(traj.steps)
        assert all(isinstance(r, float) for r in rewards)


def test_dep_clean_scenario_hook_decision_is_a_real_deny(mock_trajectories):
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.steps[-1].hook_decision == "deny"
    assert dep.steps[-1].hook_action == "restart_instance"
    assert dep.steps[-1].hook_target == "product-catalog"


def test_hookdeny_res_scenario_has_one_deny_and_one_allow(mock_trajectories):
    res = _by_alertname(mock_trajectories, "MockAdCpuThrottled")
    decisions = [s.hook_decision for s in res.steps]
    assert decisions.count("deny") == 1
    assert decisions.count("allow") == 1
    allow_step = next(s for s in res.steps if s.hook_decision == "allow")
    assert allow_step.hook_action == "scale_resources"
    assert allow_step.hook_target == "ad"


def test_evidence_fabrication_case_is_penalized(mock_trajectories):
    fail_evidence = _by_alertname(mock_trajectories, "MockFabricatedEvidence")
    rewards = compute_step_rewards(fail_evidence)
    assert rewards[-1] < 0


def test_ground_truth_is_populated_from_cold_start_fixtures(mock_trajectories):
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.ground_truth.get("route") == "feishu_online_op"
    assert dep.ground_truth.get("suspect_service") == "product-catalog"
    out = compute_outcome_rewards(dep)
    assert out["route_match_reward"] > 0


def test_post_action_health_translated_to_required_schema(mock_trajectories):
    dep = _by_alertname(mock_trajectories, "MockHighErrorRate")
    assert dep.post_action_health == {"healthy_at_60s": True, "new_alert_triggered": False}
    stop_loss = compute_outcome_rewards(dep)["stop_loss_reward"]
    assert stop_loss == 0.5
