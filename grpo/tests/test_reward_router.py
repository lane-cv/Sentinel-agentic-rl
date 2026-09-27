from __future__ import annotations

import copy

import pytest

from grpo.reward_router import (
    MAX_TURNS,
    MAX_TURNS_TRUNCATION_PENALTY,
    TURN_COUNT_PENALTY_PER_EXTRA_TURN,
    TURN_COUNT_PENALTY_THRESHOLD,
    compute_group_rewards,
    compute_per_step_rewards,
    compute_trajectory_reward,
    max_turns_truncation_penalty,
    route_trajectory_reward_to_variance_tracker,
    turn_count_penalty,
)
from grpo.tests.mock_trajectories import load_all_mock_trajectories, with_signal_coverage
from reward.anti_hacking import group_reward_variance
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import compute_step_rewards
from reward.trajectory import Step, Trajectory

ALL_MOCK = load_all_mock_trajectories()


def _clean_hybrid_leak_variant() -> Trajectory:
    traj = copy.deepcopy(ALL_MOCK["mock_hybrid_leak"])
    traj.final_diagnosis["evidence"] = [
        traj.steps[0].observation,
        traj.steps[3].observation,
    ]
    traj.ground_truth = {
        "suspect_service": traj.final_diagnosis["suspect_service"],
        "kind": traj.final_diagnosis["kind"],
        "remediation_type": traj.final_diagnosis["remediation_type"],
        "route": traj.route,
        "suspect_repo": traj.final_diagnosis["suspect_repo"],
        "suspect_commit_hint": traj.final_diagnosis["suspect_commit_hint"],
        "suspect_file_hint": traj.final_diagnosis["suspect_file_hint"],
    }
    return traj


class TestDeterminism:
    def test_single_trajectory_is_deterministic(self):
        traj = ALL_MOCK["mock_hybrid_leak"]
        first = compute_trajectory_reward(traj)
        second = compute_trajectory_reward(traj)
        assert first == second

    def test_group_is_deterministic(self):
        group = [ALL_MOCK[k] for k in sorted(ALL_MOCK)]
        first = compute_group_rewards(group)
        second = compute_group_rewards(group)
        assert first == second

    def test_all_mock_trajectories_route_without_error(self):
        for name, traj in ALL_MOCK.items():
            result = compute_trajectory_reward(traj)
            assert isinstance(result["total_reward"], float), name
            assert result["total_reward"] == pytest.approx(sum(result["per_step_rewards"])), name


class TestDirectionalSanity:

    def test_fabricated_evidence_lowers_total_reward_by_exactly_the_penalty(self):
        good = _clean_hybrid_leak_variant()
        bad = copy.deepcopy(good)
        bad.final_diagnosis["evidence"] = list(bad.final_diagnosis["evidence"]) + [
            "kubectl top 显示 recommendation 内存 9999MiB（此声明从未在任何工具调用中出现，纯编造）"
        ]

        good_reward = compute_trajectory_reward(good)["total_reward"]
        bad_reward = compute_trajectory_reward(bad)["total_reward"]

        assert bad_reward == pytest.approx(good_reward - 0.5)
        assert bad_reward < good_reward

    def test_denied_command_scores_lower_than_the_same_command_allowed(self):
        allowed = _clean_hybrid_leak_variant()
        denied = copy.deepcopy(allowed)
        denied.steps[-1].hook_decision = "deny"

        allowed_reward = compute_trajectory_reward(allowed)["total_reward"]
        denied_reward = compute_trajectory_reward(denied)["total_reward"]

        assert denied_reward == pytest.approx(allowed_reward - 0.25)
        assert denied_reward < allowed_reward

    def test_richer_observability_and_clean_evidence_beats_a_thin_mismatched_trajectory(self):
        clean_reward = compute_trajectory_reward(ALL_MOCK["mock_dep_clean"])["total_reward"]
        mismatch_reward = compute_trajectory_reward(ALL_MOCK["mock_fail_mismatch"])["total_reward"]
        assert clean_reward > mismatch_reward


class TestVariancePreservation:

    BASE = ALL_MOCK["mock_hybrid_leak"]
    ALL_FIVE_TYPES = ["git_history", "logs", "kubectl_status", "prometheus", "jaeger"]

    def test_identical_outcome_group_still_shows_nonzero_variance_after_routing(self):
        variants = []
        for k in range(6):
            touched = self.ALL_FIVE_TYPES[:k]
            coverage = [
                [touched[i]] if i < k else ([touched[-1]] if touched else [])
                for i in range(len(self.BASE.steps))
            ]
            variants.append(with_signal_coverage(self.BASE, coverage))

        outcome_totals = [compute_outcome_rewards(v)["total"] for v in variants]
        assert group_reward_variance(outcome_totals) == pytest.approx(0.0)

        group = compute_group_rewards(variants)
        assert group["group_size"] == 6
        assert group["group_reward_variance"] > 0.0
        for traj_reward in group["trajectory_rewards"]:
            assert traj_reward["total_reward"] == pytest.approx(sum(traj_reward["per_step_rewards"]))

    def test_credit_assignment_is_not_washed_out_by_step_and_outcome_alone(self):
        variants = []
        for k in range(6):
            touched = self.ALL_FIVE_TYPES[:k]
            coverage = [
                [touched[i]] if i < k else ([touched[-1]] if touched else [])
                for i in range(len(self.BASE.steps))
            ]
            variants.append(with_signal_coverage(self.BASE, coverage))

        step_and_outcome_only = [
            sum(compute_step_rewards(v)) + compute_outcome_rewards(v)["total"] for v in variants
        ]
        full_totals = compute_group_rewards(variants)["total_rewards"]

        variance_without_credit = group_reward_variance(step_and_outcome_only)
        variance_with_credit = group_reward_variance(full_totals)

        assert variance_without_credit > 0.0
        assert variance_with_credit > variance_without_credit

    def test_order_isolated_ablation_ties_step_and_outcome_exactly(self):
        front_loaded = with_signal_coverage(
            self.BASE, [["git_history"], ["logs"], ["kubectl_status"], ["prometheus"], ["jaeger"]]
        )
        back_loaded = with_signal_coverage(
            self.BASE, [["jaeger"], ["prometheus"], ["kubectl_status"], ["logs"], ["git_history"]]
        )

        assert compute_step_rewards(front_loaded) == compute_step_rewards(back_loaded)
        assert compute_outcome_rewards(front_loaded)["total"] == compute_outcome_rewards(back_loaded)["total"]

        front_total = compute_trajectory_reward(front_loaded)["total_reward"]
        back_total = compute_trajectory_reward(back_loaded)["total_reward"]
        assert front_total == pytest.approx(back_total)


class TestComposition:
    def test_per_step_rewards_is_elementwise_sum_of_step_and_credit(self):
        traj = ALL_MOCK["mock_hookdeny_res"]
        combined = compute_per_step_rewards(traj)
        step_only = compute_step_rewards(traj)
        from reward.credit_assignment import compute_credit_assignment_rewards

        credit_only = compute_credit_assignment_rewards(traj)
        assert combined == [pytest.approx(s + c) for s, c in zip(step_only, credit_only)]

    def test_outcome_reward_attaches_only_to_the_last_step(self):
        traj = ALL_MOCK["mock_dep_clean"]
        per_step_before_outcome = compute_per_step_rewards(traj)
        full = compute_trajectory_reward(traj)
        outcome_total = compute_outcome_rewards(traj)["total"]

        for i in range(len(traj.steps) - 1):
            assert full["per_step_rewards"][i] == pytest.approx(per_step_before_outcome[i])
        expected_last = (
            per_step_before_outcome[-1]
            + outcome_total
            + full["dual_source_coverage_penalty"]
            + full["turn_count_penalty"]
            + full["max_turns_truncation_penalty"]
        )
        assert full["per_step_rewards"][-1] == pytest.approx(expected_last)

    def test_zero_step_trajectory_gets_a_synthetic_terminal_reward(self):
        traj = Trajectory(
            alert={"service": "x"},
            steps=[],
            final_diagnosis={"remediation_type": "info_only", "evidence": []},
            ground_truth={},
            route="info_only",
        )
        result = compute_trajectory_reward(traj)
        assert len(result["per_step_rewards"]) == 1
        assert result["total_reward"] == pytest.approx(result["per_step_rewards"][0])


def _step(**overrides) -> Step:
    defaults = dict(
        tool_name="Bash",
        tool_input={"command": "true"},
        hook_decision="passthrough",
        observation="ok",
    )
    defaults.update(overrides)
    return Step(**defaults)


class TestTurnBudgetPenalties:
    def _trajectory_with_n_steps(self, n: int) -> Trajectory:
        return Trajectory(
            alert={"service": "x"},
            steps=[_step() for _ in range(n)],
            final_diagnosis={"remediation_type": "info_only", "evidence": []},
            ground_truth={},
            route="info_only",
        )

    def test_no_penalty_at_or_below_threshold(self):
        for n in (0, 1, TURN_COUNT_PENALTY_THRESHOLD):
            assert turn_count_penalty(self._trajectory_with_n_steps(n)) == 0.0

    def test_penalty_scales_linearly_beyond_threshold(self):
        n = TURN_COUNT_PENALTY_THRESHOLD + 3
        traj = self._trajectory_with_n_steps(n)
        assert turn_count_penalty(traj) == pytest.approx(3 * TURN_COUNT_PENALTY_PER_EXTRA_TURN)

    def test_no_truncation_penalty_at_or_below_max_turns(self):
        assert max_turns_truncation_penalty(self._trajectory_with_n_steps(MAX_TURNS)) == 0.0

    def test_truncation_penalty_applies_past_max_turns(self):
        traj = self._trajectory_with_n_steps(MAX_TURNS + 1)
        assert max_turns_truncation_penalty(traj) == MAX_TURNS_TRUNCATION_PENALTY

    def test_long_trajectory_total_reward_reflects_both_penalties(self):
        short = compute_trajectory_reward(self._trajectory_with_n_steps(TURN_COUNT_PENALTY_THRESHOLD))
        long_ = compute_trajectory_reward(self._trajectory_with_n_steps(MAX_TURNS + 2))
        assert long_["total_reward"] < short["total_reward"]


class TestGroupRewardsShape:
    def test_group_of_one_has_zero_variance(self):
        group = compute_group_rewards([ALL_MOCK["mock_dep_clean"]])
        assert group["group_size"] == 1
        assert group["group_reward_variance"] == 0.0

    def test_group_rewards_preserves_input_order(self):
        ordered = [ALL_MOCK["mock_dep_clean"], ALL_MOCK["mock_fail_mismatch"]]
        group = compute_group_rewards(ordered)
        assert group["total_rewards"][0] == pytest.approx(compute_trajectory_reward(ordered[0])["total_reward"])
        assert group["total_rewards"][1] == pytest.approx(compute_trajectory_reward(ordered[1])["total_reward"])


class TestVarianceTrackerHook:
    def test_does_not_mutate_input_history(self):
        history = {"dependency": [0.0, 0.0]}
        result = route_trajectory_reward_to_variance_tracker("dependency", [1.0, 1.0, 1.0], history)
        assert history == {"dependency": [0.0, 0.0]}
        assert result["updated_variance_history"]["dependency"] == [0.0, 0.0, 0.0]

    def test_flags_downsample_after_enough_consecutive_zero_variance_rounds(self):
        history: dict[str, list[float]] = {}
        result = None
        for _ in range(5):
            result = route_trajectory_reward_to_variance_tracker(
                "dependency", [0.5, 0.5, 0.5], history
            )
            history = result["updated_variance_history"]
        assert result["variance"] == 0.0
        assert result["should_downsample"] is True

    def test_does_not_flag_downsample_when_variance_is_real(self):
        history: dict[str, list[float]] = {}
        result = None
        for _ in range(5):
            result = route_trajectory_reward_to_variance_tracker(
                "hybrid_capstone", [0.1, 0.9, 1.5], history
            )
            history = result["updated_variance_history"]
        assert result["variance"] > 0.0
        assert result["should_downsample"] is False
