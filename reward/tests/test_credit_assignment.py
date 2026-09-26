from __future__ import annotations

import pytest

from reward.anti_hacking import group_reward_variance
from reward.credit_assignment import (
    SIGNAL_WEIGHTS_BY_KIND,
    compute_credit_assignment_rewards,
    cumulative_signal_types,
    evidence_value_reward,
)
from reward.outcome_reward import compute_outcome_rewards
from reward.step_reward import REPEAT_NO_NEW_INFO_PENALTY
from reward.tests.fixtures import (
    _hybrid_wrong_route,
    hybrid_wrong_route_broad_coverage,
    hybrid_wrong_route_narrow_coverage,
)


def test_identical_outcome_but_credit_assignment_reward_differs():
    broad = hybrid_wrong_route_broad_coverage()
    narrow = hybrid_wrong_route_narrow_coverage()

    assert broad.route == narrow.route == "auto_remediated"
    outcome_broad = compute_outcome_rewards(broad)
    outcome_narrow = compute_outcome_rewards(narrow)
    assert outcome_broad["total"] == pytest.approx(outcome_narrow["total"])

    credit_broad = compute_credit_assignment_rewards(broad)
    credit_narrow = compute_credit_assignment_rewards(narrow)
    assert sum(credit_broad) != pytest.approx(sum(credit_narrow))

    variance = group_reward_variance([sum(credit_broad), sum(credit_narrow)])
    assert variance > 0.0


def test_evidence_value_reward_marginal_for_resource_kind():
    traj = hybrid_wrong_route_broad_coverage()
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    assert evidence_value_reward(traj, 0) == pytest.approx(weights["prometheus"])
    assert evidence_value_reward(traj, 1) == pytest.approx(weights["kubectl_status"])
    assert evidence_value_reward(traj, 2) == pytest.approx(weights["logs"])
    assert evidence_value_reward(traj, 3) == pytest.approx(weights["git_history"])


def test_cumulative_signal_types_is_monotonic():
    traj = hybrid_wrong_route_broad_coverage()
    covered_sets = [cumulative_signal_types(traj, i) for i in range(len(traj.steps))]
    for earlier, later in zip(covered_sets, covered_sets[1:]):
        assert earlier <= later


def test_narrow_coverage_credit_zero_after_first_step():
    traj = hybrid_wrong_route_narrow_coverage()
    rewards = compute_credit_assignment_rewards(traj)
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    assert rewards[0] == pytest.approx(weights["kubectl_status"])
    assert rewards[1] == rewards[2] == rewards[3] == 0.0


def test_twelve_step_repeat_exploit_no_longer_dominates():
    traj = _hybrid_wrong_route([["kubectl_status"]] * 12)
    for step in traj.steps[1:]:
        step.is_repeat_no_new_info = True

    credit = compute_credit_assignment_rewards(traj)
    w_kubectl = SIGNAL_WEIGHTS_BY_KIND["resource"]["kubectl_status"]

    assert sum(credit) == pytest.approx(w_kubectl)
    assert credit[0] == pytest.approx(w_kubectl)
    assert all(c == 0.0 for c in credit[1:])

    net = sum(credit) + REPEAT_NO_NEW_INFO_PENALTY * 11
    assert net < 0.0


def test_three_distinct_signals_each_pay_once():
    traj = _hybrid_wrong_route(
        [["prometheus"], ["logs"], ["git_history"]]
    )
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    credit = compute_credit_assignment_rewards(traj)
    assert credit[0] == pytest.approx(weights["prometheus"])
    assert credit[1] == pytest.approx(weights["logs"])
    assert credit[2] == pytest.approx(weights["git_history"])


def test_mixed_new_and_repeat_only_new_steps_paid():
    traj = _hybrid_wrong_route(
        [["prometheus"], ["prometheus"], ["logs"], ["logs"]]
    )
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    credit = compute_credit_assignment_rewards(traj)
    assert credit[0] == pytest.approx(weights["prometheus"])
    assert credit[1] == 0.0
    assert credit[2] == pytest.approx(weights["logs"])
    assert credit[3] == 0.0


def test_compute_credit_assignment_rewards_length_matches_steps():
    traj = hybrid_wrong_route_broad_coverage()
    assert len(compute_credit_assignment_rewards(traj)) == len(traj.steps)


def test_total_credit_equals_cumulative_weight_sum():
    traj = hybrid_wrong_route_broad_coverage()
    weights = SIGNAL_WEIGHTS_BY_KIND["resource"]
    final_covered = cumulative_signal_types(traj, len(traj.steps) - 1)
    expected = sum(weights[t] for t in final_covered)
    assert sum(compute_credit_assignment_rewards(traj)) == pytest.approx(expected)
