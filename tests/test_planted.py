import pytest
from conftest import TRIALS, pass_counts

from endstate.agents import FailureRates
from endstate.agents.scripted import MODES, applicable_trials, success_probability, trial_draws
from endstate.metrics import analytic_pass_hat_k, binomial_interval, pass_hat_k


def test_per_task_success_probability_from_planted_rates(tasks):
    rates = FailureRates()
    by = {t.scenario: success_probability(t, rates) for t in tasks}
    assert by["name_correction"] == pytest.approx(1.0)
    assert by["refund_refundable"] == pytest.approx(0.90)
    assert by["refund_nonrefundable"] == pytest.approx(0.90 * 0.85)
    assert by["cancelled_rebook"] == pytest.approx(0.90 * 0.95 * (1 - 0.05 * 0.5))
    assert success_probability(tasks[0], FailureRates.none()) == 1.0


def test_draws_are_a_pure_function_of_seed_task_and_trial():
    assert list(trial_draws("T001", 3)) == list(trial_draws("T001", 3))
    assert list(trial_draws("T001", 3)) != list(trial_draws("T001", 4))
    assert list(trial_draws("T001", 3)) != list(trial_draws("T002", 3))
    assert list(trial_draws("T001", 3, seed=1)) != list(trial_draws("T001", 3))


def test_harness_recovers_analytic_pass_hat_k(tasks, planted_run):
    _, grades = planted_run
    counts = pass_counts(grades, tasks)
    probs = [success_probability(t, FailureRates()) for t in tasks]
    est1, ana1 = pass_hat_k(counts, TRIALS, 1), analytic_pass_hat_k(probs, 1)
    est8, ana8 = pass_hat_k(counts, TRIALS, 8), analytic_pass_hat_k(probs, 8)
    assert abs(est1 - ana1) <= 0.05
    assert abs(est8 - ana8) <= 0.08
    assert 0.80 < ana1 < 0.90 and 0.3 < ana8 < 0.45  # the analytic target itself


def test_each_failure_mode_count_is_inside_its_99_percent_interval(tasks, planted_run):
    _, grades = planted_run
    rates = FailureRates()
    trials = applicable_trials(tasks, TRIALS)
    assert trials == {
        "skip_verification": 50 * TRIALS,
        "cash_refund_nonrefundable": 10 * TRIALS,
        "wrong_cabin": 30 * TRIALS,
        "hallucinated_flight_id": 30 * TRIALS,
    }
    for mode in MODES:
        observed = sum(mode in g.fired for g in grades)
        lo, hi = binomial_interval(trials[mode], rates.rate(mode), 0.99)
        assert lo <= observed <= hi, (mode, observed, lo, hi)


def test_failures_are_classified_by_the_rule_that_caught_them(planted_run):
    _, grades = planted_run
    failed = [g for g in grades if not g.passed]
    labels = {g.label for g in failed}
    assert labels <= {
        "POLICY:identity_verified",
        "POLICY:nonrefundable_cash",
        "POLICY:same_cabin",
        "TOOL_ERROR_UNRECOVERED",
    }
    for g in failed:
        if "skip_verification" in g.fired:
            assert any(v["rule"] == "identity_verified" for v in g.violations)
        if "cash_refund_nonrefundable" in g.fired:
            assert any(v["rule"] == "nonrefundable_cash" for v in g.violations)
        if "wrong_cabin" in g.fired:
            assert any(v["rule"] == "same_cabin" for v in g.violations)


def test_a_trial_without_planted_failures_passes_and_only_recovered_hallucinations_pass_with_one(
    planted_run,
):
    _, grades = planted_run
    for g in grades:
        if not g.fired:
            assert g.passed, (g.task_id, g.trial, g.label)
        elif g.passed:
            assert g.fired == ["hallucinated_flight_id"]  # a recovered hallucination
        else:
            assert g.fired  # every failure traces back to a planted mode
