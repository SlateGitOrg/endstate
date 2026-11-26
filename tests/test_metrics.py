import pytest

from endstate.metrics import (
    analytic_pass_at_k,
    analytic_pass_hat_k,
    binomial_interval,
    bootstrap_ci,
    pass_at_k,
    pass_hat_k,
    rouge_l,
)


def test_estimator_known_values_n4_c2():
    assert pass_hat_k([2], 4, 1) == pytest.approx(0.5)
    assert pass_hat_k([2], 4, 2) == pytest.approx(1 / 6)
    assert pass_at_k([2], 4, 2) == pytest.approx(5 / 6)


def test_estimator_means_over_tasks_and_edge_counts():
    counts = [0, 4, 2, 4]
    assert pass_hat_k(counts, 4, 4) == pytest.approx(0.5)  # only the all-pass tasks count
    assert pass_hat_k(counts, 4, 1) == pytest.approx((0 + 1 + 0.5 + 1) / 4)
    assert pass_at_k(counts, 4, 1) == pytest.approx(pass_hat_k(counts, 4, 1))
    assert pass_at_k([0], 4, 4) == 0.0
    with pytest.raises(ValueError):
        pass_hat_k([2], 4, 5)


def test_pass_hat_is_nonincreasing_in_k_and_pass_at_nondecreasing():
    counts = [8, 7, 5, 3, 0, 6]
    hats = [pass_hat_k(counts, 8, k) for k in range(1, 9)]
    ats = [pass_at_k(counts, 8, k) for k in range(1, 9)]
    assert hats == sorted(hats, reverse=True)
    assert ats == sorted(ats)


def test_estimator_is_unbiased_for_analytic_value():
    # Exact expectation of C(c,k)/C(n,k) with c ~ Binomial(n, p) equals p**k.
    from math import comb

    n, p, k = 8, 0.7, 3
    expected = sum(
        comb(n, c) * p**c * (1 - p) ** (n - c) * pass_hat_k([c], n, k) for c in range(n + 1)
    )
    assert expected == pytest.approx(p**k)
    assert analytic_pass_hat_k([p], k) == pytest.approx(p**k)
    assert analytic_pass_at_k([p], 2) == pytest.approx(1 - (1 - p) ** 2)


def test_bootstrap_interval_brackets_estimate_and_is_reproducible():
    counts = [8] * 20 + [4] * 20 + [0] * 20
    lo, hi = bootstrap_ci(counts, 8, 1, seed=3)
    assert lo < pass_hat_k(counts, 8, 1) < hi
    assert (lo, hi) == bootstrap_ci(counts, 8, 1, seed=3)
    assert bootstrap_ci([8] * 10, 8, 8) == (1.0, 1.0)


def test_binomial_interval_values():
    lo, hi = binomial_interval(80, 0.15)
    assert lo < 12 < hi and hi - lo < 25
    assert binomial_interval(10, 0.0) == (0, 0)
    lo, hi = binomial_interval(400, 0.1)
    assert (lo, hi) == (25, 56)


def test_rouge_l_known_values():
    assert rouge_l("the cat sat", "the cat sat") == 1.0
    assert rouge_l("a b c d", "x y z") == 0.0
    # LCS("a b c d", "a c d e") = 3 -> P = 3/4, R = 3/4
    assert rouge_l("a b c d", "a c d e") == pytest.approx(0.75)
    assert rouge_l("", "a") == 0.0
