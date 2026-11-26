"""Reliability metrics: pass^k, pass@k, bootstrap intervals, binomial intervals, ROUGE-L.

For a task with n trials of which c passed, the unbiased estimators are

    pass^k = C(c, k) / C(n, k)           P(k trials drawn without replacement all pass)
    pass@k = 1 - C(n - c, k) / C(n, k)   P(at least one of k drawn trials passes)

A benchmark's number is the mean over tasks.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from math import comb

import numpy as np


def pass_hat_task(n: int, c: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    return comb(c, k) / comb(n, k)


def pass_at_task(n: int, c: int, k: int) -> float:
    if not 1 <= k <= n:
        raise ValueError(f"k must be in 1..{n}, got {k}")
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(counts: Sequence[int], n: int, k: int) -> float:
    """pass^k: mean over tasks of C(c, k) / C(n, k)."""
    return float(np.mean([pass_hat_task(n, c, k) for c in counts]))


def pass_at_k(counts: Sequence[int], n: int, k: int) -> float:
    """pass@k: mean over tasks of 1 - C(n - c, k) / C(n, k)."""
    return float(np.mean([pass_at_task(n, c, k) for c in counts]))


def analytic_pass_hat_k(probs: Sequence[float], k: int) -> float:
    """Expected pass^k when task i passes each trial independently with probability p_i."""
    return float(np.mean([p**k for p in probs]))


def analytic_pass_at_k(probs: Sequence[float], k: int) -> float:
    return float(np.mean([1.0 - (1.0 - p) ** k for p in probs]))


def bootstrap_ci(
    counts: Sequence[int],
    n: int,
    k: int,
    kind: str = "hat",
    *,
    resamples: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval over tasks. ``kind`` is ``"hat"`` (pass^k) or ``"at"``."""
    task_value = pass_hat_task if kind == "hat" else pass_at_task
    per_task = np.array([task_value(n, c, k) for c in counts])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(per_task), size=(resamples, len(per_task)))
    means = per_task[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def binomial_interval(n: int, p: float, level: float = 0.99) -> tuple[int, int]:
    """Smallest central integer interval holding at least ``level`` of Binomial(n, p)."""
    pmf = np.array([comb(n, x) * p**x * (1 - p) ** (n - x) for x in range(n + 1)])
    cdf = np.cumsum(pmf)
    tail = (1 - level) / 2
    lo = int(np.searchsorted(cdf, tail, side="right"))
    hi = int(np.searchsorted(cdf, 1 - tail, side="left"))
    return lo, min(hi, n)


_WORD = re.compile(r"[a-z0-9$.']+")


def _tokens(text: str) -> list[str]:
    return [w.strip(".'") for w in _WORD.findall(text.lower()) if w.strip(".'")]


def _lcs(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b, start=1):
            cur.append(prev[j - 1] + 1 if x == y else max(prev[j], cur[-1]))
        prev = cur
    return prev[-1]


def rouge_l(candidate: str, reference: str) -> float:
    """ROUGE-L F1: longest common subsequence of word tokens."""
    cand, ref = _tokens(candidate), _tokens(reference)
    if not cand or not ref:
        return 0.0
    lcs = _lcs(cand, ref)
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(cand), lcs / len(ref)
    return 2 * precision * recall / (precision + recall)
