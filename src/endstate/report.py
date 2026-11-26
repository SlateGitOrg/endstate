"""Builds and renders the reliability report for one or more graded runs."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rich import box
from rich.console import Console
from rich.table import Table

from endstate.agents.scripted import (
    MODES,
    FailureRates,
    applicable_trials,
    success_probability,
)
from endstate.grader import Grade, read_grades
from endstate.metrics import (
    analytic_pass_at_k,
    analytic_pass_hat_k,
    binomial_interval,
    bootstrap_ci,
    pass_at_k,
    pass_hat_k,
)
from endstate.runner import read_meta
from endstate.tasks import REFUND_NONREFUNDABLE, Task

KS = (1, 2, 4, 8)


def build_report(run_dir: Path, tasks: list[Task]) -> dict[str, Any]:
    meta = read_meta(run_dir)
    grades = read_grades(run_dir)
    n = int(meta["trials"])
    by_task: dict[str, list[Grade]] = defaultdict(list)
    for g in grades:
        by_task[g.task_id].append(g)
    order = [t.id for t in tasks if t.id in by_task]
    counts = [sum(g.passed for g in by_task[tid]) for tid in order]
    task_by_id = {t.id: t for t in tasks}

    rates = FailureRates(**meta["rates"]) if meta.get("rates") else None
    probs = [success_probability(task_by_id[tid], rates) for tid in order] if rates else None

    rows = []
    for k in [k for k in KS if k <= n]:
        lo, hi = bootstrap_ci(counts, n, k)
        rows.append(
            {
                "k": k,
                "pass_hat": pass_hat_k(counts, n, k),
                "ci": (lo, hi),
                "analytic": analytic_pass_hat_k(probs, k) if probs else None,
                "pass_at": pass_at_k(counts, n, k),
                "analytic_at": analytic_pass_at_k(probs, k) if probs else None,
            }
        )

    taxonomy = Counter(g.label for g in grades if not g.passed)
    violations = Counter(v["rule"] for g in grades for v in g.violations)
    nonref = {t.id for t in tasks if t.scenario == REFUND_NONREFUNDABLE}
    rouge_all = sum(g.rouge_l for g in grades) / len(grades)
    rouge_nonref = [g.rouge_l for g in grades if g.task_id in nonref]
    planted = None
    if rates is not None and meta["agent"] == "scripted":
        planted = {}
        trials_by_mode = applicable_trials([task_by_id[tid] for tid in order], n)
        for mode in MODES:
            observed = sum(mode in g.fired for g in grades)
            expected = rates.rate(mode) * trials_by_mode[mode]
            lo, hi = binomial_interval(trials_by_mode[mode], rates.rate(mode))
            planted[mode] = {"observed": observed, "expected": expected, "interval": (lo, hi)}
    return {
        "agent": meta["agent"],
        "trials": n,
        "tasks": len(order),
        "rows": rows,
        "taxonomy": dict(taxonomy),
        "violations": dict(violations),
        "rouge_all": rouge_all,
        "rouge_nonref": sum(rouge_nonref) / len(rouge_nonref) if rouge_nonref else None,
        "planted": planted,
    }


def _fmt(x: float | None, digits: int = 3) -> str:
    return "-" if x is None else f"{x:.{digits}f}"


def render_report(reports: list[dict[str, Any]], console: Console) -> None:
    for rep in reports:
        console.print(
            f"\n[bold]{rep['agent']}[/bold]: {rep['tasks']} tasks x {rep['trials']} trials"
        )
        table = Table(box=box.SIMPLE_HEAD)
        for col in ("k", "pass^k", "95% CI", "analytic", "pass@k"):
            table.add_column(col, justify="right")
        for r in rep["rows"]:
            table.add_row(
                str(r["k"]),
                _fmt(r["pass_hat"]),
                f"{_fmt(r['ci'][0])}-{_fmt(r['ci'][1])}",
                _fmt(r["analytic"]),
                _fmt(r["pass_at"]),
            )
        console.print(table)

        console.print("Failure taxonomy (primary label per failed trial)")
        tax = Table(box=box.SIMPLE_HEAD)
        tax.add_column("label")
        tax.add_column("trials", justify="right")
        for label, n in sorted(rep["taxonomy"].items(), key=lambda kv: (-kv[1], kv[0])):
            tax.add_row(label, str(n))
        if not rep["taxonomy"]:
            tax.add_row("(none)", "0")
        console.print(tax)

        if rep["violations"]:
            console.print("Fare-rule violations (every violation)")
            viol = Table(box=box.SIMPLE_HEAD)
            viol.add_column("rule")
            viol.add_column("count", justify="right")
            for rule, n in sorted(rep["violations"].items(), key=lambda kv: (-kv[1], kv[0])):
                viol.add_row(rule, str(n))
            console.print(viol)

        if rep["planted"]:
            console.print("Planted failure modes fired (99% binomial interval)")
            pl = Table(box=box.SIMPLE_HEAD)
            for col in ("mode", "observed", "expected", "interval"):
                pl.add_column(col, justify="right" if col != "mode" else "left")
            for mode, v in rep["planted"].items():
                pl.add_row(
                    mode,
                    str(v["observed"]),
                    f"{v['expected']:.1f}",
                    f"{v['interval'][0]}-{v['interval'][1]}",
                )
            console.print(pl)

    contrast = Table(box=box.SIMPLE_HEAD)
    for col in ("agent", "pass^1", "ROUGE-L (all)", "ROUGE-L (non-refundable refunds)"):
        contrast.add_column(col, justify="left" if col == "agent" else "right")
    for rep in reports:
        contrast.add_row(
            rep["agent"],
            _fmt(rep["rows"][0]["pass_hat"]),
            _fmt(rep["rouge_all"]),
            _fmt(rep["rouge_nonref"]),
        )
    console.print("\nTranscript similarity vs end state")
    console.print(contrast)
