"""Command line: seed, run, grade, report, replay."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console

from endstate.agents import AGENT_NAMES, FailureRates, OpenAICompatModel, make_agent
from endstate.env.db import Database, describe_diff
from endstate.generate import write_seed
from endstate.grader import GRADES_FILE, grade_all, grade_trajectory, write_grades
from endstate.report import build_report, render_report
from endstate.runner import (
    STEP_LIMIT,
    TOOL_BUDGET,
    read_meta,
    read_trajectories,
    run_all,
    write_run,
)
from endstate.tasks import Task, load_tasks

app = typer.Typer(
    help="Grade a travel-rebooking support agent by the booking database it leaves behind.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()

DataOpt = typer.Option(Path("data"), "--data", help="Directory written by `endstate seed`.")


def _load(data: Path) -> tuple[Database, list[Task]]:
    if not (data / "db.json").exists():
        raise typer.BadParameter(
            f"{data}/db.json not found; run `endstate seed --out {data}` first"
        )
    return Database.load(data / "db.json"), load_tasks(data / "tasks.yaml")


@app.command()
def seed(
    out: Path = typer.Option(Path("data"), help="Output directory."),
    seed: int = typer.Option(7, help="Generator seed."),
) -> None:
    """Generate the travel database (db.json) and the 60 tasks (tasks.yaml)."""
    db, tasks = write_seed(out, seed)
    nonref = sum(not b.refundable for b in db.bookings.values())
    console.print(
        f"wrote {out}/db.json and {out}/tasks.yaml: {len(db.airports)} airports, "
        f"{len(db.flights)} flights, {len(db.passengers)} passengers, {len(db.bookings)} bookings "
        f"({nonref} non-refundable), {len(tasks)} tasks"
    )


@app.command()
def run(
    agent: str = typer.Option("scripted", help=f"One of: {', '.join(AGENT_NAMES)}."),
    trials: int = typer.Option(8, min=1, help="Trials per task."),
    data: Path = DataOpt,
    out: Path | None = typer.Option(None, help="Run directory (default runs/<agent>)."),
    seed: int = typer.Option(0, help="Seed for the scripted agent's failure draws."),
    failures: bool = typer.Option(True, "--failures/--no-failures", help="Plant failure modes."),
) -> None:
    """Run every task `--trials` times and save trajectories as JSONL."""
    db, tasks = _load(data)
    rates = FailureRates() if failures else FailureRates.none()
    out = out or Path("runs") / agent

    model = OpenAICompatModel.from_env() if agent == "llm" else None

    def make(task: Task, trial: int):
        return make_agent(agent, task, trial, rates=rates, seed=seed, model=model)

    trajectories = run_all(tasks, db, make, trials)
    meta = {
        "agent": agent,
        "trials": trials,
        "seed": seed,
        "step_limit": STEP_LIMIT,
        "tool_budget": TOOL_BUDGET,
        "rates": rates.as_dict() if agent == "scripted" else None,
    }
    digest = write_run(out, trajectories, meta)
    console.print(f"wrote {len(trajectories)} trajectories to {out} (sha256 {digest[:16]})")


@app.command()
def grade(
    run_dir: Path = typer.Argument(..., help="Run directory written by `endstate run`."),
    data: Path = DataOpt,
) -> None:
    """Grade a run by end state and fare rules; writes grades.jsonl."""
    db, tasks = _load(data)
    grades = grade_all(tasks, read_trajectories(run_dir), db)
    write_grades(run_dir, grades)
    passed = sum(g.passed for g in grades)
    console.print(f"graded {len(grades)} trials in {run_dir}: {passed} passed")


@app.command()
def report(
    run_dirs: list[Path] = typer.Argument(None, help="Graded run directories (default runs/*)."),
    data: Path = DataOpt,
) -> None:
    """pass^k with intervals, failure taxonomy and the transcript-similarity contrast."""
    _, tasks = _load(data)
    dirs = run_dirs or sorted(p for p in Path("runs").glob("*") if (p / GRADES_FILE).exists())
    if not dirs:
        raise typer.BadParameter("no graded runs found; run `endstate grade <run_dir>` first")
    render_report([build_report(d, tasks) for d in dirs], console)


@app.command()
def replay(
    task_id: str = typer.Argument(..., help="Task id, e.g. T045."),
    trial: int = typer.Option(0, "--trial", help="Trial number."),
    run_dir: Path = typer.Option(Path("runs/scripted"), "--run-dir", help="Run directory."),
    data: Path = DataOpt,
) -> None:
    """Print one trial step by step with its state diff and policy violations."""
    db, tasks = _load(data)
    task = next((t for t in tasks if t.id == task_id), None)
    if task is None:
        raise typer.BadParameter(f"unknown task {task_id}")
    traj = next(
        (t for t in read_trajectories(run_dir) if t.task_id == task_id and t.trial == trial), None
    )
    if traj is None:
        raise typer.BadParameter(f"no trial {trial} of {task_id} in {run_dir}")
    g = grade_trajectory(task, traj, db)
    meta = read_meta(run_dir)
    console.print(f"[bold]{task.id}[/bold] {task.scenario}, trial {trial}, agent {meta['agent']}")
    console.print(f"customer: {task.persona.opening}\n")
    for event in traj.events[1:]:
        role = event["role"]
        if role == "user":
            console.print(f"  customer: {event['text']}", highlight=False)
        elif role == "agent":
            console.print(f"  agent:    {event['text']}", highlight=False)
        elif role == "call":
            args = json.dumps(event["args"], sort_keys=True)
            console.print(f"  call:     {event['tool']}({args})", highlight=False)
        elif role == "result":
            body = event["error"] or json.dumps(event["data"], sort_keys=True)
            status = "ok" if event["ok"] else "error"
            console.print(f"  result:   {status} {body[:110]}", highlight=False)
    console.print("\nState diff (initial -> final):")
    for line in describe_diff(g.diff):
        console.print(f"  {line}", highlight=False)
    console.print("Accepted goal diff(s):")
    for i, goal in enumerate(task.goals):
        console.print(f"  [{i}]", highlight=False)
        for line in describe_diff(goal):
            console.print(f"    {line}", highlight=False)
    for v in g.violations:
        console.print(f"violation {v['rule']}: {v['detail']}", highlight=False)
    console.print(f"outcome: {'pass' if g.passed else 'FAIL ' + str(g.label)}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
