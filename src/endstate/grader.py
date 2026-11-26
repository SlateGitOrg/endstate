"""Grades trajectories by the database they leave behind.

The grader replays the trajectory's tool calls on a fresh copy of the seeded database (so it
never trusts a recorded state), diffs initial against final canonical state, and matches that
diff against the task's acceptable goal diffs. The policy rules then run over the action log.
A trial passes only when a goal diff matches and no rule is violated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from endstate.env.db import Database, diff_states, diffs_equal
from endstate.env.tools import Toolbox
from endstate.metrics import rouge_l
from endstate.policy import CallRecord, PolicyContext, Violation, check_policy
from endstate.runner import Trajectory
from endstate.tasks import Task

GRADES_FILE = "grades.jsonl"
WRONG_END_STATE = "WRONG_END_STATE"
STEP_LIMIT = "STEP_LIMIT"
BUDGET = "BUDGET"
TOOL_ERROR_UNRECOVERED = "TOOL_ERROR_UNRECOVERED"


class ReplayMismatch(RuntimeError):
    """Raised when replaying a trajectory does not reproduce its recorded end state."""


@dataclass
class Grade:
    task_id: str
    trial: int
    passed: bool
    label: str | None  # None when passed
    goal_index: int | None
    violations: list[dict[str, str]] = field(default_factory=list)
    diff: dict[str, Any] = field(default_factory=dict)
    rouge_l: float = 0.0
    fired: list[str] = field(default_factory=list)
    n_steps: int = 0
    n_tool_calls: int = 0
    terminated: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Grade:
        return cls(**json.loads(line))


def replay(traj: Trajectory, initial: Database) -> tuple[Database, list[CallRecord]]:
    """Re-execute the recorded tool calls; return the final database and the action log."""
    db = initial.copy()
    box = Toolbox(db)
    calls: list[CallRecord] = []
    for event in traj.events:
        if event["role"] == "call":
            result = box.execute(event["tool"], event["args"])
            calls.append(
                CallRecord(
                    step=len(calls) + 1,
                    tool=event["tool"],
                    args=event["args"],
                    ok=result.ok,
                    error=result.error.value if result.error else None,
                    data=result.data,
                )
            )
    return db, calls


def failure_label(
    traj: Trajectory, matched: bool, violations: list[Violation], calls: list[CallRecord]
) -> str | None:
    """Primary failure class; None for a pass.

    Order: a limit hit, then an unrecovered tool error (the last call failed and the goal was
    not reached), then the first violated rule, then a plain wrong end state.
    """
    if traj.terminated in (STEP_LIMIT, BUDGET):
        return traj.terminated
    if matched and not violations:
        return None
    if calls and not calls[-1].ok and not matched:
        return TOOL_ERROR_UNRECOVERED
    if violations:
        return f"POLICY:{violations[0].rule}"
    return WRONG_END_STATE


def grade_trajectory(task: Task, traj: Trajectory, initial: Database, reference: str = "") -> Grade:
    final, calls = replay(traj, initial)
    if final.digest() != traj.final_digest:
        raise ReplayMismatch(f"{task.id} trial {traj.trial}: replay does not match the recording")
    diff = diff_states(initial.canonical(), final.canonical())
    goal_index = next((i for i, g in enumerate(task.goals) if diffs_equal(diff, g)), None)
    violations = check_policy(PolicyContext(task, calls, initial, final))
    matched = goal_index is not None
    label = failure_label(traj, matched, violations, calls)
    transcript = reference or " ".join(task.reference_transcript)
    return Grade(
        task_id=task.id,
        trial=traj.trial,
        passed=label is None,
        label=label,
        goal_index=goal_index,
        violations=[{"rule": v.rule, "detail": v.detail} for v in violations],
        diff=diff,
        rouge_l=round(rouge_l(traj.agent_text(), transcript), 4),
        fired=list(traj.fired),
        n_steps=traj.n_steps,
        n_tool_calls=traj.n_tool_calls,
        terminated=traj.terminated,
    )


def grade_all(tasks: list[Task], trajectories: list[Trajectory], initial: Database) -> list[Grade]:
    by_id = {t.id: t for t in tasks}
    return [grade_trajectory(by_id[tr.task_id], tr, initial) for tr in trajectories]


def write_grades(run_dir: Path, grades: list[Grade]) -> None:
    (run_dir / GRADES_FILE).write_text("".join(g.to_json() + "\n" for g in grades))


def read_grades(run_dir: Path) -> list[Grade]:
    lines = (run_dir / GRADES_FILE).read_text().splitlines()
    return [Grade.from_json(line) for line in lines if line.strip()]
