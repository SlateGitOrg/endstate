from __future__ import annotations

from collections import defaultdict
from typing import Any

import pytest

from endstate.agents import FailureRates, Message, Observation, ToolCall, make_agent
from endstate.env.db import Database
from endstate.env.tools import Toolbox
from endstate.generate import generate
from endstate.grader import Grade, grade_all
from endstate.policy import CallRecord, PolicyContext
from endstate.runner import Trajectory, run_all, run_trial
from endstate.tasks import Task

TRIALS = 8


@pytest.fixture(scope="session")
def seeded() -> tuple[Database, list[Task]]:
    return generate(7)


@pytest.fixture(scope="session")
def db(seeded) -> Database:
    return seeded[0]


@pytest.fixture(scope="session")
def tasks(seeded) -> list[Task]:
    return seeded[1]


def by_scenario(tasks: list[Task], scenario: str) -> list[Task]:
    return [t for t in tasks if t.scenario == scenario]


class FixedAgent:
    """Plays a fixed list of tool calls, then hangs up."""

    name = "fixed"

    def __init__(self, actions: list[dict[str, Any]]) -> None:
        self.actions = list(actions)
        self.fired: list[str] = []

    def step(self, observation: Observation) -> ToolCall | Message:
        if self.actions:
            a = self.actions.pop(0)
            return ToolCall(a["tool"], a["args"])
        return Message("done", end=True)


def play(task: Task, db: Database, actions: list[dict[str, Any]]) -> Trajectory:
    return run_trial(task, 0, FixedAgent(actions), db)


def context(task: Task, db: Database, actions: list[dict[str, Any]]) -> PolicyContext:
    """Execute hand-built calls against a copy of the database and return the policy context."""
    work = db.copy()
    box = Toolbox(work)
    calls = []
    for i, a in enumerate(actions, start=1):
        r = box.execute(a["tool"], a["args"])
        calls.append(
            CallRecord(i, a["tool"], a["args"], r.ok, r.error.value if r.error else None, r.data)
        )
    return PolicyContext(task, calls, db, work)


def pass_counts(grades: list[Grade], tasks: list[Task]) -> list[int]:
    per_task: dict[str, int] = defaultdict(int)
    for g in grades:
        per_task[g.task_id] += g.passed
    return [per_task[t.id] for t in tasks]


@pytest.fixture(scope="session")
def clean_run(db, tasks):
    trajs = run_all(
        tasks, db, lambda t, i: make_agent("scripted", t, i, rates=FailureRates.none()), TRIALS
    )
    return trajs, grade_all(tasks, trajs, db)


@pytest.fixture(scope="session")
def planted_run(db, tasks):
    trajs = run_all(tasks, db, lambda t, i: make_agent("scripted", t, i), TRIALS)
    return trajs, grade_all(tasks, trajs, db)


@pytest.fixture(scope="session")
def polite_run(db, tasks):
    trajs = run_all(tasks, db, lambda t, i: make_agent("polite-wrong", t, i), TRIALS)
    return trajs, grade_all(tasks, trajs, db)
