"""Runs task x trial conversations and saves trajectories as JSONL.

A trial is one conversation between an agent and the simulated customer against a fresh copy
of the seeded database. The step limit counts agent actions (messages and tool calls); the
tool budget counts tool calls. Hitting either one ends the trial.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from endstate.agents.base import Agent, Message, Observation, ToolCall
from endstate.env.db import Database
from endstate.env.tools import Toolbox, ToolResult
from endstate.tasks import Task
from endstate.user_sim import UserSimulator

STEP_LIMIT = 30
TOOL_BUDGET = 12
TRAJECTORY_FILE = "trajectories.jsonl"
META_FILE = "run.json"


@dataclass
class Trajectory:
    task_id: str
    trial: int
    agent: str
    terminated: str  # goal_met | agent_end | transferred | STEP_LIMIT | BUDGET
    events: list[dict[str, Any]] = field(default_factory=list)
    n_steps: int = 0
    n_tool_calls: int = 0
    fired: list[str] = field(default_factory=list)
    final_digest: str = ""

    def to_json(self) -> str:
        return json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Trajectory:
        return cls(**json.loads(line))

    def agent_text(self) -> str:
        return " ".join(e["text"] for e in self.events if e["role"] == "agent")


def _observation(result: ToolResult) -> Observation:
    text = json.dumps(result.to_dict(), sort_keys=True)
    err = result.error.value if result.error else None
    return Observation("tool", text, result.data, ok=result.ok, error=err)


def run_trial(
    task: Task,
    trial: int,
    agent: Agent,
    initial: Database,
    *,
    step_limit: int = STEP_LIMIT,
    tool_budget: int = TOOL_BUDGET,
) -> Trajectory:
    db = initial.copy()
    box = Toolbox(db)
    user = UserSimulator(task, initial)
    traj = Trajectory(task.id, trial, agent.name, terminated="STEP_LIMIT")
    obs = user.opening()
    traj.events.append({"role": "user", "text": obs.text, "data": obs.data})
    for _ in range(step_limit):
        action = agent.step(obs)
        traj.n_steps += 1
        if isinstance(action, Message):
            traj.events.append({"role": "agent", "text": action.text, "end": action.end})
            if action.end:
                traj.terminated = "agent_end"
                break
            turn = user.respond(action.text, db)
            obs = turn.observation
            traj.events.append({"role": "user", "text": obs.text, "data": obs.data})
            if turn.done:
                traj.terminated = "goal_met"
                break
            continue
        assert isinstance(action, ToolCall)
        if traj.n_tool_calls >= tool_budget:
            traj.events.append({"role": "refused", "tool": action.tool, "args": action.args})
            traj.terminated = "BUDGET"
            break
        traj.n_tool_calls += 1
        result = box.execute(action.tool, action.args)
        traj.events.append({"role": "call", "tool": action.tool, "args": action.args})
        traj.events.append({"role": "result", **result.to_dict()})
        obs = _observation(result)
        if result.data.get("transferred"):
            turn = user.on_transfer()
            traj.events.append({"role": "user", "text": turn.observation.text, "data": {}})
            traj.terminated = "transferred"
            break
    traj.fired = list(getattr(agent, "fired", []))
    traj.final_digest = db.digest()
    return traj


def run_all(
    tasks: Iterable[Task],
    initial: Database,
    make: Callable[[Task, int], Agent],
    trials: int,
) -> list[Trajectory]:
    return [
        run_trial(task, trial, make(task, trial), initial)
        for task in tasks
        for trial in range(trials)
    ]


def write_run(out_dir: Path, trajectories: list[Trajectory], meta: dict[str, Any]) -> str:
    """Write trajectories.jsonl and run.json; return the trajectory file's sha256."""
    import hashlib

    out_dir.mkdir(parents=True, exist_ok=True)
    body = "".join(t.to_json() + "\n" for t in trajectories)
    (out_dir / TRAJECTORY_FILE).write_text(body)
    (out_dir / META_FILE).write_text(json.dumps(meta, sort_keys=True, indent=1) + "\n")
    return hashlib.sha256(body.encode()).hexdigest()


def read_trajectories(run_dir: Path) -> list[Trajectory]:
    lines = (run_dir / TRAJECTORY_FILE).read_text().splitlines()
    return [Trajectory.from_json(line) for line in lines if line.strip()]


def read_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / META_FILE).read_text())
