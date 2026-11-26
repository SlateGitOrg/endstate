import hashlib
import json

from conftest import TRIALS, FixedAgent, by_scenario, play

from endstate.agents import FailureRates, Message, Observation, ToolCall, make_agent
from endstate.grader import grade_trajectory
from endstate.metrics import pass_hat_k
from endstate.runner import (
    STEP_LIMIT,
    TOOL_BUDGET,
    read_trajectories,
    run_all,
    run_trial,
    write_run,
)


class LoopingAgent:
    name = "looping"
    fired: list[str] = []

    def step(self, observation: Observation) -> Message:
        return Message("Let me look into that for you.")


class CallHeavyAgent:
    name = "call-heavy"
    fired: list[str] = []

    def step(self, observation: Observation) -> ToolCall:
        return ToolCall(
            "search_flights",
            {"origin": "SFO", "destination": "LAX", "date": "2026-03-02", "cabin": "economy"},
        )


def test_clean_scripted_agent_passes_all_tasks_on_all_trials(tasks, clean_run):
    trajs, grades = clean_run
    assert len(grades) == len(tasks) * TRIALS == 480
    failed = [(g.task_id, g.trial, g.label) for g in grades if not g.passed]
    assert failed == []
    assert pass_hat_k([TRIALS] * len(tasks), TRIALS, TRIALS) == 1.0
    assert all(t.fired == [] for t in trajs)
    assert all(t.terminated in ("goal_met", "transferred") for t in trajs)


def test_looping_agent_stops_at_step_30_with_step_limit(tasks, db):
    traj = run_trial(tasks[0], 0, LoopingAgent(), db)
    assert STEP_LIMIT == 30 and traj.n_steps == 30 and traj.terminated == "STEP_LIMIT"
    grade = grade_trajectory(tasks[0], traj, db)
    assert not grade.passed and grade.label == "STEP_LIMIT"


def test_call_heavy_agent_stops_on_tool_budget(tasks, db):
    traj = run_trial(tasks[0], 0, CallHeavyAgent(), db)
    assert TOOL_BUDGET == 12 and traj.n_tool_calls == 12 and traj.terminated == "BUDGET"
    assert traj.events[-1]["role"] == "refused"
    assert grade_trajectory(tasks[0], traj, db).label == "BUDGET"


def test_either_acceptable_goal_diff_passes_for_each_dual_goal_task(tasks, db):
    dual = [t for t in tasks if len(t.goals) == 2]
    assert len(dual) == 8
    for t in dual:
        flights = [next(iter(g["bookings"].values()))["flight_id"] for g in t.goals]
        assert len(set(flights)) == 2
        for index, flight in enumerate(flights):
            actions = [dict(a, args=dict(a["args"])) for a in t.reference_actions]
            for a in actions:
                if a["tool"] == "rebook":
                    a["args"]["flight_id"] = flight
            grade = grade_trajectory(t, play(t, db, actions), db)
            assert grade.passed and grade.goal_index == index, (t.id, flight)


def test_goal_diff_rejects_the_wrong_alternative(tasks, db):
    t = by_scenario(tasks, "cancelled_rebook")[0]  # single goal, requested flight is full
    actions = [dict(a, args=dict(a["args"])) for a in t.reference_actions]
    wrong = t.persona.requested_flight
    for a in actions:
        if a["tool"] == "rebook":
            a["args"]["flight_id"] = wrong
    traj = play(t, db, actions)
    assert any(e.get("error") == "NO_SEATS" for e in traj.events)
    assert not grade_trajectory(t, traj, db).passed


def test_full_requested_flight_is_recovered_by_scripted_agent(tasks, db):
    t = next(t for t in tasks if t.persona.requested_flight)
    traj = run_trial(t, 0, make_agent("scripted", t, 0, rates=FailureRates.none()), db)
    errors = [e["error"] for e in traj.events if e["role"] == "result" and not e["ok"]]
    assert errors == ["NO_SEATS"]
    assert grade_trajectory(t, traj, db).passed


def test_polite_wrong_agent_matches_transcripts_but_fails_every_end_state(tasks, polite_run):
    _, grades = polite_run
    nonref = {t.id for t in tasks if t.scenario == "refund_nonrefundable"}
    rows = [g for g in grades if g.task_id in nonref]
    assert len(rows) == 10 * TRIALS
    assert sum(g.rouge_l for g in rows) / len(rows) >= 0.8
    assert sum(g.passed for g in rows) == 0
    assert {g.label for g in rows} == {"POLICY:nonrefundable_cash"}
    assert sum(g.passed for g in grades) == 0


def test_same_seed_gives_identical_trajectory_hash(tasks, db, tmp_path):
    subset = tasks[:12]

    def digest(seed: int, name: str) -> str:
        trajs = run_all(subset, db, lambda t, i: make_agent("scripted", t, i, seed=seed), 4)
        return write_run(tmp_path / name, trajs, {"agent": "scripted", "trials": 4})

    a, b, c = digest(0, "a"), digest(0, "b"), digest(1, "c")
    assert a == b != c
    file_hash = hashlib.sha256((tmp_path / "a" / "trajectories.jsonl").read_bytes()).hexdigest()
    assert file_hash == a
    assert len(read_trajectories(tmp_path / "a")) == 48


def test_trajectories_record_every_step_as_json(tasks, db):
    t = by_scenario(tasks, "name_correction")[0]
    traj = run_trial(t, 0, make_agent("scripted", t, 0, rates=FailureRates.none()), db)
    roles = [e["role"] for e in traj.events]
    assert roles[0] == "user" and "call" in roles and traj.terminated == "transferred"
    assert json.loads(traj.to_json())["task_id"] == t.id
    assert FixedAgent([]).step(Observation("user")).end
