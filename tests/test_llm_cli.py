import json
import os

import pytest
from typer.testing import CliRunner

from endstate.agents import LLMAgent, Message, Observation, OpenAICompatModel, ToolCall
from endstate.cli import app
from endstate.grader import GRADES_FILE


class FakeModel:
    """Deterministic stand-in for a chat endpoint: one tool call, then a text reply."""

    def __init__(self):
        self.seen = []
        self.replies = [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_booking",
                            "arguments": json.dumps({"booking_ref": "ABC123", "last_name": "Lee"}),
                        },
                    }
                ],
            },
            {"role": "assistant", "content": "Found it."},
        ]

    def complete(self, messages, tools):
        self.seen.append((list(messages), tools))
        return self.replies.pop(0)


def test_llm_agent_maps_messages_and_tool_calls_to_openai_shape():
    model = FakeModel()
    agent = LLMAgent(model)
    first = agent.step(Observation("user", "Hi, my flight was cancelled."))
    assert first == ToolCall("get_booking", {"booking_ref": "ABC123", "last_name": "Lee"})
    second = agent.step(Observation("tool", "{}", {"cabin": "economy"}, ok=True))
    assert second == Message("Found it.")
    messages, tools = model.seen[1]
    assert messages[0]["role"] == "system" and "Non-refundable" in messages[0]["content"]
    assert messages[-1]["role"] == "tool" and messages[-1]["tool_call_id"] == "call_1"
    assert {t["function"]["name"] for t in tools} >= {"rebook", "issue_voucher"}


def test_llm_agent_turns_unparseable_arguments_into_invalid_call():
    model = FakeModel()
    model.replies[0]["tool_calls"][0]["function"]["arguments"] = "{not json"
    call = LLMAgent(model).step(Observation("user", "hi"))
    assert call.tool == "get_booking" and "_unparsed" in call.args


def test_openai_compat_model_requires_env(monkeypatch):
    for name in ("ENDSTATE_LLM_BASE_URL", "ENDSTATE_LLM_API_KEY", "ENDSTATE_LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="ENDSTATE_LLM_BASE_URL"):
        OpenAICompatModel.from_env()


@pytest.mark.live
@pytest.mark.skipif(os.environ.get("ENDSTATE_LIVE") != "1", reason="set ENDSTATE_LIVE=1")
def test_live_agent_completes_a_conversation(tasks, db):  # pragma: no cover
    from endstate.runner import run_trial

    agent = LLMAgent(OpenAICompatModel.from_env())
    traj = run_trial(tasks[50], 0, agent, db)
    assert traj.n_steps > 0


def test_cli_seed_run_grade_report_replay(tmp_path):
    runner = CliRunner()
    data, runs = tmp_path / "data", tmp_path / "runs"
    assert runner.invoke(app, ["seed", "--out", str(data)]).exit_code == 0
    for agent in ("scripted", "polite-wrong"):
        out = runner.invoke(
            app,
            [
                "run",
                "--agent",
                agent,
                "--trials",
                "2",
                "--data",
                str(data),
                "--out",
                str(runs / agent),
            ],
        )
        assert out.exit_code == 0, out.output
        graded = runner.invoke(app, ["grade", str(runs / agent), "--data", str(data)])
        assert graded.exit_code == 0 and (runs / agent / GRADES_FILE).exists()
    report = runner.invoke(
        app, ["report", str(runs / "scripted"), str(runs / "polite-wrong"), "--data", str(data)]
    )
    assert report.exit_code == 0
    assert "pass^k" in report.output and "ROUGE-L" in report.output
    replay = runner.invoke(
        app,
        [
            "replay",
            "T041",
            "--trial",
            "0",
            "--run-dir",
            str(runs / "polite-wrong"),
            "--data",
            str(data),
        ],
    )
    assert replay.exit_code == 0
    assert "nonrefundable_cash" in replay.output and "outcome: FAIL" in replay.output


def test_cli_no_failures_flag_gives_perfect_run(tmp_path):
    runner = CliRunner()
    data, run = tmp_path / "data", tmp_path / "run"
    runner.invoke(app, ["seed", "--out", str(data)])
    runner.invoke(
        app, ["run", "--trials", "1", "--no-failures", "--data", str(data), "--out", str(run)]
    )
    out = runner.invoke(app, ["grade", str(run), "--data", str(data)])
    assert "60 passed" in out.output
