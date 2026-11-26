"""Agents under test."""

from __future__ import annotations

from endstate.agents.base import Agent, Message, Observation, ToolCall
from endstate.agents.llm import ChatModel, LLMAgent, OpenAICompatModel
from endstate.agents.polite import PoliteWrongAgent
from endstate.agents.scripted import FailureRates, ScriptedAgent
from endstate.tasks import Task

AGENT_NAMES = ("scripted", "polite-wrong", "llm")


def make_agent(
    name: str,
    task: Task,
    trial: int,
    *,
    rates: FailureRates | None = None,
    seed: int = 0,
    model: ChatModel | None = None,
) -> Agent:
    if name == "scripted":
        return ScriptedAgent(task, trial, rates, seed)
    if name == "polite-wrong":
        return PoliteWrongAgent(task)
    if name == "llm":
        return LLMAgent(model or OpenAICompatModel.from_env())
    raise ValueError(f"unknown agent {name!r}; choose from {', '.join(AGENT_NAMES)}")


__all__ = [
    "AGENT_NAMES",
    "Agent",
    "ChatModel",
    "FailureRates",
    "LLMAgent",
    "Message",
    "Observation",
    "OpenAICompatModel",
    "PoliteWrongAgent",
    "ScriptedAgent",
    "ToolCall",
    "make_agent",
]
