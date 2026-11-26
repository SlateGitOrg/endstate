"""Types shared by agents, the user simulator and the runner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolCall:
    tool: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Message:
    text: str
    end: bool = False  # the agent is done and hangs up without waiting for the customer


@dataclass(frozen=True)
class Observation:
    """What an agent sees next.

    ``kind`` is ``"user"`` (a customer message) or ``"tool"`` (a tool result). ``data`` is a
    structured copy of the same content for rule-based agents; a language-model agent reads
    only ``text``.
    """

    kind: str
    text: str = ""
    data: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


class Agent(Protocol):
    name: str

    def step(self, observation: Observation) -> ToolCall | Message: ...
