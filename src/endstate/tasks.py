"""Task specifications: persona facts, an initial-state seed and acceptable goal diffs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

CANCELLED_REBOOK = "cancelled_rebook"
MISSED_CONNECTION = "missed_connection"
VOLUNTARY_CHANGE = "voluntary_change"
REFUND_REFUNDABLE = "refund_refundable"
REFUND_NONREFUNDABLE = "refund_nonrefundable"
NAME_CORRECTION = "name_correction"

SCENARIOS = (
    CANCELLED_REBOOK,
    MISSED_CONNECTION,
    VOLUNTARY_CHANGE,
    REFUND_REFUNDABLE,
    REFUND_NONREFUNDABLE,
    NAME_CORRECTION,
)
REBOOK_SCENARIOS = frozenset({CANCELLED_REBOOK, MISSED_CONNECTION, VOLUNTARY_CHANGE})
VOLUNTARY_SCENARIOS = frozenset({VOLUNTARY_CHANGE, REFUND_REFUNDABLE})


@dataclass
class Persona:
    name: str
    booking_ref: str
    last_name: str
    opening: str  # the customer's first message
    ticket: dict[str, Any]  # what the customer can read off their own ticket
    requested_flight: str | None = None
    desired_date: str | None = None


@dataclass
class Task:
    id: str
    scenario: str
    persona: Persona
    goals: list[dict[str, Any]]  # acceptable goal diffs; any one of them passes
    reference_transcript: list[str]  # the agent turns of a correct conversation
    reference_actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def voluntary(self) -> bool:
        return self.scenario in VOLUNTARY_SCENARIOS

    @property
    def has_write(self) -> bool:
        return self.scenario != NAME_CORRECTION

    @property
    def has_rebook(self) -> bool:
        return self.scenario in REBOOK_SCENARIOS

    @property
    def booking_ref(self) -> str:
        return self.persona.booking_ref


def task_from_dict(raw: dict[str, Any]) -> Task:
    return Task(
        id=raw["id"],
        scenario=raw["scenario"],
        persona=Persona(**raw["persona"]),
        goals=raw["goals"],
        reference_transcript=raw["reference_transcript"],
        reference_actions=raw.get("reference_actions", []),
    )


def save_tasks(tasks: list[Task], path: Path) -> None:
    path.write_text(yaml.safe_dump([asdict(t) for t in tasks], sort_keys=False, width=100))


def load_tasks(path: Path) -> list[Task]:
    return [task_from_dict(raw) for raw in yaml.safe_load(path.read_text())]
