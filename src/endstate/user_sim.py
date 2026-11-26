"""The simulated customer: a deterministic state machine driven by persona facts.

The customer opens with the request, answers the agent's questions from persona facts, and
ends the conversation when the database matches one of its acceptable goal states or when it
has been transferred to a human. It never judges wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from endstate.agents.base import Observation
from endstate.env.db import Database, diff_states, diffs_equal
from endstate.tasks import Task

IDENTITY_WORDS = ("booking reference", "confirmation", "last name", "surname")
TICKET_WORDS = ("flight number", "cabin", "fare you paid", "your ticket", "ticket details")


class UserState(StrEnum):
    START = "start"
    OPEN = "open"  # request made, nothing verified yet
    IDENTIFIED = "identified"  # gave booking reference and last name
    TICKET_SHARED = "ticket_shared"  # also read the ticket details out
    DONE = "done"


@dataclass
class UserTurn:
    observation: Observation
    done: bool = False


class UserSimulator:
    def __init__(self, task: Task, initial: Database) -> None:
        self.task = task
        self._initial = initial.canonical()
        self.state = UserState.START

    # ------------------------------------------------------------------ helpers
    def goal_met(self, db: Database) -> bool:
        actual = diff_states(self._initial, db.canonical())
        return any(diffs_equal(actual, goal) for goal in self.task.goals)

    def _intent_data(self) -> dict[str, Any]:
        p = self.task.persona
        return {
            "intent": self.task.scenario,
            "requested_flight": p.requested_flight,
            "desired_date": p.desired_date,
        }

    # ------------------------------------------------------------------ dialogue
    def opening(self) -> Observation:
        self.state = UserState.OPEN
        return Observation("user", self.task.persona.opening, self._intent_data())

    def on_transfer(self) -> UserTurn:
        self.state = UserState.DONE
        return UserTurn(
            Observation("user", "Okay, I will wait for the specialist. Thank you."), done=True
        )

    def respond(self, agent_text: str, db: Database) -> UserTurn:
        """Reply to an agent message. ``done`` means the customer has ended the dialogue."""
        if self.goal_met(db):
            self.state = UserState.DONE
            return UserTurn(
                Observation("user", "That is exactly what I needed, thank you. Goodbye."),
                done=True,
            )
        # Only the questions in the agent's message count as questions.
        questions = " ".join(s for s in agent_text.lower().split(".") if "?" in s)
        persona = self.task.persona
        data: dict[str, Any] = {}
        parts: list[str] = []
        if any(w in questions for w in IDENTITY_WORDS):
            data.update(booking_ref=persona.booking_ref, last_name=persona.last_name)
            parts.append(
                f"My booking reference is {persona.booking_ref} and my last name is "
                f"{persona.last_name}."
            )
            if self.state is UserState.OPEN:
                self.state = UserState.IDENTIFIED
        if any(w in questions for w in TICKET_WORDS):
            data["ticket"] = dict(persona.ticket)
            t = persona.ticket
            parts.append(
                f"It is flight {t['flight_id']} from {t['origin']} to {t['destination']} on "
                f"{t['date']}, in {t['cabin']}, and I paid ${t['fare_paid']:.2f}."
            )
            self.state = UserState.TICKET_SHARED
        if parts:
            return UserTurn(Observation("user", " ".join(parts), data))
        return UserTurn(Observation("user", "Yes, please go ahead.", {}))
