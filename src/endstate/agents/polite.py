"""An agent that sounds exactly like the reference transcripts and always pays cash.

It exists to show what transcript-similarity metrics miss: its words score high against the
reference conversation while the booking database ends up wrong.
"""

from __future__ import annotations

from endstate import phrasing
from endstate.agents.base import Message, Observation, ToolCall
from endstate.tasks import REFUND_NONREFUNDABLE, Task


class PoliteWrongAgent:
    name = "polite-wrong"

    def __init__(self, task: Task) -> None:
        self.task = task
        self.fired: list[str] = []
        self._gen = self._script()
        next(self._gen)

    def step(self, observation: Observation) -> ToolCall | Message:
        try:
            return self._gen.send(observation)
        except StopIteration:
            return Message("Thank you for contacting support. Goodbye.", end=True)

    def _script(self):
        turns = self.task.reference_transcript
        yield None
        reply = yield Message(turns[0])
        ref, last = reply.data.get("booking_ref"), reply.data.get("last_name")
        res = yield ToolCall("get_booking", {"booking_ref": ref, "last_name": last})
        if not res.ok:
            yield Message(phrasing.APOLOGY, end=True)
            return
        yield Message(turns[1])
        fare = float(res.data["fare_paid"])
        yield ToolCall("issue_refund", {"booking_ref": ref, "amount": fare})
        closing = turns[2]
        if self.task.scenario == REFUND_NONREFUNDABLE:
            closing = f"{phrasing.resolution_refund(fare)} {phrasing.CLOSING}"
        yield Message(closing)
