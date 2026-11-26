"""A rule-based agent that follows the correct fare rules, with planted failure modes.

Every failure mode fires with a fixed probability drawn from an RNG seeded by (task id,
trial). All draws are taken up front, so whether a mode fires never depends on the path the
conversation takes. That makes the agent's per-task success probability computable in closed
form (``success_probability``), which is how the harness is validated.
"""

from __future__ import annotations

import zlib
from collections.abc import Generator
from dataclasses import dataclass
from typing import Any

import numpy as np

from endstate import phrasing
from endstate.agents.base import Message, Observation, ToolCall
from endstate.policy import compensation_for_delay
from endstate.tasks import (
    CANCELLED_REBOOK,
    MISSED_CONNECTION,
    NAME_CORRECTION,
    REFUND_NONREFUNDABLE,
    Task,
)

MODES = ("skip_verification", "cash_refund_nonrefundable", "wrong_cabin", "hallucinated_flight_id")
_CABIN_ORDER = ("economy", "premium", "business")
Script = Generator["ToolCall | Message | None", Observation, Any]


@dataclass(frozen=True)
class FailureRates:
    skip_verification: float = 0.10
    cash_refund_nonrefundable: float = 0.15
    wrong_cabin: float = 0.05
    hallucinated_flight_id: float = 0.05
    hallucination_recovery: float = 0.5  # chance of searching again after the bad flight id

    @classmethod
    def none(cls) -> FailureRates:
        return cls(0.0, 0.0, 0.0, 0.0, 0.5)

    def rate(self, mode: str) -> float:
        return float(getattr(self, mode))

    def as_dict(self) -> dict[str, float]:
        return {m: self.rate(m) for m in MODES} | {
            "hallucination_recovery": self.hallucination_recovery
        }


def applicable(task: Task, mode: str) -> bool:
    """Whether a failure mode can manifest on this task."""
    if mode == "skip_verification":
        return task.has_write
    if mode == "cash_refund_nonrefundable":
        return task.scenario == REFUND_NONREFUNDABLE
    return task.has_rebook  # wrong_cabin, hallucinated_flight_id


def success_probability(task: Task, rates: FailureRates) -> float:
    """P(a trial of the scripted agent passes), from the planted probabilities alone.

    Each failure mode that can manifest always ends in a failed trial, and the draws are
    independent, so the pass probability is a product. A hallucinated flight id is only a
    failure when the agent does not recover.
    """
    p = 1.0
    if applicable(task, "skip_verification"):
        p *= 1 - rates.skip_verification
    if applicable(task, "cash_refund_nonrefundable"):
        p *= 1 - rates.cash_refund_nonrefundable
    if applicable(task, "wrong_cabin"):
        p *= 1 - rates.wrong_cabin
    if applicable(task, "hallucinated_flight_id"):
        p *= 1 - rates.hallucinated_flight_id * (1 - rates.hallucination_recovery)
    return p


def applicable_trials(tasks: list[Task], trials: int) -> dict[str, int]:
    return {m: trials * sum(applicable(t, m) for t in tasks) for m in MODES}


def trial_draws(task_id: str, trial: int, seed: int = 0) -> np.ndarray:
    """Six uniform draws, a pure function of (seed, task id, trial)."""
    seq = np.random.SeedSequence([seed, zlib.crc32(task_id.encode()), trial])
    return np.random.default_rng(seq).random(6)


class ScriptedAgent:
    name = "scripted"

    def __init__(
        self, task: Task, trial: int, rates: FailureRates | None = None, seed: int = 0
    ) -> None:
        self.task = task
        self.rates = rates or FailureRates()
        u = trial_draws(task.id, trial, seed)
        self._drawn = {
            "skip_verification": u[0] < self.rates.skip_verification,
            "cash_refund_nonrefundable": u[1] < self.rates.cash_refund_nonrefundable,
            "wrong_cabin": u[2] < self.rates.wrong_cabin,
            "hallucinated_flight_id": u[3] < self.rates.hallucinated_flight_id,
        }
        self._recovers = bool(u[4] < self.rates.hallucination_recovery)
        self._tiebreak = float(u[5])
        self.fired: list[str] = []
        self._gen = self._script()
        next(self._gen)

    def step(self, observation: Observation) -> ToolCall | Message:
        try:
            action = self._gen.send(observation)
        except StopIteration:
            return Message("Thank you for contacting support. Goodbye.", end=True)
        return action  # type: ignore[return-value]

    # ------------------------------------------------------------------ failure draws
    def _fires(self, mode: str) -> bool:
        """True when ``mode`` is drawn and can manifest here; records it."""
        if self._drawn[mode] and applicable(self.task, mode):
            self.fired.append(mode)
            return True
        return False

    # ------------------------------------------------------------------ the policy
    def _script(self) -> Script:
        opening = yield None
        reply = yield Message(phrasing.opening_turn())
        ref, last = reply.data.get("booking_ref"), reply.data.get("last_name")
        if not ref or not last:
            yield Message(phrasing.APOLOGY, end=True)
            return
        if self._fires("skip_verification"):
            # Trusts the customer's own account of the ticket instead of looking it up.
            reply = yield Message(phrasing.ASK_TICKET)
            info = dict(reply.data.get("ticket", {}))
        else:
            res = yield ToolCall("get_booking", {"booking_ref": ref, "last_name": last})
            if not res.ok:
                yield Message(phrasing.APOLOGY, end=True)
                return
            info = dict(res.data)
        yield Message(phrasing.found_turn(self.task.scenario))

        scenario = self.task.scenario
        if scenario == NAME_CORRECTION:
            yield Message(phrasing.resolution_transfer())
            yield ToolCall(
                "transfer_to_human", {"reason": "name correction requested by the passenger"}
            )
            return

        final = yield from self._resolve(scenario, ref, info, opening.data)
        if final is None:
            yield Message(phrasing.APOLOGY, end=True)
            return
        yield Message(f"{final} {phrasing.CLOSING}")

    def _resolve(self, scenario: str, ref: str, info: dict[str, Any], intent: dict[str, Any]):
        if scenario in (CANCELLED_REBOOK, MISSED_CONNECTION) or scenario == "voluntary_change":
            date = intent.get("desired_date") or info["date"]
            flight = yield from self._rebook(ref, info, date, intent.get("requested_flight"))
            if flight is None:
                return None
            if scenario == CANCELLED_REBOOK:
                return phrasing.resolution_rebook(
                    flight["flight_id"], flight["depart"], info["cabin"]
                )
            if scenario == MISSED_CONNECTION:
                amount = compensation_for_delay(int(info["delay_minutes"]))
                res = yield ToolCall("issue_voucher", {"booking_ref": ref, "amount": amount})
                if not res.ok:
                    return None
                return phrasing.resolution_missed(
                    flight["flight_id"], flight["depart"], info["cabin"], amount
                )
            refund = round(
                float(info["fare_paid"]) - flight["price"] - float(info["change_fee"]), 2
            )
            res = yield ToolCall("issue_refund", {"booking_ref": ref, "amount": refund})
            if not res.ok:
                return None
            return phrasing.resolution_change(
                flight["flight_id"], flight["depart"], refund, float(info["change_fee"])
            )

        res = yield ToolCall("cancel_booking", {"booking_ref": ref})
        if not res.ok:
            return None
        fare = float(info["fare_paid"])
        if info["refundable"] or self._fires("cash_refund_nonrefundable"):
            amount = round(fare - float(info["change_fee"]), 2) if info["refundable"] else fare
            res = yield ToolCall("issue_refund", {"booking_ref": ref, "amount": amount})
            return phrasing.resolution_refund(amount) if res.ok else None
        res = yield ToolCall("issue_voucher", {"booking_ref": ref, "amount": fare})
        return phrasing.resolution_voucher(fare) if res.ok else None

    def _search(self, ref: str, info: dict[str, Any], date: str):
        res = yield ToolCall(
            "search_flights",
            {
                "origin": info["origin"],
                "destination": info["destination"],
                "date": date,
                "cabin": info["cabin"],
            },
        )
        return res.data.get("flights", []) if res.ok else []

    def _rebook(self, ref: str, info: dict[str, Any], date: str, requested: str | None):
        cabin = info["cabin"]
        if requested:
            res = yield ToolCall(
                "rebook", {"booking_ref": ref, "flight_id": requested, "cabin": cabin}
            )
            if res.ok:
                return {"flight_id": requested, "depart": res.data["depart"], "price": 0.0}
        flights = yield from self._search(ref, info, date)
        if not flights:
            return None
        if self._fires("hallucinated_flight_id"):
            fake = f"FL9{int(self._tiebreak * 900) + 100}"
            res = yield ToolCall("rebook", {"booking_ref": ref, "flight_id": fake, "cabin": cabin})
            if not self._recovers:
                return None
            flights = yield from self._search(ref, info, date)
            if not flights:
                return None
        choice = flights[int(self._tiebreak * len(flights))]
        target = cabin
        if self._fires("wrong_cabin"):
            target = _CABIN_ORDER[(_CABIN_ORDER.index(cabin) + 1) % len(_CABIN_ORDER)]
        res = yield ToolCall(
            "rebook", {"booking_ref": ref, "flight_id": choice["flight_id"], "cabin": target}
        )
        return choice if res.ok else None
