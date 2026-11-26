"""Declarative fare rules, checked over the action log and the final state.

Each ``Rule`` is a name, a plain-English statement and a check function. Rules never look at
transcript wording: they read the tool calls the agent made and the databases before and
after.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from endstate.env.db import Database, money
from endstate.env.tools import WRITE_TOOLS
from endstate.tasks import NAME_CORRECTION, Task

# Synthetic delay-compensation table: (minimum delay in minutes, voucher dollars).
COMPENSATION_BANDS: tuple[tuple[int, float], ...] = ((480, 200.0), (240, 100.0), (120, 50.0))


def compensation_for_delay(minutes: int) -> float:
    for floor, amount in COMPENSATION_BANDS:
        if minutes >= floor:
            return amount
    return 0.0


POLICY_TEXT = (
    "Fare rules you must follow:\n"
    "1. Verify the passenger (a successful get_booking with the right last name) before any "
    "rebook, cancel, refund or voucher.\n"
    "2. Non-refundable fares never get cash back; issue a voucher for the fare paid instead.\n"
    "3. A refund for a voluntary change or cancellation is at most the fare paid minus the "
    "change fee.\n"
    "4. A rebooking stays in the same cabin.\n"
    "5. Name corrections cannot be done by you: transfer the customer to a human.\n"
    "6. Delay compensation is a voucher by delay band: 120-239 minutes $50, 240-479 minutes "
    "$100, 480 minutes or more $200.\n"
    "7. At most one compensation voucher per booking."
)


@dataclass(frozen=True)
class CallRecord:
    step: int
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None
    data: dict[str, Any]


@dataclass(frozen=True)
class Violation:
    rule: str
    detail: str


@dataclass
class PolicyContext:
    task: Task
    calls: list[CallRecord]
    initial: Database
    final: Database


@dataclass(frozen=True)
class Rule:
    name: str
    statement: str
    check: Callable[[PolicyContext], list[Violation]]


def _ref(call: CallRecord) -> str:
    return str(call.args.get("booking_ref", "")).strip().upper()


def _identity_verified(ctx: PolicyContext) -> list[Violation]:
    verified: set[str] = set()
    out = []
    for call in ctx.calls:
        if call.tool == "get_booking" and call.ok:
            verified.add(_ref(call))
        elif call.tool in WRITE_TOOLS and _ref(call) not in verified:
            out.append(
                Violation("identity_verified", f"{call.tool} on {_ref(call)} before verifying")
            )
    return out


def _nonrefundable_cash(ctx: PolicyContext) -> list[Violation]:
    out = []
    for call in ctx.calls:
        if call.tool != "issue_refund" or not call.ok:
            continue
        booking = ctx.initial.bookings.get(_ref(call))
        if booking is not None and not booking.refundable:
            out.append(
                Violation(
                    "nonrefundable_cash",
                    f"cash refund of {money(call.args['amount']):.2f} on non-refundable "
                    f"{booking.ref}",
                )
            )
    return out


def _voluntary_refund_cap(ctx: PolicyContext) -> list[Violation]:
    if not ctx.task.voluntary:
        return []
    totals: dict[str, float] = defaultdict(float)
    for call in ctx.calls:
        if call.tool == "issue_refund" and call.ok:
            totals[_ref(call)] += float(call.args["amount"])
    out = []
    for ref, total in totals.items():
        booking = ctx.initial.bookings[ref]
        cap = money(booking.fare_paid - booking.change_fee)
        if money(total) > cap:
            out.append(
                Violation(
                    "voluntary_refund_cap",
                    f"refunded {money(total):.2f}, cap is {cap:.2f} on {ref}",
                )
            )
    return out


def _same_cabin(ctx: PolicyContext) -> list[Violation]:
    out = []
    for ref, before in ctx.initial.bookings.items():
        after = ctx.final.bookings[ref]
        if after.cabin != before.cabin:
            out.append(Violation("same_cabin", f"{ref} moved from {before.cabin} to {after.cabin}"))
    return out


def _name_correction_handoff(ctx: PolicyContext) -> list[Violation]:
    if ctx.task.scenario != NAME_CORRECTION:
        return []
    if any(c.tool == "transfer_to_human" and c.ok for c in ctx.calls):
        return []
    return [Violation("name_correction_handoff", "name correction was not transferred to a human")]


def _delay_compensation_table(ctx: PolicyContext) -> list[Violation]:
    out = []
    for call in ctx.calls:
        if call.tool != "issue_voucher" or not call.ok:
            continue
        booking = ctx.initial.bookings.get(_ref(call))
        if booking is None:
            continue
        delay = ctx.initial.flights[booking.flight_id].delay_minutes
        if delay <= 0:
            continue
        expected = compensation_for_delay(delay)
        if money(call.args["amount"]) != money(expected):
            out.append(
                Violation(
                    "delay_compensation_table",
                    f"voucher {money(call.args['amount']):.2f} for a {delay} minute delay, "
                    f"table says {expected:.2f}",
                )
            )
    return out


def _one_compensation(ctx: PolicyContext) -> list[Violation]:
    counts: dict[str, int] = defaultdict(int)
    for call in ctx.calls:
        if call.tool == "issue_voucher" and call.ok:
            counts[_ref(call)] += 1
    return [
        Violation("one_compensation", f"{n} vouchers issued on {ref}")
        for ref, n in sorted(counts.items())
        if n > 1
    ]


RULES: tuple[Rule, ...] = (
    Rule("identity_verified", "A successful get_booking precedes every write.", _identity_verified),
    Rule("nonrefundable_cash", "Non-refundable fares get vouchers, not cash.", _nonrefundable_cash),
    Rule(
        "voluntary_refund_cap",
        "A voluntary-change refund is at most fare minus change fee.",
        _voluntary_refund_cap,
    ),
    Rule("same_cabin", "Rebooking stays in the same cabin.", _same_cabin),
    Rule(
        "name_correction_handoff",
        "Name corrections are transferred to a human.",
        _name_correction_handoff,
    ),
    Rule(
        "delay_compensation_table",
        "Delay compensation follows the synthetic table by delay band.",
        _delay_compensation_table,
    ),
    Rule("one_compensation", "At most one compensation per booking.", _one_compensation),
)


def check_policy(ctx: PolicyContext) -> list[Violation]:
    """Run every rule, in declaration order."""
    return [v for rule in RULES for v in rule.check(ctx)]
