"""The agent's tools.

Each tool has a pydantic argument model, and its JSON Schema is generated from that model.
Tools enforce physical invariants only: seats never go negative and a refund never exceeds
what was paid. Business policy (fare rules) lives in ``endstate.policy``, so an agent that
breaks a rule still gets a successful tool result and the grader is what notices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from endstate.env.db import Database, Handoff, Refund, Voucher, money


class ErrorCode(StrEnum):
    UNKNOWN_TOOL = "UNKNOWN_TOOL"
    INVALID_ARGUMENTS = "INVALID_ARGUMENTS"
    BOOKING_NOT_FOUND = "BOOKING_NOT_FOUND"
    AUTH_FAILED = "AUTH_FAILED"
    FLIGHT_NOT_FOUND = "FLIGHT_NOT_FOUND"
    NO_SEATS = "NO_SEATS"
    BOOKING_CANCELLED = "BOOKING_CANCELLED"
    SAME_FLIGHT = "SAME_FLIGHT"
    EXCEEDS_PAID = "EXCEEDS_PAID"


class Cabin(StrEnum):
    economy = "economy"
    premium = "premium"
    business = "business"


class _Args(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GetBookingArgs(_Args):
    booking_ref: str = Field(description="Six-character booking reference.")
    last_name: str = Field(description="Passenger last name as on the booking.")


class SearchFlightsArgs(_Args):
    origin: str = Field(min_length=3, max_length=3, description="Origin airport code.")
    destination: str = Field(min_length=3, max_length=3, description="Destination airport code.")
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$", description="Departure date, YYYY-MM-DD.")
    cabin: Cabin = Field(description="Cabin that needs a free seat.")


class RebookArgs(_Args):
    booking_ref: str
    flight_id: str = Field(description="Flight to move the booking to.")
    cabin: Cabin | None = Field(
        default=None, description="Cabin on the new flight. Defaults to the booking's cabin."
    )


class CancelBookingArgs(_Args):
    booking_ref: str


class IssueRefundArgs(_Args):
    booking_ref: str
    amount: float = Field(gt=0, description="Cash refund in dollars to the original payment.")


class IssueVoucherArgs(_Args):
    booking_ref: str
    amount: float = Field(gt=0, description="Travel voucher value in dollars.")


class TransferToHumanArgs(_Args):
    reason: str = Field(min_length=1, description="Why the case needs a human agent.")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    model: type[BaseModel]
    writes: bool


TOOL_SPECS: dict[str, ToolSpec] = {
    spec.name: spec
    for spec in (
        ToolSpec(
            "get_booking",
            "Look up a booking. Succeeds only when the last name matches the booking.",
            GetBookingArgs,
            False,
        ),
        ToolSpec(
            "search_flights",
            "List scheduled flights on a route and date that have a free seat in a cabin.",
            SearchFlightsArgs,
            False,
        ),
        ToolSpec("rebook", "Move a booking to another flight.", RebookArgs, True),
        ToolSpec(
            "cancel_booking", "Cancel a booking and release its seat.", CancelBookingArgs, True
        ),
        ToolSpec("issue_refund", "Refund cash to the original payment.", IssueRefundArgs, True),
        ToolSpec("issue_voucher", "Issue a travel voucher for a booking.", IssueVoucherArgs, True),
        ToolSpec(
            "transfer_to_human",
            "Hand the conversation to a human agent.",
            TransferToHumanArgs,
            False,
        ),
    )
}
WRITE_TOOLS = frozenset(name for name, spec in TOOL_SPECS.items() if spec.writes)


@dataclass
class ToolResult:
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: ErrorCode | None = None
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "data": self.data,
            "error": self.error.value if self.error else None,
            "message": self.message,
        }


def _fail(code: ErrorCode, message: str) -> ToolResult:
    return ToolResult(ok=False, error=code, message=message)


class Toolbox:
    """Executes tool calls against a ``Database``."""

    def __init__(self, db: Database) -> None:
        self.db = db

    @staticmethod
    def schemas() -> list[dict[str, Any]]:
        """OpenAI function-calling specs generated from the pydantic models."""
        out = []
        for spec in TOOL_SPECS.values():
            schema = spec.model.model_json_schema()
            schema.pop("title", None)
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": spec.name,
                        "description": spec.description,
                        "parameters": schema,
                    },
                }
            )
        return out

    def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        spec = TOOL_SPECS.get(name)
        if spec is None:
            return _fail(ErrorCode.UNKNOWN_TOOL, f"no such tool: {name}")
        try:
            parsed = spec.model.model_validate(args)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(p) for p in first["loc"]) or "arguments"
            return _fail(ErrorCode.INVALID_ARGUMENTS, f"{where}: {first['msg']}")
        return getattr(self, f"_{name}")(parsed)

    # ------------------------------------------------------------------ reads
    def _flight_view(self, flight_id: str) -> dict[str, Any]:
        f = self.db.flights[flight_id]
        return {
            "flight_id": f.flight_id,
            "origin": f.origin,
            "destination": f.destination,
            "date": f.date,
            "depart": f.depart,
            "flight_status": f.status,
            "delay_minutes": f.delay_minutes,
        }

    def _get_booking(self, a: GetBookingArgs) -> ToolResult:
        booking = self.db.bookings.get(a.booking_ref.strip().upper())
        if booking is None:
            return _fail(ErrorCode.BOOKING_NOT_FOUND, f"no booking {a.booking_ref}")
        passenger = self.db.passengers[booking.passenger_id]
        if passenger.last_name.casefold() != a.last_name.strip().casefold():
            return _fail(ErrorCode.AUTH_FAILED, "last name does not match the booking")
        data = {
            "booking_ref": booking.ref,
            "first_name": passenger.first_name,
            "last_name": passenger.last_name,
            "cabin": booking.cabin,
            "fare_class": booking.fare_class,
            "fare_paid": money(booking.fare_paid),
            "refundable": booking.refundable,
            "change_fee": money(booking.change_fee),
            "status": booking.status,
            "refunds_issued": money(
                sum(r.amount for r in self.db.refunds if r.booking_ref == booking.ref)
            ),
            "vouchers_issued": money(
                sum(v.amount for v in self.db.vouchers if v.booking_ref == booking.ref)
            ),
            **self._flight_view(booking.flight_id),
        }
        return ToolResult(ok=True, data=data)

    def _search_flights(self, a: SearchFlightsArgs) -> ToolResult:
        rows = []
        for f in self.db.flights.values():
            if (f.origin, f.destination, f.date) != (
                a.origin.upper(),
                a.destination.upper(),
                a.date,
            ):
                continue
            if f.status != "scheduled" or f.seats.get(a.cabin.value, 0) <= 0:
                continue
            rows.append(
                {
                    "flight_id": f.flight_id,
                    "depart": f.depart,
                    "cabin": a.cabin.value,
                    "seats_available": f.seats[a.cabin.value],
                    "price": money(f.fares[a.cabin.value]),
                }
            )
        rows.sort(key=lambda r: (r["depart"], r["flight_id"]))
        return ToolResult(ok=True, data={"flights": rows})

    # ------------------------------------------------------------------ writes
    def _booking(self, ref: str):
        return self.db.bookings.get(ref.strip().upper())

    def _rebook(self, a: RebookArgs) -> ToolResult:
        booking = self._booking(a.booking_ref)
        if booking is None:
            return _fail(ErrorCode.BOOKING_NOT_FOUND, f"no booking {a.booking_ref}")
        if booking.status == "cancelled":
            return _fail(ErrorCode.BOOKING_CANCELLED, "booking is cancelled")
        flight = self.db.flights.get(a.flight_id)
        if flight is None:
            return _fail(ErrorCode.FLIGHT_NOT_FOUND, f"no flight {a.flight_id}")
        cabin = (a.cabin or Cabin(booking.cabin)).value
        if flight.flight_id == booking.flight_id and cabin == booking.cabin:
            return _fail(ErrorCode.SAME_FLIGHT, "booking is already on that flight and cabin")
        if flight.seats.get(cabin, 0) <= 0:
            return _fail(ErrorCode.NO_SEATS, f"{flight.flight_id} has no {cabin} seats left")
        self.db.adjust_seats(booking.flight_id, booking.cabin, +1)
        self.db.adjust_seats(flight.flight_id, cabin, -1)
        previous = {"flight_id": booking.flight_id, "cabin": booking.cabin}
        booking.flight_id, booking.cabin = flight.flight_id, cabin
        return ToolResult(
            ok=True,
            data={
                "booking_ref": booking.ref,
                "previous": previous,
                **self._flight_view(flight.flight_id),
                "cabin": cabin,
            },
        )

    def _cancel_booking(self, a: CancelBookingArgs) -> ToolResult:
        booking = self._booking(a.booking_ref)
        if booking is None:
            return _fail(ErrorCode.BOOKING_NOT_FOUND, f"no booking {a.booking_ref}")
        if booking.status == "cancelled":
            return _fail(ErrorCode.BOOKING_CANCELLED, "booking is already cancelled")
        booking.status = "cancelled"
        self.db.adjust_seats(booking.flight_id, booking.cabin, +1)
        return ToolResult(ok=True, data={"booking_ref": booking.ref, "status": "cancelled"})

    def _issue_refund(self, a: IssueRefundArgs) -> ToolResult:
        booking = self._booking(a.booking_ref)
        if booking is None:
            return _fail(ErrorCode.BOOKING_NOT_FOUND, f"no booking {a.booking_ref}")
        already = sum(r.amount for r in self.db.refunds if r.booking_ref == booking.ref)
        if money(a.amount) > money(booking.fare_paid - already):
            return _fail(ErrorCode.EXCEEDS_PAID, "refund would exceed the amount paid")
        refund = Refund(
            self.db.next_id("RF", len(self.db.refunds)),
            booking.ref,
            money(a.amount),
            self.db.tick(),
        )
        self.db.refunds.append(refund)
        return ToolResult(
            ok=True,
            data={
                "refund_id": refund.refund_id,
                "booking_ref": booking.ref,
                "amount": refund.amount,
            },
        )

    def _issue_voucher(self, a: IssueVoucherArgs) -> ToolResult:
        booking = self._booking(a.booking_ref)
        if booking is None:
            return _fail(ErrorCode.BOOKING_NOT_FOUND, f"no booking {a.booking_ref}")
        if money(a.amount) > money(booking.fare_paid):
            return _fail(ErrorCode.EXCEEDS_PAID, "voucher would exceed the amount paid")
        voucher = Voucher(
            self.db.next_id("VC", len(self.db.vouchers)),
            booking.ref,
            money(a.amount),
            self.db.tick(),
        )
        self.db.vouchers.append(voucher)
        return ToolResult(
            ok=True,
            data={
                "voucher_id": voucher.voucher_id,
                "booking_ref": booking.ref,
                "amount": voucher.amount,
            },
        )

    def _transfer_to_human(self, a: TransferToHumanArgs) -> ToolResult:
        handoff = Handoff(self.db.next_id("HO", len(self.db.handoffs)), a.reason, self.db.tick())
        self.db.handoffs.append(handoff)
        return ToolResult(ok=True, data={"handoff_id": handoff.handoff_id, "transferred": True})
