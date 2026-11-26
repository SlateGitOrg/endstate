"""In-memory booking database with snapshot/restore and a canonical form for diffs.

The database is plain dataclasses. Everything a grader needs is derived from two
operations: ``canonical()`` (a normalised, id-free view of the state) and
``diff_states`` (what changed between two canonical views).
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

CABINS = ("economy", "premium", "business")

_CLOCK_START = datetime(2026, 3, 1, 9, 0, 0)


@dataclass
class Airport:
    code: str
    city: str


@dataclass
class Flight:
    flight_id: str
    origin: str
    destination: str
    date: str
    depart: str
    status: str = "scheduled"  # scheduled | cancelled | departed
    delay_minutes: int = 0
    seats: dict[str, int] = field(default_factory=dict)  # seats still available, by cabin
    fares: dict[str, float] = field(default_factory=dict)  # current fare, by cabin


@dataclass
class Passenger:
    passenger_id: str
    first_name: str
    last_name: str


@dataclass
class Booking:
    ref: str
    passenger_id: str
    flight_id: str
    cabin: str
    fare_class: str
    fare_paid: float
    refundable: bool
    change_fee: float
    status: str = "confirmed"  # confirmed | cancelled


@dataclass
class Payment:
    payment_id: str
    booking_ref: str
    amount: float
    created_at: str


@dataclass
class Refund:
    refund_id: str
    booking_ref: str
    amount: float
    created_at: str


@dataclass
class Voucher:
    voucher_id: str
    booking_ref: str
    amount: float
    created_at: str


@dataclass
class Handoff:
    handoff_id: str
    reason: str
    created_at: str


class SeatError(ValueError):
    """Raised when a seat count would go negative."""


def money(value: float) -> float:
    return round(float(value), 2)


class Database:
    """Mutable booking state. Use ``snapshot``/``restore`` to rewind it."""

    def __init__(self) -> None:
        self.airports: dict[str, Airport] = {}
        self.flights: dict[str, Flight] = {}
        self.passengers: dict[str, Passenger] = {}
        self.bookings: dict[str, Booking] = {}
        self.payments: list[Payment] = []
        self.refunds: list[Refund] = []
        self.vouchers: list[Voucher] = []
        self.handoffs: list[Handoff] = []
        self.clock: int = 0

    # ------------------------------------------------------------------ ids / time
    def tick(self) -> str:
        """Advance the simulated clock one minute and return its timestamp."""
        self.clock += 1
        return (_CLOCK_START + timedelta(minutes=self.clock)).isoformat()

    def next_id(self, prefix: str, existing: int) -> str:
        return f"{prefix}-{existing + 1:05d}"

    # ------------------------------------------------------------------ seats
    def adjust_seats(self, flight_id: str, cabin: str, delta: int) -> None:
        flight = self.flights[flight_id]
        if flight.seats.get(cabin, 0) + delta < 0:
            raise SeatError(f"{flight_id}/{cabin} would go negative")
        flight.seats[cabin] = flight.seats.get(cabin, 0) + delta

    def total_seats(self) -> int:
        return sum(sum(f.seats.values()) for f in self.flights.values())

    # ------------------------------------------------------------------ snapshot
    def to_dict(self) -> dict[str, Any]:
        return {
            "airports": [asdict(a) for a in self.airports.values()],
            "flights": [asdict(f) for f in self.flights.values()],
            "passengers": [asdict(p) for p in self.passengers.values()],
            "bookings": [asdict(b) for b in self.bookings.values()],
            "payments": [asdict(p) for p in self.payments],
            "refunds": [asdict(r) for r in self.refunds],
            "vouchers": [asdict(v) for v in self.vouchers],
            "handoffs": [asdict(h) for h in self.handoffs],
            "clock": self.clock,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Database:
        db = cls()
        db._load(data)
        return db

    def _load(self, data: dict[str, Any]) -> None:
        data = copy.deepcopy(data)
        self.airports = {a["code"]: Airport(**a) for a in data["airports"]}
        self.flights = {f["flight_id"]: Flight(**f) for f in data["flights"]}
        self.passengers = {p["passenger_id"]: Passenger(**p) for p in data["passengers"]}
        self.bookings = {b["ref"]: Booking(**b) for b in data["bookings"]}
        self.payments = [Payment(**p) for p in data["payments"]]
        self.refunds = [Refund(**r) for r in data["refunds"]]
        self.vouchers = [Voucher(**v) for v in data["vouchers"]]
        self.handoffs = [Handoff(**h) for h in data["handoffs"]]
        self.clock = int(data.get("clock", 0))

    def snapshot(self) -> dict[str, Any]:
        return self.to_dict()

    def restore(self, snap: dict[str, Any]) -> None:
        self._load(snap)

    def copy(self) -> Database:
        """A fast independent copy. Airports, passengers and ledger rows are never mutated by
        the tools, so they are shared; flights and bookings are copied."""
        new = Database()
        new.airports = self.airports
        new.passengers = self.passengers
        new.flights = {
            fid: replace(f, seats=dict(f.seats), fares=dict(f.fares))
            for fid, f in self.flights.items()
        }
        new.bookings = {ref: replace(b) for ref, b in self.bookings.items()}
        new.payments = list(self.payments)
        new.refunds = list(self.refunds)
        new.vouchers = list(self.vouchers)
        new.handoffs = list(self.handoffs)
        new.clock = self.clock
        return new

    # ------------------------------------------------------------------ files
    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), sort_keys=True, indent=1) + "\n")

    @classmethod
    def load(cls, path: Path) -> Database:
        return cls.from_dict(json.loads(path.read_text()))

    # ------------------------------------------------------------------ canonical form
    def canonical(self) -> dict[str, Any]:
        """A normalised view of the state with generated ids and timestamps removed."""

        def entries(items: list[Any]) -> list[dict[str, Any]]:
            rows = [{"booking_ref": i.booking_ref, "amount": money(i.amount)} for i in items]
            return sorted(rows, key=lambda r: (r["booking_ref"], r["amount"]))

        return {
            "bookings": {
                ref: {
                    "flight_id": b.flight_id,
                    "cabin": b.cabin,
                    "status": b.status,
                    "fare_class": b.fare_class,
                    "fare_paid": money(b.fare_paid),
                    "refundable": b.refundable,
                    "change_fee": money(b.change_fee),
                }
                for ref, b in sorted(self.bookings.items())
            },
            "seats": {
                fid: dict(sorted(f.seats.items())) for fid, f in sorted(self.flights.items())
            },
            "payments": entries(self.payments),
            "refunds": entries(self.refunds),
            "vouchers": entries(self.vouchers),
            "handoffs": len(self.handoffs),
        }

    def digest(self) -> str:
        import hashlib

        blob = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------- diffs
def _multiset_added(before: list[dict[str, Any]], after: list[dict[str, Any]]):
    def key(row: dict[str, Any]) -> tuple[str, float]:
        return (row["booking_ref"], row["amount"])

    added = Counter(map(key, after)) - Counter(map(key, before))
    removed = Counter(map(key, before)) - Counter(map(key, after))
    rows = [
        {"booking_ref": r, "amount": a} for (r, a), n in sorted(added.items()) for _ in range(n)
    ]
    gone = [
        {"booking_ref": r, "amount": a} for (r, a), n in sorted(removed.items()) for _ in range(n)
    ]
    return rows, gone


def diff_states(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    """Return what changed between two canonical states.

    Shape (every key is always present so diffs compare with ``==``)::

        {"bookings": {ref: {field: new_value}},
         "seats": {flight_id: {cabin: delta}},
         "payments"/"refunds"/"vouchers": [added rows],
         "handoffs": added count}
    """
    bookings: dict[str, dict[str, Any]] = {}
    for ref, new in after["bookings"].items():
        old = before["bookings"].get(ref)
        if old is None:
            bookings[ref] = dict(new)
            continue
        changed = {k: v for k, v in new.items() if old.get(k) != v}
        if changed:
            bookings[ref] = changed
    seats: dict[str, dict[str, int]] = {}
    for fid, cabins in after["seats"].items():
        old = before["seats"].get(fid, {})
        deltas = {c: n - old.get(c, 0) for c, n in cabins.items() if n != old.get(c, 0)}
        if deltas:
            seats[fid] = deltas
    out: dict[str, Any] = {"bookings": bookings, "seats": seats}
    for key in ("payments", "refunds", "vouchers"):
        added, _ = _multiset_added(before[key], after[key])
        out[key] = added
    out["handoffs"] = after["handoffs"] - before["handoffs"]
    return out


def normalize_diff(diff: dict[str, Any]) -> dict[str, Any]:
    """Fill defaults and round money so diffs from YAML and from the grader compare equal."""
    rows = {
        key: sorted(
            (
                {"booking_ref": r["booking_ref"], "amount": money(r["amount"])}
                for r in diff.get(key, [])
            ),
            key=lambda r: (r["booking_ref"], r["amount"]),
        )
        for key in ("payments", "refunds", "vouchers")
    }
    bookings = {
        ref: {k: (money(v) if k in ("fare_paid", "change_fee") else v) for k, v in fields.items()}
        for ref, fields in sorted(diff.get("bookings", {}).items())
    }
    seats = {fid: dict(sorted(c.items())) for fid, c in sorted(diff.get("seats", {}).items())}
    return {"bookings": bookings, "seats": seats, **rows, "handoffs": int(diff.get("handoffs", 0))}


def diffs_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    return normalize_diff(a) == normalize_diff(b)


def describe_diff(diff: dict[str, Any]) -> list[str]:
    """Human-readable lines for a diff, used by ``replay``."""
    diff = normalize_diff(diff)
    lines: list[str] = []
    for ref, fields in diff["bookings"].items():
        lines.append(f"booking {ref}: " + ", ".join(f"{k} -> {v}" for k, v in fields.items()))
    for fid, cabins in diff["seats"].items():
        lines.append(f"seats {fid}: " + ", ".join(f"{c} {d:+d}" for c, d in cabins.items()))
    for key in ("payments", "refunds", "vouchers"):
        for row in diff[key]:
            lines.append(f"+ {key[:-1]} {row['booking_ref']} {row['amount']:.2f}")
    if diff["handoffs"]:
        lines.append(f"+ {diff['handoffs']} handoff to a human")
    return lines or ["(no change)"]
