"""Seeded generator for the travel database and the 60 tasks.

Synthetic properties (all deterministic in the seed):

* 12 airports, 300 flights over 14 days. Flights come in 25 routes x 4 days x 3 departures
  (07:05, 13:20, 19:40), so every route-day has two alternatives to a given flight.
* 200 passengers and 240 bookings, exactly 40% of them on non-refundable fares.
* 60 tasks, 10 for each of 6 scenario types. 8 tasks accept two different goal diffs (two
  equivalent alternative flights) and in 4 tasks the flight the customer asks for is full, so
  the correct end state is the other flight.

Goal diffs are not hand-written: each is produced by running a reference action list through
the real tools on a copy of the initial database.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from endstate import phrasing
from endstate.env.db import (
    CABINS,
    Airport,
    Booking,
    Database,
    Flight,
    Passenger,
    Payment,
    diff_states,
    normalize_diff,
)
from endstate.env.tools import Toolbox
from endstate.policy import compensation_for_delay
from endstate.tasks import (
    CANCELLED_REBOOK,
    MISSED_CONNECTION,
    NAME_CORRECTION,
    REFUND_NONREFUNDABLE,
    REFUND_REFUNDABLE,
    VOLUNTARY_CHANGE,
    Persona,
    Task,
    save_tasks,
)

AIRPORTS = [
    ("SFO", "San Francisco"),
    ("LAX", "Los Angeles"),
    ("SEA", "Seattle"),
    ("DEN", "Denver"),
    ("ORD", "Chicago"),
    ("DFW", "Dallas"),
    ("ATL", "Atlanta"),
    ("JFK", "New York"),
    ("BOS", "Boston"),
    ("MIA", "Miami"),
    ("PHX", "Phoenix"),
    ("MSP", "Minneapolis"),
]
DATES = [f"2026-03-{day:02d}" for day in range(2, 16)]  # 14 days
DEPARTS = ("07:05", "13:20", "19:40")
N_ROUTES, DAYS_PER_ROUTE = 25, 4
N_PASSENGERS, N_BOOKINGS, NONREFUNDABLE_SHARE = 200, 240, 0.4
CABIN_MULTIPLIER = {"economy": 1.0, "premium": 1.7, "business": 3.1}
SEAT_RANGE = {"economy": (18, 80), "premium": (3, 14), "business": (2, 8)}
FIRST_NAMES = (
    "Ava Ben Chloe Dev Elena Farid Grace Hugo Imani Jonas Keiko Leo Maya Nico Olga Pablo Quinn "
    "Rosa Sami Tara Uma Victor Wren Xavier Yara Zane Amara Bruno Carla Dmitri Esther Felix Gita "
    "Hana Ivan Jade Kofi Lena Marco Nadia"
).split()
LAST_NAMES = (
    "Abbott Barros Castillo Dumont Eriksen Fontaine Garza Hayashi Iqbal Jansen Kowalski Lindqvist "
    "Moreau Nakamura Okafor Petrov Quintero Rahman Santos Takahashi Underwood Vasquez Weber Xu "
    "Yilmaz Zhou Alvarez Brennan Chaudhry Delgado Ellison Farouk Gallagher Haddad Ibarra Jovanovic "
    "Keller Laurent Mbeki Novak Ortega Pereira Quist Reyes Schmidt Torres Ueda Valdez Whitaker "
    "Yamada Zielinski Aguirre Bianchi Cardenas Donovan Esposito Fischer Greco Holm Ivanov Jorgensen"
).split()
REF_ALPHABET = np.array(list("ABCDEFGHJKLMNPQRSTUVWXYZ23456789"))

# Task layout: which of the 10 tasks per scenario are dual-goal or ask for a full flight.
FULL_REQUEST = {CANCELLED_REBOOK: range(0, 4)}
DUAL_GOAL = {
    CANCELLED_REBOOK: range(4, 6),
    MISSED_CONNECTION: range(0, 2),
    VOLUNTARY_CHANGE: range(0, 4),
}
DELAYS = (130, 150, 180, 200, 250, 300, 390, 470, 480, 560)  # minutes, spans all three bands
CHANGE_FEES = (50.0, 75.0, 100.0)


class _Builder:
    def __init__(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.db = Database()
        self.route_days: list[list[list[str]]] = []  # route -> day -> flight ids (3 slots)
        self.task_flights: set[str] = set()
        self.refs: set[str] = set()

    # ------------------------------------------------------------------ primitives
    def ref(self) -> str:
        while True:
            ref = "".join(self.rng.choice(REF_ALPHABET, 6))
            if ref not in self.refs:
                self.refs.add(ref)
                return ref

    def pick(self, options: Any) -> Any:
        return options[int(self.rng.integers(len(options)))]

    def flight(self, fid: str) -> Flight:
        return self.db.flights[fid]

    # ------------------------------------------------------------------ network
    def build_network(self) -> None:
        for code, city in AIRPORTS:
            self.db.airports[code] = Airport(code, city)
        pairs = [(a, b) for a, _ in AIRPORTS for b, _ in AIRPORTS if a != b]
        chosen = self.rng.choice(len(pairs), N_ROUTES, replace=False)
        n = 0
        for idx in chosen:
            origin, dest = pairs[int(idx)]
            base = float(self.rng.integers(140, 320))
            days = sorted(
                int(d) for d in self.rng.choice(len(DATES), DAYS_PER_ROUTE, replace=False)
            )
            by_day = []
            for d in days:
                ids = []
                for depart in DEPARTS:
                    n += 1
                    fid = f"FL{n:04d}"
                    economy = round(base * float(self.rng.uniform(0.92, 1.08)))
                    seats = {
                        c: int(self.rng.integers(SEAT_RANGE[c][0], SEAT_RANGE[c][1] + 1))
                        for c in CABINS
                    }
                    self.db.flights[fid] = Flight(
                        fid,
                        origin,
                        dest,
                        DATES[d],
                        depart,
                        seats=seats,
                        fares={c: float(round(economy * CABIN_MULTIPLIER[c])) for c in CABINS},
                    )
                    ids.append(fid)
                by_day.append(ids)
            self.route_days.append(by_day)

    # ------------------------------------------------------------------ people / bookings
    def build_passengers(self) -> None:
        names: set[tuple[str, str]] = set()
        lasts = [str(n) for n in self.rng.permutation(LAST_NAMES)]
        for i in range(N_PASSENGERS):
            # The first 60 (task passengers) get distinct last names.
            last = lasts[i] if i < len(lasts) else str(self.pick(LAST_NAMES))
            first = str(self.pick(FIRST_NAMES))
            while (first, last) in names:
                first = str(self.pick(FIRST_NAMES))
            names.add((first, last))
            pid = f"P{i + 1:04d}"
            self.db.passengers[pid] = Passenger(pid, first, last)

    def terms(self, refundable: bool, *, fee: float | None = None) -> tuple[str, float]:
        if refundable:
            if fee is None:
                fee = 0.0 if self.rng.random() < 0.5 else self.pick(CHANGE_FEES)
            return ("flex" if fee == 0 else "standard"), fee
        return "saver", fee if fee is not None else self.pick((100.0, 150.0))

    def add_booking(
        self,
        passenger: int,
        fid: str,
        cabin: str,
        fare_paid: float,
        refundable: bool,
        fee: float | None = None,
    ) -> Booking:
        fare_class, change_fee = self.terms(refundable, fee=fee)
        ref = self.ref()
        booking = Booking(
            ref,
            f"P{passenger + 1:04d}",
            fid,
            cabin,
            fare_class,
            float(fare_paid),
            refundable,
            change_fee,
        )
        self.db.bookings[ref] = booking
        self.db.payments.append(
            Payment(
                f"PM-{len(self.db.payments) + 1:05d}", ref, float(fare_paid), "2026-02-01T10:00:00"
            )
        )
        return booking

    # ------------------------------------------------------------------ helpers for tasks
    def set_free(self, fid: str, cabin: str, n: int) -> None:
        """Set a cabin's free seats and make sure every other cabin still has room."""
        f = self.flight(fid)
        f.seats[cabin] = n
        for other in CABINS:
            if other != cabin:
                f.seats[other] = max(f.seats[other], 3)
        self.task_flights.add(fid)

    def cabin_choice(self) -> str:
        return str(self.rng.choice(CABINS, p=[0.7, 0.2, 0.1]))


def _reference(
    scenario: str, t: dict[str, Any], *, flight: str | None = None, refund: float | None = None
) -> list[dict[str, Any]]:
    """The ideal action list for a task, used to derive its goal diff."""
    ref = t["booking_ref"]
    actions = [{"tool": "get_booking", "args": {"booking_ref": ref, "last_name": t["last_name"]}}]
    if scenario in (CANCELLED_REBOOK, MISSED_CONNECTION, VOLUNTARY_CHANGE):
        actions.append(
            {
                "tool": "search_flights",
                "args": {
                    "origin": t["origin"],
                    "destination": t["destination"],
                    "date": t["search_date"],
                    "cabin": t["cabin"],
                },
            }
        )
        actions.append(
            {
                "tool": "rebook",
                "args": {"booking_ref": ref, "flight_id": flight, "cabin": t["cabin"]},
            }
        )
        if scenario == MISSED_CONNECTION:
            amount = compensation_for_delay(t["delay_minutes"])
            actions.append(
                {"tool": "issue_voucher", "args": {"booking_ref": ref, "amount": amount}}
            )
        if scenario == VOLUNTARY_CHANGE:
            actions.append({"tool": "issue_refund", "args": {"booking_ref": ref, "amount": refund}})
    elif scenario == REFUND_REFUNDABLE:
        actions.append({"tool": "cancel_booking", "args": {"booking_ref": ref}})
        actions.append(
            {
                "tool": "issue_refund",
                "args": {"booking_ref": ref, "amount": t["fare_paid"] - t["change_fee"]},
            }
        )
    elif scenario == REFUND_NONREFUNDABLE:
        actions.append({"tool": "cancel_booking", "args": {"booking_ref": ref}})
        actions.append(
            {"tool": "issue_voucher", "args": {"booking_ref": ref, "amount": t["fare_paid"]}}
        )
    else:
        actions.append(
            {
                "tool": "transfer_to_human",
                "args": {"reason": "name correction requested by the passenger"},
            }
        )
    return actions


def _goal(initial: Database, actions: list[dict[str, Any]]) -> dict[str, Any]:
    db = initial.copy()
    box = Toolbox(db)
    for action in actions:
        result = box.execute(action["tool"], action["args"])
        if not result.ok:
            raise RuntimeError(f"reference action failed: {action} -> {result.message}")
    return normalize_diff(diff_states(initial.canonical(), db.canonical()))


def _transcript(
    scenario: str, t: dict[str, Any], flight: Flight | None, refund: float | None
) -> list[str]:
    if scenario == CANCELLED_REBOOK:
        end = phrasing.resolution_rebook(flight.flight_id, flight.depart, t["cabin"])
    elif scenario == MISSED_CONNECTION:
        end = phrasing.resolution_missed(
            flight.flight_id, flight.depart, t["cabin"], compensation_for_delay(t["delay_minutes"])
        )
    elif scenario == VOLUNTARY_CHANGE:
        end = phrasing.resolution_change(flight.flight_id, flight.depart, refund, t["change_fee"])
    elif scenario == REFUND_REFUNDABLE:
        end = phrasing.resolution_refund(t["fare_paid"] - t["change_fee"])
    elif scenario == REFUND_NONREFUNDABLE:
        end = phrasing.resolution_voucher(t["fare_paid"])
    else:
        end = phrasing.resolution_transfer()
    closing = "" if scenario == NAME_CORRECTION else f" {phrasing.CLOSING}"
    return [phrasing.opening_turn(), phrasing.found_turn(scenario), end + closing]


def generate(seed: int = 7) -> tuple[Database, list[Task]]:
    b = _Builder(seed)
    b.build_network()
    b.build_passengers()
    db = b.db
    specs: list[dict[str, Any]] = []  # task specs, resolved to Tasks after the db is final

    # --- 30 rebooking tasks live on dedicated route-days ------------------------------------
    for i in range(10):
        # cancelled-flight rebooking: route 10+i, day 0
        fa, fb, fc = b.route_days[10 + i][0]
        cabin = b.cabin_choice()
        full = i in FULL_REQUEST[CANCELLED_REBOOK]
        dual = i in DUAL_GOAL[CANCELLED_REBOOK]
        b.flight(fa).status = "cancelled"
        b.task_flights.add(fa)
        b.set_free(fb, cabin, 5 + i)
        b.set_free(fc, cabin, 6 if dual else 0)
        booking = b.add_booking(i, fa, cabin, b.flight(fa).fares[cabin], bool(b.rng.random() < 0.6))
        specs.append(
            {
                "scenario": CANCELLED_REBOOK,
                "booking": booking,
                "passenger": i,
                "alts": [fb, fc] if dual else [fb],
                "requested": fc if full else None,
                "dual": dual,
            }
        )
    for i in range(10):
        # missed connection: route 10+i, day 1; the booked flight left after a long delay
        fa, fb, fc = b.route_days[10 + i][1]
        cabin = b.cabin_choice()
        dual = i in DUAL_GOAL[MISSED_CONNECTION]
        flight = b.flight(fa)
        flight.status, flight.delay_minutes = "departed", DELAYS[i]
        b.task_flights.add(fa)
        b.set_free(fb, cabin, 4 + i)
        b.set_free(fc, cabin, 5 if dual else 0)
        booking = b.add_booking(
            10 + i, fa, cabin, max(flight.fares[cabin], 220.0), bool(b.rng.random() < 0.6)
        )
        specs.append(
            {
                "scenario": MISSED_CONNECTION,
                "booking": booking,
                "passenger": 10 + i,
                "alts": [fb, fc] if dual else [fb],
                "requested": None,
                "dual": dual,
            }
        )
    for i in range(10):
        # voluntary change: booked on route i day 0, wants day 1 of the same route
        fa = b.route_days[i][0][0]
        targets = b.route_days[i][1]
        cabin = b.cabin_choice()
        dual = i in DUAL_GOAL[VOLUNTARY_CHANGE]
        open_idx = [i % 3, (i + 1) % 3] if dual else [i % 3]
        price = float(round(b.flight(targets[0]).fares[cabin]))
        fee = CHANGE_FEES[i % 3]
        for idx, fid in enumerate(targets):
            b.flight(fid).fares[cabin] = price  # equivalent alternatives share a price
            b.set_free(fid, cabin, 4 + i if idx in open_idx else 0)
        b.task_flights.add(fa)
        fare_paid = price + fee + float(5 * int(b.rng.integers(6, 23)))
        booking = b.add_booking(20 + i, fa, cabin, fare_paid, True, fee=fee)
        specs.append(
            {
                "scenario": VOLUNTARY_CHANGE,
                "booking": booking,
                "passenger": 20 + i,
                "alts": [targets[idx] for idx in open_idx],
                "requested": None,
                "dual": dual,
                "price": price,
                "desired_date": b.flight(targets[0]).date,
            }
        )

    # --- 30 tasks on background flights -------------------------------------------------------
    used = {fid for days in b.route_days[:20] for day in days[:2] for fid in day}
    free_flights = [fid for fid in db.flights if fid not in used]
    for scenario, base in (
        (REFUND_REFUNDABLE, 30),
        (REFUND_NONREFUNDABLE, 40),
        (NAME_CORRECTION, 50),
    ):
        for i in range(10):
            fid = b.pick(free_flights)
            cabin = b.cabin_choice()
            fare = float(5 * int(b.rng.integers(36, 120)))
            if scenario == REFUND_REFUNDABLE:
                booking = b.add_booking(base + i, fid, cabin, fare, True, fee=0.0)
            elif scenario == REFUND_NONREFUNDABLE:
                booking = b.add_booking(base + i, fid, cabin, fare, False)
            else:
                booking = b.add_booking(base + i, fid, cabin, fare, bool(b.rng.random() < 0.4))
            specs.append({"scenario": scenario, "booking": booking, "passenger": base + i})

    # --- background bookings, 40% non-refundable overall -------------------------------------
    task_nonref = sum(1 for s in specs if not s["booking"].refundable)
    n_background = N_BOOKINGS - len(specs)
    target_nonref = round(N_BOOKINGS * NONREFUNDABLE_SHARE)
    flags = np.array(
        [True] * (target_nonref - task_nonref)
        + [False] * (n_background - (target_nonref - task_nonref))
    )
    flags = b.rng.permutation(flags)
    for j in range(n_background):
        passenger = 60 + j if 60 + j < N_PASSENGERS else int(b.rng.integers(N_PASSENGERS))
        fid = b.pick(free_flights)
        cabin = b.cabin_choice()
        fare = round(db.flights[fid].fares[cabin] * float(b.rng.uniform(0.9, 1.1)))
        b.add_booking(passenger, fid, cabin, float(fare), not bool(flags[j]))

    # --- tasks -------------------------------------------------------------------------------
    tasks: list[Task] = []
    for n, spec in enumerate(specs, start=1):
        tasks.append(_make_task(f"T{n:03d}", spec, db))
    return db, tasks


def _make_task(task_id: str, spec: dict[str, Any], db: Database) -> Task:
    scenario, booking = spec["scenario"], spec["booking"]
    passenger = db.passengers[booking.passenger_id]
    flight = db.flights[booking.flight_id]
    ticket = {
        "booking_ref": booking.ref,
        "last_name": passenger.last_name,
        "flight_id": flight.flight_id,
        "origin": flight.origin,
        "destination": flight.destination,
        "date": flight.date,
        "cabin": booking.cabin,
        "fare_paid": booking.fare_paid,
        "refundable": booking.refundable,
        "change_fee": booking.change_fee,
        "flight_status": flight.status,
        "delay_minutes": flight.delay_minutes,
    }
    t = dict(ticket, search_date=spec.get("desired_date", flight.date))
    name = f"{passenger.first_name} {passenger.last_name}"
    where = (
        f"flight {flight.flight_id} from {flight.origin} to {flight.destination} on {flight.date}"
    )
    requested = spec.get("requested")
    desired = spec.get("desired_date")
    extra = ""
    if scenario == CANCELLED_REBOOK:
        opening = (
            f"Hi, my {where} was cancelled and I am booked in {booking.cabin}. "
            "I need another flight that same day in the same cabin."
        )
        if requested:
            opening += f" I would like flight {requested} if it has room."
        if spec["dual"]:
            extra = " Either of the later flights that day is fine."
    elif scenario == MISSED_CONNECTION:
        opening = (
            f"Hi, I missed my connection: my {where} was held up for hours and I could not "
            "make it. Please put me on the next flight on that route the same day, in the "
            "same cabin."
        )
        if spec["dual"]:
            extra = " Either of the later flights is fine."
    elif scenario == VOLUNTARY_CHANGE:
        opening = (
            f"Hi, I would like to change my {where}. I would rather fly on {desired} on the "
            "same route and in the same cabin. I understand there is a change fee."
        )
        if spec["dual"]:
            extra = " Any departure that day works for me."
    elif scenario == REFUND_REFUNDABLE:
        opening = (
            f"Hi, my plans changed and I would like to cancel my {where} and get my money back."
        )
    elif scenario == REFUND_NONREFUNDABLE:
        opening = f"Hi, I need to cancel my {where} and I would like a refund to my card."
    else:
        opening = (
            f"Hi, there is a typo in my first name on my ticket for {where}. "
            f"Please correct it to {passenger.first_name}."
        )
    opening += extra

    alts = spec.get("alts", [])
    reference_actions: list[list[dict[str, Any]]] = []
    refund = None
    if scenario == VOLUNTARY_CHANGE:
        refund = round(booking.fare_paid - spec["price"] - booking.change_fee, 2)
    if scenario in (CANCELLED_REBOOK, MISSED_CONNECTION, VOLUNTARY_CHANGE):
        for alt in alts:
            reference_actions.append(_reference(scenario, t, flight=alt, refund=refund))
    else:
        reference_actions.append(_reference(scenario, t))
    goals = [_goal(db, actions) for actions in reference_actions]
    chosen = db.flights[alts[0]] if alts else None
    transcript = _transcript(scenario, t, chosen, refund)
    persona = Persona(
        name=name,
        booking_ref=booking.ref,
        last_name=passenger.last_name,
        opening=opening,
        ticket=ticket,
        requested_flight=requested,
        desired_date=desired,
    )
    return Task(task_id, scenario, persona, goals, transcript, reference_actions[0])


def write_seed(out: Path, seed: int = 7) -> tuple[Database, list[Task]]:
    out.mkdir(parents=True, exist_ok=True)
    db, tasks = generate(seed)
    db.save(out / "db.json")
    save_tasks(tasks, out / "tasks.yaml")
    return db, tasks
