import pytest
from conftest import by_scenario

from endstate.env.db import SeatError, diff_states, diffs_equal
from endstate.env.tools import TOOL_SPECS, Toolbox
from endstate.tasks import REFUND_NONREFUNDABLE


def test_snapshot_restore_roundtrip(db):
    work = db.copy()
    snap = work.snapshot()
    Toolbox(work).execute("transfer_to_human", {"reason": "x"})
    work.flights["FL0001"].seats["economy"] = 0
    work.restore(snap)
    assert work.canonical() == db.canonical()
    assert work.handoffs == []


def test_copy_is_independent(db):
    work = db.copy()
    ref = next(iter(work.bookings))
    work.bookings[ref].status = "cancelled"
    work.flights["FL0001"].seats["economy"] += 5
    assert db.bookings[ref].status == "confirmed"
    assert db.canonical() != work.canonical()


def test_seats_never_negative(db):
    work = db.copy()
    work.flights["FL0001"].seats["business"] = 0
    with pytest.raises(SeatError):
        work.adjust_seats("FL0001", "business", -1)
    assert work.flights["FL0001"].seats["business"] == 0


def test_diff_ignores_generated_ids_and_timestamps(db):
    ref = first_ref(db)
    a, b = db.copy(), db.copy()
    b.clock = 40  # a different simulated time, so a different timestamp
    Toolbox(a).execute("issue_refund", {"booking_ref": ref, "amount": 50})
    Toolbox(b).execute("issue_refund", {"booking_ref": ref, "amount": 50})
    b.refunds[0].refund_id = "RF-99999"
    assert a.refunds[0].created_at != b.refunds[0].created_at
    assert a.refunds[0].refund_id != b.refunds[0].refund_id
    base = db.canonical()
    assert diffs_equal(diff_states(base, a.canonical()), diff_states(base, b.canonical()))


def first_ref(db) -> str:
    return next(iter(db.bookings))


def test_diff_detects_changed_refund_amount(db):
    ref = first_ref(db)
    a, b = db.copy(), db.copy()
    Toolbox(a).execute("issue_refund", {"booking_ref": ref, "amount": 100})
    Toolbox(b).execute("issue_refund", {"booking_ref": ref, "amount": 90})
    base = db.canonical()
    assert not diffs_equal(diff_states(base, a.canonical()), diff_states(base, b.canonical()))


def test_rebook_onto_full_cabin_returns_no_seats_and_changes_nothing(db):
    work = db.copy()
    booking = next(b for b in work.bookings.values() if b.cabin == "economy")
    target = next(f for f in work.flights.values() if f.flight_id != booking.flight_id)
    target.seats["economy"] = 0
    before = {fid: dict(f.seats) for fid, f in work.flights.items()}
    result = Toolbox(work).execute(
        "rebook", {"booking_ref": booking.ref, "flight_id": target.flight_id}
    )
    assert not result.ok and result.error.value == "NO_SEATS"
    assert {fid: dict(f.seats) for fid, f in work.flights.items()} == before
    assert booking.flight_id != target.flight_id


def test_total_seats_conserved_across_rebooks(db):
    work = db.copy()
    box = Toolbox(work)
    total = work.total_seats()
    moved = 0
    flights = list(work.flights.values())
    for booking in list(work.bookings.values())[:25]:
        target = next(
            f for f in flights if f.seats[booking.cabin] > 0 and f.flight_id != booking.flight_id
        )
        assert box.execute("rebook", {"booking_ref": booking.ref, "flight_id": target.flight_id}).ok
        moved += 1
        assert work.total_seats() == total
    assert moved == 25


def test_schemas_are_generated_from_pydantic_models(db):
    schemas = Toolbox.schemas()
    assert [s["function"]["name"] for s in schemas] == list(TOOL_SPECS)
    rebook = next(s for s in schemas if s["function"]["name"] == "rebook")["function"]["parameters"]
    assert set(rebook["required"]) == {"booking_ref", "flight_id"}
    assert "cabin" in rebook["properties"]
    assert rebook["additionalProperties"] is False


def test_invalid_arguments_are_typed_errors(db):
    box = Toolbox(db.copy())
    bad = box.execute("issue_refund", {"booking_ref": "X", "amount": -5})
    assert bad.error.value == "INVALID_ARGUMENTS" and "amount" in bad.message
    extra = box.execute("cancel_booking", {"booking_ref": "X", "force": True})
    assert extra.error.value == "INVALID_ARGUMENTS"
    assert box.execute("teleport", {}).error.value == "UNKNOWN_TOOL"
    assert (
        box.execute(
            "search_flights",
            {"origin": "SFO", "destination": "LAX", "date": "tomorrow", "cabin": "economy"},
        ).error.value
        == "INVALID_ARGUMENTS"
    )


def test_get_booking_requires_matching_last_name(db):
    work = db.copy()
    booking = next(iter(work.bookings.values()))
    last = work.passengers[booking.passenger_id].last_name
    box = Toolbox(work)
    assert (
        box.execute("get_booking", {"booking_ref": booking.ref, "last_name": "Nobody"}).error.value
        == "AUTH_FAILED"
    )
    assert (
        box.execute("get_booking", {"booking_ref": "ZZZZZZ", "last_name": last}).error.value
        == "BOOKING_NOT_FOUND"
    )
    ok = box.execute("get_booking", {"booking_ref": booking.ref.lower(), "last_name": last.upper()})
    assert ok.ok and ok.data["fare_paid"] == booking.fare_paid


def test_refund_cannot_exceed_amount_paid_but_tools_do_not_enforce_policy(tasks, db):
    task = by_scenario(tasks, REFUND_NONREFUNDABLE)[0]
    work = db.copy()
    box = Toolbox(work)
    ref = task.booking_ref
    paid = work.bookings[ref].fare_paid
    assert (
        box.execute("issue_refund", {"booking_ref": ref, "amount": paid + 0.01}).error.value
        == "EXCEEDS_PAID"
    )
    # A cash refund on a non-refundable fare is a policy violation, not a physical one.
    assert box.execute("issue_refund", {"booking_ref": ref, "amount": paid}).ok
    assert (
        box.execute("issue_refund", {"booking_ref": ref, "amount": 1}).error.value == "EXCEEDS_PAID"
    )


def test_cancel_releases_seat_once(db):
    work = db.copy()
    booking = next(iter(work.bookings.values()))
    free = work.flights[booking.flight_id].seats[booking.cabin]
    box = Toolbox(work)
    assert box.execute("cancel_booking", {"booking_ref": booking.ref}).ok
    assert work.flights[booking.flight_id].seats[booking.cabin] == free + 1
    assert (
        box.execute("cancel_booking", {"booking_ref": booking.ref}).error.value
        == "BOOKING_CANCELLED"
    )
    assert work.flights[booking.flight_id].seats[booking.cabin] == free + 1
