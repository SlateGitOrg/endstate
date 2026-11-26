from collections import Counter

from conftest import context

from endstate.env.db import diff_states, diffs_equal
from endstate.generate import generate, write_seed
from endstate.policy import check_policy
from endstate.tasks import SCENARIOS, load_tasks


def test_seeded_database_sizes_and_mix(db):
    assert len(db.airports) == 12
    assert len(db.flights) == 300
    assert len({f.date for f in db.flights.values()}) <= 14
    assert len(db.passengers) == 200
    assert len(db.bookings) == 240
    nonref = sum(not b.refundable for b in db.bookings.values())
    assert nonref == 96  # exactly 40%
    assert all(min(f.seats.values()) >= 0 for f in db.flights.values())
    assert len(db.payments) == 240


def test_sixty_tasks_ten_per_scenario(tasks):
    assert len(tasks) == 60
    assert Counter(t.scenario for t in tasks) == {s: 10 for s in SCENARIOS}
    assert len({t.booking_ref for t in tasks}) == 60


def test_eight_dual_goal_tasks_and_four_full_flight_tasks(db, tasks):
    dual = [t for t in tasks if len(t.goals) == 2]
    assert len(dual) == 8
    full = [t for t in tasks if t.persona.requested_flight]
    assert len(full) == 4
    for t in full:
        flight = db.flights[t.persona.requested_flight]
        assert flight.seats[db.bookings[t.booking_ref].cabin] == 0
        assert len(t.goals) == 1
        # the goal books the other flight, not the requested one
        assert t.persona.requested_flight not in t.goals[0]["seats"]


def test_generation_is_deterministic_and_seed_sensitive():
    a, ta = generate(7)
    b, tb = generate(7)
    c, _ = generate(8)
    assert a.canonical() == b.canonical() and [t.id for t in ta] == [t.id for t in tb]
    assert a.canonical() != c.canonical()


def test_yaml_roundtrip(tmp_path):
    db, tasks = write_seed(tmp_path, 7)
    loaded = load_tasks(tmp_path / "tasks.yaml")
    assert loaded == tasks
    assert (tmp_path / "db.json").exists()


def test_reference_actions_reach_the_goal_with_no_violations(db, tasks):
    for t in tasks:
        ctx = context(t, db, t.reference_actions)
        assert all(c.ok for c in ctx.calls), t.id
        actual = diff_states(db.canonical(), ctx.final.canonical())
        assert any(diffs_equal(actual, g) for g in t.goals), t.id
        assert check_policy(ctx) == [], t.id


def test_missed_connection_delays_span_all_compensation_bands(db, tasks):
    delays = [
        db.flights[t.persona.ticket["flight_id"]].delay_minutes
        for t in tasks
        if t.scenario == "missed_connection"
    ]
    assert min(delays) >= 120 and max(delays) >= 480 and any(240 <= d < 480 for d in delays)
