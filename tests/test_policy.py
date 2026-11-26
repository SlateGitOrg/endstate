import pytest
from conftest import by_scenario, context

from endstate.policy import RULES, check_policy, compensation_for_delay
from endstate.tasks import (
    CANCELLED_REBOOK,
    MISSED_CONNECTION,
    NAME_CORRECTION,
    REFUND_NONREFUNDABLE,
    VOLUNTARY_CHANGE,
)


def get(task):
    return {
        "tool": "get_booking",
        "args": {"booking_ref": task.booking_ref, "last_name": task.persona.last_name},
    }


def rules_fired(task, db, actions) -> set[str]:
    return {v.rule for v in check_policy(context(task, db, actions))}


def alt_flight(task) -> str:
    return next(a["args"]["flight_id"] for a in task.reference_actions if a["tool"] == "rebook")


def test_identity_verified_fires_on_write_before_get_booking(tasks, db):
    t = by_scenario(tasks, CANCELLED_REBOOK)[4]
    rebook = {"tool": "rebook", "args": {"booking_ref": t.booking_ref, "flight_id": alt_flight(t)}}
    assert rules_fired(t, db, [rebook]) == {"identity_verified"}
    assert "identity_verified" not in rules_fired(t, db, [get(t), rebook])


def test_identity_needs_a_successful_lookup(tasks, db):
    t = by_scenario(tasks, CANCELLED_REBOOK)[4]
    bad = {"tool": "get_booking", "args": {"booking_ref": t.booking_ref, "last_name": "Nobody"}}
    rebook = {"tool": "rebook", "args": {"booking_ref": t.booking_ref, "flight_id": alt_flight(t)}}
    assert "identity_verified" in rules_fired(t, db, [bad, rebook])


def test_nonrefundable_cash_fires(tasks, db):
    t = by_scenario(tasks, REFUND_NONREFUNDABLE)[0]
    fare = t.persona.ticket["fare_paid"]
    cash = {"tool": "issue_refund", "args": {"booking_ref": t.booking_ref, "amount": fare}}
    assert rules_fired(t, db, [get(t), cash]) == {"nonrefundable_cash"}


def test_voluntary_refund_cap_fires_above_fare_minus_fee(tasks, db):
    t = by_scenario(tasks, VOLUNTARY_CHANGE)[4]
    ticket = t.persona.ticket
    cap = ticket["fare_paid"] - ticket["change_fee"]
    over = {"tool": "issue_refund", "args": {"booking_ref": t.booking_ref, "amount": cap + 1}}
    at_cap = {"tool": "issue_refund", "args": {"booking_ref": t.booking_ref, "amount": cap}}
    assert rules_fired(t, db, [get(t), over]) == {"voluntary_refund_cap"}
    assert rules_fired(t, db, [get(t), at_cap]) == set()


def test_same_cabin_fires_when_cabin_changes(tasks, db):
    t = by_scenario(tasks, CANCELLED_REBOOK)[4]
    cabin = t.persona.ticket["cabin"]
    other = "business" if cabin != "business" else "economy"
    move = {
        "tool": "rebook",
        "args": {"booking_ref": t.booking_ref, "flight_id": alt_flight(t), "cabin": other},
    }
    assert rules_fired(t, db, [get(t), move]) == {"same_cabin"}


def test_name_correction_must_be_transferred(tasks, db):
    t = by_scenario(tasks, NAME_CORRECTION)[0]
    assert rules_fired(t, db, [get(t)]) == {"name_correction_handoff"}
    handoff = {"tool": "transfer_to_human", "args": {"reason": "name correction"}}
    assert rules_fired(t, db, [get(t), handoff]) == set()


def test_delay_compensation_follows_table(tasks, db):
    t = by_scenario(tasks, MISSED_CONNECTION)[2]
    delay = t.persona.ticket["delay_minutes"]
    right = compensation_for_delay(delay)
    wrong = {"tool": "issue_voucher", "args": {"booking_ref": t.booking_ref, "amount": right + 25}}
    good = {"tool": "issue_voucher", "args": {"booking_ref": t.booking_ref, "amount": right}}
    assert rules_fired(t, db, [get(t), wrong]) == {"delay_compensation_table"}
    assert rules_fired(t, db, [get(t), good]) == set()


def test_at_most_one_compensation_per_booking(tasks, db):
    t = by_scenario(tasks, MISSED_CONNECTION)[2]
    amount = compensation_for_delay(t.persona.ticket["delay_minutes"])
    v = {"tool": "issue_voucher", "args": {"booking_ref": t.booking_ref, "amount": amount}}
    assert rules_fired(t, db, [get(t), v, v]) == {"one_compensation"}


def test_every_rule_has_a_firing_case_above():
    assert {r.name for r in RULES} == {
        "identity_verified",
        "nonrefundable_cash",
        "voluntary_refund_cap",
        "same_cabin",
        "name_correction_handoff",
        "delay_compensation_table",
        "one_compensation",
    }


@pytest.mark.parametrize(
    ("minutes", "amount"),
    [
        (0, 0.0),
        (119, 0.0),
        (120, 50.0),
        (239, 50.0),
        (240, 100.0),
        (479, 100.0),
        (480, 200.0),
        (900, 200.0),
    ],
)
def test_compensation_bands(minutes, amount):
    assert compensation_for_delay(minutes) == amount
