from conftest import by_scenario

from endstate.env.tools import Toolbox
from endstate.tasks import NAME_CORRECTION, REFUND_NONREFUNDABLE
from endstate.user_sim import UserSimulator, UserState


def test_user_opens_with_request_and_answers_identity_from_persona(tasks, db):
    t = tasks[0]
    sim = UserSimulator(t, db)
    first = sim.opening()
    assert first.text == t.persona.opening and first.data["intent"] == t.scenario
    turn = sim.respond("Could you give me your booking reference and the last name?", db.copy())
    assert not turn.done
    assert turn.observation.data == {"booking_ref": t.booking_ref, "last_name": t.persona.last_name}
    assert sim.state is UserState.IDENTIFIED


def test_user_reads_ticket_only_when_asked_a_question(tasks, db):
    sim = UserSimulator(tasks[0], db)
    sim.opening()
    quiet = sim.respond("Your cabin has been updated.", db.copy())
    assert "ticket" not in quiet.observation.data
    asked = sim.respond(
        "Could you confirm your flight number, cabin and the fare you paid?", db.copy()
    )
    assert asked.observation.data["ticket"]["flight_id"] == tasks[0].persona.ticket["flight_id"]


def test_user_ends_dialogue_when_goal_state_is_reached(tasks, db):
    t = by_scenario(tasks, REFUND_NONREFUNDABLE)[0]
    work = db.copy()
    sim = UserSimulator(t, db)
    sim.opening()
    assert not sim.respond("Anything else?", work).done
    for a in t.reference_actions:
        Toolbox(work).execute(a["tool"], a["args"])
    turn = sim.respond("All done.", work)
    assert turn.done and sim.state is UserState.DONE


def test_user_not_satisfied_by_wrong_state(tasks, db):
    t = by_scenario(tasks, REFUND_NONREFUNDABLE)[0]
    work = db.copy()
    Toolbox(work).execute("cancel_booking", {"booking_ref": t.booking_ref})
    Toolbox(work).execute("issue_refund", {"booking_ref": t.booking_ref, "amount": 10})
    assert not UserSimulator(t, db).respond("Done.", work).done


def test_user_is_done_after_transfer(tasks, db):
    sim = UserSimulator(by_scenario(tasks, NAME_CORRECTION)[0], db)
    assert sim.on_transfer().done
