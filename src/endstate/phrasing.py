"""Agent phrasing shared by the reference transcripts and the rule-based agents."""

from __future__ import annotations

GREETING = "Hello, thanks for contacting support."
ASK_IDENTITY = "Could you give me your booking reference and the last name on the booking?"
ASK_TICKET = "Could you confirm your flight number, cabin and the fare you paid?"
FOUND = "Thank you, I have your booking in front of me."
CLOSING = "Is there anything else I can help with today?"
APOLOGY = "I'm sorry, I could not complete that request."


def opening_turn() -> str:
    return f"{GREETING} {ASK_IDENTITY}"


def found_turn(scenario: str) -> str:
    detail = {
        "cancelled_rebook": (
            "I'm sorry your flight was cancelled. I will look for the nearest alternative."
        ),
        "missed_connection": (
            "I'm sorry about the missed connection. I will find you the next flight."
        ),
        "voluntary_change": "I will check the change fee and the new fare before moving you.",
        "refund_refundable": "Your fare is refundable, so I can cancel it and return the money.",
        "refund_nonrefundable": "I will check what this fare allows before cancelling it.",
        "name_correction": "I can see the booking and the name you want corrected.",
    }[scenario]
    return f"{FOUND} {detail}"


def resolution_rebook(flight_id: str, depart: str, cabin: str) -> str:
    return (
        f"I have moved you to flight {flight_id}, departing at {depart}, in the same {cabin} "
        "cabin. There is no charge for this move."
    )


def resolution_missed(flight_id: str, depart: str, cabin: str, voucher: float) -> str:
    return (
        f"I have moved you to flight {flight_id}, departing at {depart}, in {cabin}. "
        f"I have also added a travel voucher worth ${voucher:.2f} for the delay."
    )


def resolution_change(flight_id: str, depart: str, refund: float, fee: float) -> str:
    return (
        f"I have moved you to flight {flight_id}, departing at {depart}. After the ${fee:.2f} "
        f"change fee, ${refund:.2f} of the fare difference is refunded to your card."
    )


def resolution_refund(amount: float) -> str:
    return (
        f"I have cancelled the booking and refunded ${amount:.2f} to your original payment method."
    )


def resolution_voucher(amount: float) -> str:
    return (
        f"I have cancelled the booking and issued a voucher for ${amount:.2f} to your account, "
        "since this fare is not refundable."
    )


def resolution_transfer() -> str:
    return (
        "I am not able to change the name on a ticket myself, so I am passing you to a "
        "colleague who can."
    )
