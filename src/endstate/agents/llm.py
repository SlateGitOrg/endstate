"""An optional language-model agent over an OpenAI-compatible chat endpoint.

``ENDSTATE_LLM_BASE_URL``, ``ENDSTATE_LLM_API_KEY`` and ``ENDSTATE_LLM_MODEL`` select the
endpoint. Nothing in the tests or the default demo uses it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Protocol

import httpx

from endstate.agents.base import Message, Observation, ToolCall
from endstate.env.tools import Toolbox
from endstate.policy import POLICY_TEXT

SYSTEM_PROMPT = (
    "You are a support agent for an online travel agency. Customers contact you to rebook "
    "after cancellations or delays, change flights, or ask for refunds. Use the tools to look "
    "up and change bookings. Ask for the booking reference and last name first. Reply in a "
    "short, polite message when you need something from the customer or when you are done.\n\n"
    + POLICY_TEXT
)


class ChatModel(Protocol):
    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Return the assistant message (OpenAI chat shape) for the conversation so far."""
        ...


class OpenAICompatModel:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self._client = client or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls) -> OpenAICompatModel:
        try:
            return cls(
                os.environ["ENDSTATE_LLM_BASE_URL"],
                os.environ["ENDSTATE_LLM_API_KEY"],
                os.environ["ENDSTATE_LLM_MODEL"],
            )
        except KeyError as exc:
            raise RuntimeError(f"set {exc.args[0]} to use the llm agent") from exc

    def complete(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        response = self._client.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "model": self.model,
                "messages": messages,
                "tools": tools,
                "parallel_tool_calls": False,
                "temperature": 0,
            },
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]


class LLMAgent:
    name = "llm"

    def __init__(self, model: ChatModel) -> None:
        self.model = model
        self.fired: list[str] = []
        self.tools = Toolbox.schemas()
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        self._pending_call: str | None = None

    def step(self, observation: Observation) -> ToolCall | Message:
        if observation.kind == "tool" and self._pending_call is not None:
            payload = {"ok": observation.ok, "error": observation.error, "data": observation.data}
            self.messages.append(
                {"role": "tool", "tool_call_id": self._pending_call, "content": json.dumps(payload)}
            )
            self._pending_call = None
        else:
            self.messages.append({"role": "user", "content": observation.text})
        reply = self.model.complete(self.messages, self.tools)
        self.messages.append(reply)
        calls = reply.get("tool_calls") or []
        if calls:
            call = calls[0]
            self._pending_call = call["id"]
            try:
                args = json.loads(call["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {"_unparsed": call["function"].get("arguments")}
            return ToolCall(call["function"]["name"], args)
        text = (reply.get("content") or "").strip()
        return Message(text or "Goodbye.", end=not text)
