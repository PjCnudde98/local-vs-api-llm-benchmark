"""One thin client for every variant, so only model and hosting differ.

Azure OpenAI and local servers (Ollama, vLLM, llama.cpp, SGLang) are driven
through the OpenAI chat-completions API. Claude goes through the Anthropic SDK:
Microsoft Foundry when ANTHROPIC_FOUNDRY_RESOURCE is set, else the Claude API.
`fake` needs no network and is used for dry runs and tests.

The request builders are pure functions so tests can check exactly what would
be sent without a network.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from .config import Variant

Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    model_id: str  # model/version string reported by the server


def openai_request(v: Variant, system: str, messages: list[Message], tier: int | None = None) -> dict[str, Any]:
    """Chat-completions kwargs for Azure OpenAI and OpenAI-compatible local servers."""
    kwargs: dict[str, Any] = {
        "model": v.model,
        "messages": [{"role": "system", "content": system}, *messages],
        v.token_param: v.tokens_for(tier),
    }
    if v.temperature is not None:
        kwargs["temperature"] = v.temperature
    effort = v.request.get("reasoning_effort") or v.extra.get("reasoning_effort")
    if effort:
        kwargs["reasoning_effort"] = effort
    if v.request.get("extra_body"):
        kwargs["extra_body"] = v.request["extra_body"]  # e.g. chat_template_kwargs.enable_thinking
    if v.json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    return kwargs


def anthropic_request(v: Variant, system: str, messages: list[Message], tier: int | None = None) -> dict[str, Any]:
    """Messages API kwargs. Temperature is never sent: the API rejects it since 2026-10-21.

    JSON output is asked for in the prompt (same text as every other variant)
    rather than with a per-tier schema, so the instruction stays identical.
    """
    kwargs: dict[str, Any] = {
        "model": v.model,
        "max_tokens": v.tokens_for(tier),
        "system": system,
        "messages": list(messages),
    }
    if v.request.get("thinking"):
        kwargs["thinking"] = v.request["thinking"]
    if v.request.get("effort"):
        kwargs["output_config"] = {"effort": v.request["effort"]}
    return kwargs


class LLMClient:
    def __init__(self, variant: Variant):
        self.v = variant
        self._client = None
        if variant.provider == "azure":
            from openai import AzureOpenAI

            self._client = AzureOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                api_key=os.environ["AZURE_OPENAI_API_KEY"],
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        elif variant.provider == "openai_compatible":
            from openai import OpenAI

            self._client = OpenAI(
                base_url=variant.base_url,
                # local servers ignore the key; a hosted endpoint names its own variable (api_key_env)
                api_key=os.environ.get(variant.extra.get("api_key_env") or "LOCAL_API_KEY", "not-needed"),
                timeout=float(os.environ.get("LOCAL_TIMEOUT_S", "600")),
            )
        elif variant.provider == "anthropic":
            import anthropic

            resource = os.environ.get("ANTHROPIC_FOUNDRY_RESOURCE")
            if resource:
                self._client = anthropic.AnthropicFoundry(
                    resource=resource, api_key=os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
                )
            else:
                self._client = anthropic.Anthropic()
        elif variant.provider != "fake":
            raise ValueError(f"unknown provider {variant.provider!r}")

    def complete(self, system: str, user: str, tier: int | None = None) -> Completion:
        return self.chat(system, [{"role": "user", "content": user}], tier)

    def chat(self, system: str, messages: list[Message], tier: int | None = None) -> Completion:
        if self.v.provider == "fake":
            return _fake(system, messages)
        t0 = time.perf_counter()
        if self.v.provider == "anthropic":
            resp = self._client.messages.create(**anthropic_request(self.v, system, messages, tier))
            latency = time.perf_counter() - t0
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return Completion(text, latency, resp.usage.input_tokens, resp.usage.output_tokens,
                              getattr(resp, "model", None) or self.v.model)
        resp = self._client.chat.completions.create(**openai_request(self.v, system, messages, tier))
        latency = time.perf_counter() - t0
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            model_id=getattr(resp, "model", None) or self.v.model,
        )


def _fake(system: str, messages: list[Message]) -> Completion:
    """Deterministic answer in the shape the task asks for. No network."""
    t0 = time.perf_counter()
    user = messages[-1]["content"] if messages else ""
    m = re.search(r"# Task \(tier (\d)\)", system)
    tier = int(m.group(1)) if m else 3
    lang = re.search(r"language: (\w+)", messages[0]["content"] if messages else "")
    if "# Abstraction" in system:
        out = {"summary": "Customer reports a problem with an order and asks for a solution."}
    elif "# Routing" in system:
        out = {"complexity": "easy"}
    elif tier == 1:
        out = {"type": "delivery", "language": lang.group(1) if lang else "en", "urgency": "normal",
               "sentiment": "negative"}
    elif tier == 2:
        order = re.search(r"ORD-\d+|<ORDER_\d+>", user)
        out = {"order_number": order.group(0) if order else None, "product": None,
               "requested_remedy": "other", "deadline": None}
    elif tier == 5:
        out = {"final": {"decision": "escalate", "justification": "Fake model.", "agent_note": "Dry run."}}
    else:
        first = user.strip().splitlines()[1] if len(user.strip().splitlines()) > 1 else ""
        out = {"draft_reply": f"Thank you for your message. We are looking into: {first[:80]}",
               "agent_note": "Fake model output for a dry run."}
    text = json.dumps(out)
    return Completion(text, time.perf_counter() - t0, len(system + user) // 4, len(text) // 4, "fake-1")
