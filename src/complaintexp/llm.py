"""One thin client for every variant, so only model and hosting differ.

Azure and local servers (Ollama, vLLM) are all driven through the OpenAI
chat-completions API. `fake` needs no network and is used for dry runs/tests.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass

from .config import Variant


@dataclass
class Completion:
    text: str
    latency_s: float
    prompt_tokens: int | None
    completion_tokens: int | None
    model_id: str  # model/version string reported by the server


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
                api_key=os.environ.get("LOCAL_API_KEY", "not-needed"),
            )
        elif variant.provider != "fake":
            raise ValueError(f"unknown provider {variant.provider!r}")

    def complete(self, system: str, user: str) -> Completion:
        if self.v.provider == "fake":
            return _fake(system, user)
        kwargs = {
            "model": self.v.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.v.temperature,
            self.v.token_param: self.v.max_tokens,
        }
        if self.v.extra.get("reasoning_effort"):
            # e.g. "none" turns off qwen3-style thinking, which otherwise eats max_tokens
            kwargs["reasoning_effort"] = self.v.extra["reasoning_effort"]
        if self.v.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        latency = time.perf_counter() - t0
        usage = resp.usage
        return Completion(
            text=resp.choices[0].message.content or "",
            latency_s=latency,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
            model_id=getattr(resp, "model", None) or self.v.model,
        )


def _fake(system: str, user: str) -> Completion:
    """Echoes the complaint back inside the required JSON shape."""
    t0 = time.perf_counter()
    first_line = user.strip().splitlines()[0] if user.strip() else ""
    out = json.dumps(
        {
            "draft_reply": f"Thank you for your message. We are looking into: {first_line[:80]}",
            "agent_note": "Fake model output for a dry run.",
        }
    )
    return Completion(out, time.perf_counter() - t0, len(system + user) // 4, len(out) // 4, "fake-1")
