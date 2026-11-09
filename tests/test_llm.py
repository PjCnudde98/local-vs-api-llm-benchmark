"""LLM client (llm.py): request shape per provider, without any network."""

import json
from types import SimpleNamespace

import pytest

from complaintexp.llm import LLMClient
from complaintexp.prompting import parse_output

from .conftest import make_variant


class _Recorder:
    def __init__(self):
        self.kwargs = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        msg = SimpleNamespace(content='{"draft_reply": "x"}')
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)],
                               usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7), model="served-model")


def _client(**kw):
    c = LLMClient(make_variant(provider="openai_compatible", base_url="http://127.0.0.1:1/v1", **kw))
    c._client = _Recorder()
    return c


def test_fake_provider_returns_parseable_output():
    out = LLMClient(make_variant(provider="fake")).complete("sys", "# Complaint\nhello")
    assert parse_output(out.text).ok and out.model_id == "fake-1"


def test_unknown_provider_is_rejected():
    with pytest.raises(ValueError, match="unknown provider"):
        LLMClient(make_variant(provider="bedrock"))


def test_request_carries_shared_generation_settings():
    c = _client(temperature=0.2, max_tokens=800)
    c.complete("S", "U")
    k = c._client.kwargs
    assert k["temperature"] == 0.2 and k["max_tokens"] == 800
    assert k["messages"] == [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]
    assert k["response_format"] == {"type": "json_object"}


def test_token_param_name_is_configurable():
    c = _client(token_param="max_completion_tokens")
    c.complete("S", "U")
    assert "max_completion_tokens" in c._client.kwargs and "max_tokens" not in c._client.kwargs


def test_json_mode_off_sends_no_response_format():
    c = _client(json_mode=False)
    c.complete("S", "U")
    assert "response_format" not in c._client.kwargs


def test_reasoning_effort_is_passed_through_from_extra():
    c = _client(extra={"reasoning_effort": "none"})
    c.complete("S", "U")
    assert c._client.kwargs["reasoning_effort"] == "none"


def test_completion_reports_usage_and_served_model():
    out = _client().complete("S", "U")
    assert (out.prompt_tokens, out.completion_tokens, out.model_id) == (11, 7, "served-model")
    assert out.latency_s >= 0 and json.loads(out.text)
