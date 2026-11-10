"""LLM client (llm.py): request shape per provider, without any network."""

import json
from types import SimpleNamespace

import pytest

from complaintexp.config import load_variants
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


def test_hosted_openai_compatible_endpoint_reads_its_own_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-judge")
    v = make_variant(provider="openai_compatible", base_url="http://127.0.0.1:1/v1",
                     extra={"api_key_env": "OPENAI_API_KEY"})
    assert LLMClient(v)._client.api_key == "sk-judge"


def test_shipped_second_judge_without_azure_is_a_gpt_endpoint():
    j = load_variants()["JUDGE_OPENAI"]
    assert j.provider == "openai_compatible" and j.extra["api_key_env"] == "OPENAI_API_KEY" and j.cloud


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


def test_max_tokens_follow_the_tier_and_grow_when_reasoning_is_on():
    c = _client(max_tokens_by_tier={1: 300, 4: 1200}, reasoning_extra_tokens=5000)
    c.complete("S", "U", tier=1)
    assert c._client.kwargs["max_tokens"] == 300
    t = _client(max_tokens_by_tier={1: 300}, reasoning_extra_tokens=5000, reasoning="on")
    t.complete("S", "U", tier=1)
    assert t._client.kwargs["max_tokens"] == 5300


def test_reasoning_mode_request_is_sent_as_extra_body():
    c = _client(request={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    c.complete("S", "U")
    assert c._client.kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_temperature_none_is_not_sent():
    c = _client(temperature=None)
    c.complete("S", "U")
    assert "temperature" not in c._client.kwargs


def test_chat_sends_the_whole_conversation():
    c = _client()
    c.chat("S", [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                 {"role": "user", "content": "c"}])
    assert [m["content"] for m in c._client.kwargs["messages"]] == ["S", "a", "b", "c"]


# -- Claude (Anthropic SDK) -------------------------------------------------------
class _AnthropicRecorder:
    def __init__(self):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text='{"a": 1}')],
            usage=SimpleNamespace(input_tokens=21, output_tokens=9), model="claude-haiku-5-5")


def _claude(**kw):
    c = LLMClient(make_variant(provider="fake", **kw))  # no SDK client is built for `fake`
    c.v.provider = "anthropic"
    c._client = _AnthropicRecorder()
    return c


def test_claude_request_never_sends_temperature_and_maps_reasoning():
    c = _claude(model="claude-sonnet-5-5", temperature=0.2, max_tokens_by_tier={3: 900},
                request={"thinking": {"type": "between_tools"}, "effort": "high"})
    out = c.complete("S", "U", tier=3)
    k = c._client.kwargs
    assert "temperature" not in k and k["max_tokens"] == 900 and k["system"] == "S"
    assert k["thinking"] == {"type": "between_tools"} and k["output_config"] == {"effort": "high"}
    assert k["messages"] == [{"role": "user", "content": "U"}]
    assert (out.text, out.prompt_tokens, out.completion_tokens) == ('{"a": 1}', 21, 9)


def test_claude_without_reasoning_settings_sends_neither():
    c = _claude()
    c.complete("S", "U")
    assert "thinking" not in c._client.kwargs and "output_config" not in c._client.kwargs


@pytest.mark.parametrize("tier", [1, 2, 3, 4, 5, 6])
def test_fake_answers_every_tiers_output_contract(tier):
    from complaintexp.prompting import load_system_prompt

    out = LLMClient(make_variant(provider="fake")).complete(load_system_prompt(tier), "# Complaint (language: nl)\nORD-1234567")
    assert parse_output(out.text, tier).ok
