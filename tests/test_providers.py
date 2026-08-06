"""Provider wire-format tests.

These are the assertions that would otherwise only fail in production against a
real endpoint — particularly the NIM `chat_template_kwargs` requirement, whose
failure mode is a hang rather than an error, and so would look like the cron job
silently dying rather than like a bug.
"""

from __future__ import annotations

import pytest

from doomscroller.providers import build_provider, default_model_for, strip_reasoning
from doomscroller.providers.base import ProviderUnavailable
from doomscroller.providers.nim import DEFAULT_MODEL, EFFORT_MAP, NIMProvider

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}


# -- fake OpenAI-compatible client ---------------------------------------


class FakeMessage:
    def __init__(self, content, reasoning=None):
        self.content = content
        self.reasoning_content = reasoning


class FakeChoice:
    def __init__(self, content, finish_reason="stop", reasoning=None):
        self.message = FakeMessage(content, reasoning)
        self.finish_reason = finish_reason


class FakeUsage:
    prompt_tokens = 1200
    completion_tokens = 300


class FakeCompletion:
    def __init__(self, content="{}", finish_reason="stop", reasoning=None, choices=None):
        self.choices = choices if choices is not None else [FakeChoice(content, finish_reason, reasoning)]
        self.usage = FakeUsage()


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeOpenAI:
    def __init__(self, response=None, error=None):
        self.chat = type("Chat", (), {})()
        self.chat.completions = FakeCompletions(response, error)


def nim(response=None, error=None, **kwargs) -> NIMProvider:
    provider = NIMProvider(api_key="test-key", **kwargs)
    provider._client = FakeOpenAI(response if response is not None else FakeCompletion(), error)
    return provider


def call(provider: NIMProvider, effort: str = "low", **kwargs):
    return provider.complete(
        model=kwargs.pop("model", DEFAULT_MODEL),
        system="system prompt",
        user="user payload",
        schema=SCHEMA,
        effort=effort,
        max_tokens=4096,
        **kwargs,
    )


# -- the hang-avoidance contract -----------------------------------------


def test_chat_template_kwargs_is_always_sent():
    """Omitting this doesn't error on NIM — it hangs. Regressing it would look
    like the nightly job dying, not like a bug, so pin it."""
    provider = nim()
    call(provider)
    extra = provider._client.chat.completions.calls[0]["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": True, "thinking": True}


def test_chat_template_kwargs_is_sent_even_at_zero_reasoning_effort():
    provider = nim()
    call(provider, effort="low")  # maps to reasoning_effort "none"
    sent = provider._client.chat.completions.calls[0]
    assert sent["reasoning_effort"] == "none"
    assert "chat_template_kwargs" in sent["extra_body"]


def test_thinking_can_be_turned_off_explicitly():
    provider = nim(thinking=False)
    call(provider)
    extra = provider._client.chat.completions.calls[0]["extra_body"]
    assert extra["chat_template_kwargs"] == {"enable_thinking": False, "thinking": False}


# -- structured output ---------------------------------------------------


def test_schema_is_sent_as_guided_json_not_response_format():
    provider = nim()
    call(provider)
    sent = provider._client.chat.completions.calls[0]
    assert sent["extra_body"]["nvext"]["guided_json"] == SCHEMA
    # response_format json_object would permit an empty object; guided_json enforces.
    assert "response_format" not in sent


# -- effort mapping ------------------------------------------------------


@pytest.mark.parametrize("effort,expected", sorted(EFFORT_MAP.items()))
def test_effort_maps_onto_nim_reasoning_effort(effort, expected):
    provider = nim()
    call(provider, effort=effort)
    assert provider._client.chat.completions.calls[0]["reasoning_effort"] == expected


def test_an_unknown_effort_falls_back_to_high():
    provider = nim()
    call(provider, effort="ludicrous")
    assert provider._client.chat.completions.calls[0]["reasoning_effort"] == "high"


# -- response handling ---------------------------------------------------


def test_reasoning_is_not_mixed_into_the_parsed_content():
    provider = nim(FakeCompletion(content='{"ok": true}', reasoning="Let me think about this..."))
    result = call(provider)
    assert result.text == '{"ok": true}'
    assert "think" not in result.text


def test_inline_think_tags_are_stripped_anyway():
    """Self-hosted containers sometimes inline reasoning the hosted API separates."""
    provider = nim(FakeCompletion(content='<think>hmm, maybe</think>\n{"ok": true}'))
    assert call(provider).text == '{"ok": true}'


def test_a_markdown_fence_is_stripped():
    provider = nim(FakeCompletion(content='```json\n{"ok": true}\n```'))
    assert call(provider).text == '{"ok": true}'


def test_token_usage_is_reported_and_caching_is_honestly_zero():
    result = call(nim())
    assert result.input_tokens == 1200
    assert result.output_tokens == 300
    assert result.cached_tokens == 0  # NIM has no prompt caching


def test_content_filter_is_reported_as_a_refusal():
    result = call(nim(FakeCompletion(content="", finish_reason="content_filter")))
    assert result.refused and not result.ok


def test_truncation_before_any_content_is_an_error_not_an_empty_answer():
    result = call(nim(FakeCompletion(content="", finish_reason="length")))
    assert not result.ok
    assert "max_tokens" in result.error


def test_empty_choices_is_an_error_not_a_crash():
    result = call(nim(FakeCompletion(choices=[])))
    assert not result.ok and "no choices" in result.error


def test_a_transport_error_becomes_a_result_not_an_exception():
    result = call(nim(error=RuntimeError("connection reset")))
    assert not result.ok
    assert "connection reset" in result.error


# -- credentials ---------------------------------------------------------


def test_missing_key_names_the_variable_and_where_to_get_one(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NIM_API_KEY", raising=False)
    with pytest.raises(ProviderUnavailable, match="NVIDIA_API_KEY"):
        call(NIMProvider())


def test_base_url_is_overridable_for_self_hosted(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "k")
    assert NIMProvider(base_url="http://localhost:8000/v1").base_url == "http://localhost:8000/v1"


def test_nim_key_env_var_alias_works(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NIM_API_KEY", "k")
    assert NIMProvider().api_key == "k"


# -- registry ------------------------------------------------------------


@pytest.mark.parametrize("alias", ["nvidia_nim", "nim", "nvidia"])
def test_nim_aliases_all_resolve(alias):
    assert build_provider(alias, {"api_key": "k"}).name == "nvidia_nim"


@pytest.mark.parametrize("alias", ["anthropic", "claude"])
def test_claude_aliases_all_resolve(alias):
    assert build_provider(alias).name == "anthropic"


def test_unknown_provider_lists_the_known_ones():
    with pytest.raises(ProviderUnavailable, match="nvidia_nim"):
        build_provider("mystery-llm")


def test_default_model_follows_the_provider():
    assert default_model_for("nvidia_nim") == "deepseek-ai/deepseek-v4-flash"
    assert default_model_for("anthropic") == "claude-opus-5"


# -- shared helper -------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', '{"a": 1}'),
        ('<think>reasoning</think>{"a": 1}', '{"a": 1}'),
        ('```json\n{"a": 1}\n```', '{"a": 1}'),
        ('{"a": 1}<think>unterminated reasoning', '{"a": 1}'),
        ("", ""),
    ],
)
def test_strip_reasoning(raw, expected):
    assert strip_reasoning(raw) == expected
