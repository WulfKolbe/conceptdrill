"""Unit tests for the chat-backed summariser.

The network call is injected, so every path here runs offline: malformed
replies, refusals, transport errors, throttling.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                           Throttle, resolve_api_key)

GOOD_REPLY = ('{"summary": "A faithful summary of the section.", '
              '"abstraction": "The underlying idea.", '
              '"label": "A canonical reusable concept definition."}')


def summarizer(reply, **kw):
    """A summariser whose chat call returns `reply` (or raises it)."""
    def chat(system, user):
        if isinstance(reply, Exception):
            raise reply
        return reply
    kw.setdefault("throttle", Throttle(min_interval=0))
    return NovitaSummarizer(chat, **kw)


# --------------------------------------------------------------------------
# Credentials
# --------------------------------------------------------------------------

def test_explicit_key_wins(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "from-env")
    assert resolve_api_key("explicit") == "explicit"


def test_key_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("NOVITA_API_KEY", "sk-test")
    assert resolve_api_key() == "sk-test"


def test_missing_key_returns_none(monkeypatch):
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert resolve_api_key() is None


def test_no_credentials_are_read_from_any_file(monkeypatch, tmp_path):
    """A library that harvests keys from a neighbouring dotfile is a liability.
    ~/Gemma4/env.txt holds live keys for five services and must stay untouched."""
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env_file = tmp_path / "env.txt"
    env_file.write_text("NOVITA_API_KEY=sk-should-not-be-read\n")
    monkeypatch.chdir(tmp_path)
    assert resolve_api_key() is None


# --------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------

def test_all_three_tiers_are_extracted():
    got = summarizer(GOOD_REPLY).summarize("s1", "Method", "body text")
    assert got.summary and got.abstraction
    assert got.label == "A canonical reusable concept definition."


def test_result_is_marked_non_deterministic():
    """A language model produced it; a run must be able to say so."""
    assert not summarizer(GOOD_REPLY).summarize("s1", "T", "b").deterministic


def test_model_name_is_recorded():
    got = summarizer(GOOD_REPLY, model="some/model").summarize("s1", "T", "b")
    assert got.model == "some/model"


def test_section_id_and_title_are_carried():
    got = summarizer(GOOD_REPLY).summarize("s7", "Concept Scoring", "b")
    assert got.section_id == "s7" and got.title == "Concept Scoring"


def test_fenced_reply_is_accepted():
    got = summarizer(f"```json\n{GOOD_REPLY}\n```").summarize("s1", "T", "b")
    assert got.is_usable


# --------------------------------------------------------------------------
# The prompt sent to the model
# --------------------------------------------------------------------------

def test_user_prompt_carries_title_and_body():
    prompt = summarizer(GOOD_REPLY).build_user_prompt("Method", "the body")
    assert "TITLE: Method" in prompt and "the body" in prompt


def test_body_is_truncated_to_the_budget():
    s = summarizer(GOOD_REPLY, max_body=50)
    assert len(s.build_user_prompt("T", "x" * 5000)) < 200


def test_system_prompt_is_the_packaged_one():
    captured = {}

    def chat(system, user):
        captured["system"] = system
        return GOOD_REPLY

    NovitaSummarizer(chat, throttle=Throttle(0)).summarize("s1", "T", "b")
    assert "BASIS VECTOR" in captured["system"]


# --------------------------------------------------------------------------
# Failure paths — none of which may raise
# --------------------------------------------------------------------------

def test_transport_error_becomes_a_record_not_an_exception():
    """One unreachable section must not abort a corpus build."""
    got = summarizer(ConnectionError("down")).summarize("s1", "T", "b")
    assert got.error and "ConnectionError" in got.error
    assert not got.is_usable


def test_non_json_reply_is_reported():
    got = summarizer("I'm afraid I can't do that.").summarize("s1", "T", "b")
    assert got.error == "reply was not JSON"


def test_non_json_reply_keeps_the_raw_text_for_diagnosis():
    got = summarizer("nonsense here").summarize("s1", "T", "b")
    assert any("nonsense here" in w for w in got.warnings)


def test_json_without_any_tier_is_reported():
    got = summarizer('{"unrelated": "value"}').summarize("s1", "T", "b")
    assert got.error == "reply contained no tier text"


def test_partial_reply_is_kept():
    """A label alone is still a usable basis vector."""
    got = summarizer('{"label": "just the label"}').summarize("s1", "T", "b")
    assert got.is_usable and got.basis_text == "just the label"


def test_eaten_latex_command_is_warned_about():
    """`\\tau` arrives as a tab: legal JSON, silently wrong text."""
    got = summarizer(r'{"label": "the \tau function"}').summarize("s1", "T", "b")
    assert got.warnings
    assert any("eaten" in w for w in got.warnings)


def test_clean_reply_produces_no_warnings():
    assert summarizer(GOOD_REPLY).summarize("s1", "T", "b").warnings == ()


# --------------------------------------------------------------------------
# Throttle
# --------------------------------------------------------------------------

def test_first_call_does_not_wait():
    clock = iter([0.0, 0.0, 0.0])
    t = Throttle(2.2, clock=lambda: next(clock), sleep=lambda s: None)
    assert t.wait() == 0.0


def test_second_call_waits_the_remaining_interval():
    times = iter([0.0, 0.0, 0.5, 0.5])
    slept = []
    t = Throttle(2.2, clock=lambda: next(times), sleep=slept.append)
    t.wait()
    t.wait()
    assert slept and slept[0] == pytest.approx(1.7)


def test_no_wait_when_enough_time_has_passed():
    times = iter([0.0, 0.0, 99.0, 99.0])
    slept = []
    t = Throttle(2.2, clock=lambda: next(times), sleep=slept.append)
    t.wait()
    t.wait()
    assert slept == []


def test_zero_interval_disables_throttling():
    slept = []
    t = Throttle(0, clock=lambda: 0.0, sleep=slept.append)
    t.wait()
    t.wait()
    assert slept == []
