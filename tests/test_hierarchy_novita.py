"""Unit tests for the chat-backed summariser.

The network call is injected, so every path here runs offline: malformed
replies, refusals, transport errors, throttling.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.novita import (DEFAULT_MODEL, NovitaSummarizer,
                                           Throttle, resolve_api_key)

GOOD_REPLY = ('{"summary": "A faithful summary of the span.", '
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


def test_missing_key_returns_none(monkeypatch, tmp_path):
    """Isolated to an empty directory on purpose. Run from the repo root this
    finds the developer's real .env -- which is the intended behaviour, but
    would make the assertion leak a live key into the test output."""
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    assert resolve_api_key() is None


def test_only_dotenv_is_read_never_an_arbitrary_env_file(monkeypatch, tmp_path):
    """The project's own `.env` is conventional and opt-in. Any *other* file --
    `env.txt` beside it, or one in a neighbouring project -- must be ignored.
    ~/Gemma4/env.txt holds live keys for five services and stays untouched."""
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    (tmp_path / "env.txt").write_text("NOVITA_API_KEY=sk-should-not-be-read\n")
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
    assert got.span_id == "s7" and got.title == "Concept Scoring"


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
    assert "basis vectors" in captured["system"]


# --------------------------------------------------------------------------
# Failure paths — none of which may raise
# --------------------------------------------------------------------------

def test_transport_error_becomes_a_record_not_an_exception():
    """One unreachable span must not abort a corpus build."""
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


# --------------------------------------------------------------------------
# Sanitisation of model output
# --------------------------------------------------------------------------

def test_model_output_is_sanitised():
    """A non-breaking hyphen must never reach an embedder: it tokenizes
    differently from '-', so the basis vector would not be the intended one."""
    reply = '{"label": "cross‑document linking key", "summary": "s", "abstraction": "a"}'
    got = summarizer(reply).summarize("s1", "T", "b")
    assert got.label == "cross-document linking key"
    assert "‑" not in got.label


def test_sanitisation_is_reported():
    reply = '{"label": "a‑b", "summary": "s", "abstraction": "a"}'
    got = summarizer(reply).summarize("s1", "T", "b")
    assert any("normalised" in w and "U+2011" in w for w in got.warnings)


def test_zero_width_characters_are_stripped_from_output():
    reply = '{"label": "con​cept space", "summary": "s", "abstraction": "a"}'
    assert summarizer(reply).summarize("s1", "T", "b").label == "concept space"


def test_curly_quotes_are_normalised_in_output():
    reply = '{"label": "the “concept” space", "summary": "s", "abstraction": "a"}'
    assert '"concept"' in summarizer(reply).summarize("s1", "T", "b").label


def test_real_content_survives_sanitisation():
    reply = '{"label": "Müller studied τ decay", "summary": "s", "abstraction": "a"}'
    got = summarizer(reply).summarize("s1", "T", "b")
    assert got.label == "Müller studied τ decay"


def test_eaten_escape_is_still_detected_despite_sanitising():
    """Sanitising strips the tab that evidences the damage, so detection must
    run first or the warning is lost."""
    got = summarizer(r'{"label": "the \tau function"}').summarize("s1", "T", "b")
    assert any("eaten" in w for w in got.warnings)
    assert "\t" not in got.label


# --------------------------------------------------------------------------
# .env loading — this project's file only
# --------------------------------------------------------------------------

def test_dotenv_is_loaded_from_the_project(monkeypatch, tmp_path):
    from conceptdrill.hierarchy.novita import load_dotenv
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    (tmp_path / ".env").write_text("NOVITA_API_KEY=sk-from-dotenv\n")
    monkeypatch.chdir(tmp_path)
    load_dotenv()
    assert resolve_api_key() == "sk-from-dotenv"


def test_existing_environment_beats_the_dotenv_file(monkeypatch, tmp_path):
    from conceptdrill.hierarchy.novita import load_dotenv
    monkeypatch.setenv("NOVITA_API_KEY", "sk-from-env")
    (tmp_path / ".env").write_text("NOVITA_API_KEY=sk-from-file\n")
    monkeypatch.chdir(tmp_path)
    load_dotenv()
    assert resolve_api_key() == "sk-from-env"


def test_dotenv_comments_and_blanks_are_skipped(monkeypatch, tmp_path):
    from conceptdrill.hierarchy.novita import load_dotenv
    monkeypatch.delenv("NOVITA_MODEL", raising=False)
    (tmp_path / ".env").write_text("# a comment\n\nNOVITA_MODEL=some/model\n")
    monkeypatch.chdir(tmp_path)
    assert load_dotenv()["NOVITA_MODEL"] == "some/model"


def test_missing_dotenv_is_not_an_error(monkeypatch, tmp_path):
    from conceptdrill.hierarchy.novita import load_dotenv
    monkeypatch.chdir(tmp_path)
    assert load_dotenv() == {}


def test_reasoning_effort_is_sent_and_in_the_cache_key(monkeypatch):
    """Reasoning tokens come out of the SAME completion budget as the answer,
    which is what the max_tokens ladder (900 -> 2000 -> 4000 -> 8000) was
    really fighting; 5 of 203 arXiv spans still hit 8000. Capping the
    reasoning treats the cause. Measured on the provider: 3931 completion
    tokens at default effort, 2553 at minimal (-35%)."""
    from conceptdrill.hierarchy import novita as nv

    seen = {}

    class _Msg:
        content = '{"summary": "s", "abstraction": "a", "label": "l"}'

    class _Choice:
        finish_reason = "stop"
        message = _Msg()

    class _Reply:
        choices = [_Choice()]
        usage = None

    class _Completions:
        def create(self, **kw):
            seen.update(kw)
            return _Reply()

    class _Chat:
        completions = _Completions()

    class _Client:
        chat = _Chat()

        def __init__(self, **kw):
            pass

    import openai
    monkeypatch.setattr(openai, "OpenAI", _Client)
    monkeypatch.setenv("NOVITA_API_KEY", "k")

    chat = nv.make_openai_chat(max_tokens=256)
    chat("sys", "usr")
    assert seen["reasoning_effort"] == "minimal"
    # must be in the cache key: a cache built at one effort must not serve a
    # run made at another, exactly as for max_tokens
    assert chat.params["reasoning_effort"] == "minimal"

    seen.clear()
    off = nv.make_openai_chat(max_tokens=256, reasoning_effort=None)
    off("sys", "usr")
    assert "reasoning_effort" not in seen, "None must omit the parameter entirely"
    assert off.params["reasoning_effort"] is None
