"""The real writer (T-0119, step C) -- tested with a fake chat, no key spent.

`LLMWriter` is the only stage of `kb build` that calls a model. It sees the
cited sections and the payload list, replies in JSON, and every reply is
cached by content, so a rebuild costs nothing and a failed build resumes.
"""
from __future__ import annotations

import json
import re

import pytest

from conceptdrill import cli
from conceptdrill.kb.build import BuildError
from conceptdrill.kb.writer import LLMWriter, estimate_cost
from conceptdrill.output import REPO_ROOT

LIB = REPO_ROOT / "tests" / "fixtures" / "kb-synthetic"
FLORP, ZIB = "Florp Dynamics (v2)", "zib_zab"

LEAF_REPLY = {"title": "Florp quibbling", "use_when": "When florps quibble.",
              "summary": "Florps quibble at marls.",
              "concepts": [{"name": "florp rate", "definition": "How fast florps go."}],
              "method": ["Count the florps.", "Divide by marls."],
              "pitfalls": ["Marls drift."]}
TOPIC_REPLY = {"title": "Florp matters", "purpose": "Everything florp.",
               "differ": "They differ in marls."}
AREA_REPLY = {"title": "Florp lands", "purpose": "All florp topics.",
              "differ": "They differ in zibs."}
KB_REPLY = {"title": "A florp base", "purpose": "Florps and zibs."}


class FakeChat:
    """Replies by the kind of request its system prompt names."""

    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies
        self.params = {"model": "fake", "temperature": 0.2}
        self.usage = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def __call__(self, system, user):
        self.calls.append((system, user))
        self.usage["calls"] += 1
        self.usage["prompt_tokens"] += len(user.split())
        self.usage["completion_tokens"] += 10
        if self.replies:
            return self.replies.pop(0)
        if "KIND: leaf" in system:
            return json.dumps(LEAF_REPLY)
        if "KIND: topic" in system:
            return json.dumps(TOPIC_REPLY)
        if "KIND: area" in system:
            return json.dumps(AREA_REPLY)
        return json.dumps(KB_REPLY)


SECTIONS = [{"id": "d#0", "title": "Quibbling", "text": "florps quibble at marls"}]


def test_a_leaf_reply_is_parsed_and_cached(tmp_path):
    chat = FakeChat()
    w = LLMWriter(chat, model="fake", cache_path=tmp_path / "w.json")
    assert w.leaf(SECTIONS, [])["title"] == "Florp quibbling"
    assert w.leaf(SECTIONS, [])["method"] == LEAF_REPLY["method"]
    assert len(chat.calls) == 1
    w.flush()
    again = LLMWriter(FakeChat(), model="fake", cache_path=tmp_path / "w.json")
    assert again.leaf(SECTIONS, [])["title"] == "Florp quibbling"
    assert again.usage["calls"] == 0


def test_the_prompt_carries_the_sources_and_the_payload_ids_only(tmp_path):
    chat = FakeChat()
    LLMWriter(chat, model="fake", cache_path=tmp_path / "w.json").leaf(
        SECTIONS, [{"id": "d#obj_1", "kind": "code", "content": "int x;"}])
    system, user = chat.calls[0]
    assert "florps quibble at marls" in user and "d#obj_1" in user
    assert "int x;" not in user          # payload is copied by code, never rewritten


def test_a_bad_reply_is_retried_once_then_the_build_fails(tmp_path):
    good = FakeChat(replies=["not json", json.dumps(LEAF_REPLY)])
    assert LLMWriter(good, model="fake", cache_path=tmp_path / "a.json").leaf(
        SECTIONS, [])["title"] == "Florp quibbling"
    bad = FakeChat(replies=["not json", "{\"title\": 3}"])
    with pytest.raises(BuildError, match="leaf"):
        LLMWriter(bad, model="fake", cache_path=tmp_path / "b.json").leaf(SECTIONS, [])


def test_a_build_with_the_writer_conforms_and_records_its_usage(tmp_path, monkeypatch):
    from conceptdrill.kb.check import check
    chat = FakeChat()
    monkeypatch.setattr(cli, "make_kb_writer", lambda args: LLMWriter(
        chat, model="fake", cache_path=tmp_path / "w.json"))
    code = cli.main(["kb", "build", FLORP, ZIB, "--name", "real", "--library", str(LIB),
                     "--output-root", str(tmp_path), "--llm", "novita",
                     "--embedder", "hash"])
    assert code == 0
    kb = tmp_path / "kb" / "real"
    assert check(kb).violations == []
    llm = json.loads((kb / "manifest.json").read_text())["llm"]
    assert llm["stub"] is False and llm["model"] == "fake"
    assert llm["usage"]["calls"] == len(chat.calls) > 0
    leaf = next((kb / "areas").rglob("florp-quibbling*.md"))
    text = leaf.read_text()
    assert "1. Count the florps." in text and "- Marls drift." in text


def test_estimate_names_calls_tokens_and_cost_and_writes_nothing(tmp_path, capsys):
    code = cli.main(["kb", "build", FLORP, ZIB, "--name", "est", "--library", str(LIB),
                     "--output-root", str(tmp_path), "--llm", "novita",
                     "--embedder", "hash", "--estimate"])
    assert code == 0
    out = capsys.readouterr().out
    assert "estimate:" in out and "USD" in out and "calls" in out
    assert not (tmp_path / "kb" / "est").exists()


def test_estimate_cost_uses_the_models_prices():
    cheap = estimate_cost("inclusionai/ling-3.0-flash", calls=10, prompt_tokens=10_000)
    dear = estimate_cost("inclusionai/ling-3.1-flash", calls=10, prompt_tokens=10_000)
    assert 0 < cheap["usd"] < dear["usd"]
    assert cheap["completion_tokens"] == 10 * cheap["completion_per_call"]


# --------------------------------------------------------------------------
# Replay from the cache with no key (coordinator audit of 3b06215)
# --------------------------------------------------------------------------

def _build(tmp_path, name, *extra):
    return cli.main(["kb", "build", FLORP, ZIB, "--name", name, "--library", str(LIB),
                     "--output-root", str(tmp_path / "out"), "--llm", "novita",
                     "--embedder", "hash", *extra])


def _tree(kb):
    return {p.relative_to(kb).as_posix(): p.read_bytes() for p in kb.rglob("*")
            if p.is_file() and p.name != "manifest.json"}


def _no_key(monkeypatch):
    """No key anywhere, and any attempt to make a chat fails the test."""
    from conceptdrill.hierarchy import novita
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.setattr(novita, "load_dotenv", lambda *a, **k: {})
    def refuse(**kw):
        raise AssertionError("a chat was made: this would have spent")
    monkeypatch.setattr(novita, "make_openai_chat", refuse)


def _filled_cache(tmp_path, monkeypatch):
    from conceptdrill.hierarchy import novita
    monkeypatch.setenv("CONCEPTDRILL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("NOVITA_API_KEY", "k")
    chat = FakeChat()
    monkeypatch.setattr(novita, "make_openai_chat", lambda **kw: chat)
    assert _build(tmp_path, "first") == 0
    assert chat.calls
    return tmp_path / "out" / "kb" / "first"


def test_a_full_cache_replays_with_no_key(tmp_path, monkeypatch):
    first = _filled_cache(tmp_path, monkeypatch)
    _no_key(monkeypatch)
    assert _build(tmp_path, "first", "--replace") == 0
    replay = tmp_path / "out" / "kb" / "first"
    manifest = json.loads((replay / "manifest.json").read_text())
    assert manifest["llm"]["usage"]["calls"] == 0


def test_the_replay_equals_the_cached_build(tmp_path, monkeypatch):
    first = _filled_cache(tmp_path, monkeypatch)
    before = _tree(first)
    _no_key(monkeypatch)
    assert _build(tmp_path, "second", "--cache-only") == 0
    assert _tree(tmp_path / "out" / "kb" / "second") == {
        k: v.replace(b"kb: first", b"kb: second").replace(b'"kb": "first"', b'"kb": "second"')
        for k, v in before.items()}


def test_cache_only_with_an_empty_cache_fails_and_names_the_miss(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CONCEPTDRILL_CACHE_DIR", str(tmp_path / "empty"))
    _no_key(monkeypatch)
    assert _build(tmp_path, "x", "--cache-only") == 1
    err = capsys.readouterr().err
    assert "cache miss" in err and "leaf" in err and "florp-dynamics-v2#" in err
    assert not (tmp_path / "out" / "kb" / "x").exists()


def test_without_cache_only_a_miss_with_no_key_fails_before_spending(tmp_path, monkeypatch, capsys):
    from conceptdrill.hierarchy import novita
    monkeypatch.setenv("CONCEPTDRILL_CACHE_DIR", str(tmp_path / "empty"))
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.setattr(novita, "load_dotenv", lambda *a, **k: {})
    assert _build(tmp_path, "y") == 1
    assert "NOVITA_API_KEY" in capsys.readouterr().err


def test_the_cache_key_parameters_are_the_chats(monkeypatch):
    """The writer now knows its parameters before any chat exists; they must
    be the chat's own, or a replay would miss every entry of a paid build."""
    from conceptdrill.hierarchy import novita
    from conceptdrill.kb.writer import chat_params
    monkeypatch.setenv("NOVITA_API_KEY", "k")
    assert chat_params("m") == novita.make_openai_chat(model="m").params


def test_payload_carried_for_the_chapter_is_not_in_the_writer_prompt(tmp_path, monkeypatch):
    """T-0130: the writer is asked about the payload of the sections a leaf
    cites only, so carrying a chapter's payload changes no prompt (and no
    cache key, and costs nothing)."""
    import numpy as np
    from conceptdrill.kb.build import Area, Leaf, Payload, Section, Topic, write_prose
    sec = lambda o, n: Section("d", o, f"S{o}", n, 1, 1, 1, "x\n", "x", "cited")
    s2, s21 = sec(2, [2]), sec(3, [2, 1])
    carried = Payload("d", "code", ["obj_a"], "code", "c", 1, "c", [], "d#3", "text")
    leaf = Leaf([s2], [carried], np.zeros(2))
    chat = FakeChat()
    write_prose([Area([Topic([leaf])])], LLMWriter(chat, model="fake", cache_path=tmp_path / "w.json"))
    leaf_prompt = next(u for sys_, u in chat.calls if "KIND: leaf" in sys_)
    assert "d#obj_a" not in leaf_prompt


# --------------------------------------------------------------------------
# kb/2: one call per area, and the kb call reads areas
# --------------------------------------------------------------------------

def test_an_area_reply_is_asked_with_the_area_prompt_and_needs_differ(tmp_path):
    chat = FakeChat()
    w = LLMWriter(chat, model="fake", cache_path=tmp_path / "w.json")
    got = w.area([{"title": "Florp matters", "purpose": "Everything florp."}])
    assert got == AREA_REPLY
    system, user = chat.calls[0]
    assert system.startswith("KIND: area") and "Florp matters" in user
    bad = FakeChat(replies=[json.dumps(KB_REPLY), json.dumps(KB_REPLY)])
    with pytest.raises(BuildError, match="area"):
        LLMWriter(bad, model="fake", cache_path=tmp_path / "b.json").area(
            [{"title": "x", "purpose": "y"}])


def test_the_kb_call_reads_areas(tmp_path):
    chat = FakeChat()
    LLMWriter(chat, model="fake", cache_path=tmp_path / "w.json").kb(
        [{"title": "Florp lands", "purpose": "All florp topics."}])
    system, user = chat.calls[0]
    assert system.startswith("KIND: kb") and json.loads(user) == {
        "areas": [{"title": "Florp lands", "purpose": "All florp topics."}]}


def test_the_estimate_counts_one_call_per_area(tmp_path, monkeypatch, capsys):
    from conceptdrill.kb import build as B
    monkeypatch.setattr(B, "MAX_TOPICS_PER_AREA", 1)
    assert cli.main(["kb", "build", FLORP, ZIB, "--name", "est", "--library", str(LIB),
                     "--output-root", str(tmp_path), "--llm", "novita",
                     "--embedder", "hash", "--estimate"]) == 0
    out = capsys.readouterr().out
    m = re.search(r"(\d+) leaves in (\d+) topics in (\d+) areas", out)
    leaves, topics, areas = map(int, m.groups())
    assert areas == 3
    assert re.search(rf": {leaves + topics + areas + 1} calls", out)
