"""The real writer (T-0119, step C) -- tested with a fake chat, no key spent.

`LLMWriter` is the only stage of `kb build` that calls a model. It sees the
cited sections and the payload list, replies in JSON, and every reply is
cached by content, so a rebuild costs nothing and a failed build resumes.
"""
from __future__ import annotations

import json

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
    leaf = next((kb / "topics").rglob("florp-quibbling*.md"))
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
