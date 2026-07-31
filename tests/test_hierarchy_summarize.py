"""Unit tests for summarisation: the three tiers, the extractive floor, caching."""
from __future__ import annotations

import json

import pytest

from conceptdrill.hierarchy.summarize import (BASIS_TIER,
                                              EMBEDDING_TOKEN_WINDOW,
                                              ExtractiveSummarizer,
                                              SpanSummary, SummaryCache,
                                              TIER_WORDS, TOKENS_PER_WORD,
                                              load_prompt, summary_key)

BODY = (
    "Semantic projection maps a document object into a concept space. "
    "The concept space is built from the document's own structure rather than "
    "an external ontology. Each concept is embedded once and stored as a row "
    "of a matrix. Projection is then a single matrix multiplication, which "
    "keeps the space cheap to query. The approach is encoder agnostic and "
    "works with any embedding model that returns normalised vectors. "
    "We evaluate it on a corpus of scientific documents and report precision "
    "at k for each candidate source of concepts."
)


@pytest.fixture
def summarizer():
    return ExtractiveSummarizer()


# --------------------------------------------------------------------------
# The prompt shipped with the package
# --------------------------------------------------------------------------

def test_prompt_is_packaged_and_readable():
    """Nothing may depend on ~/Gemma4 at runtime."""
    assert "basis vectors" in load_prompt()


def test_prompt_ends_with_the_json_shape():
    """Format-final. Measured: a prompt ending in prose made the model reason
    aloud and emit no JSON at all -- 2/6 parsed versus 6/6 when the last thing
    it sees is the object it must produce."""
    assert load_prompt().strip().endswith("}")


def test_prompt_forbids_thinking_aloud():
    """The model literally wrote 'That's about 35 words. Good.' instead of JSON."""
    assert "explain your reasoning" in load_prompt()


def test_prompt_contains_no_markdown_fence():
    """Showing a fenced example while forbidding fences is a mixed signal, and
    models mirror the format they are shown."""
    assert "```" not in load_prompt()


def test_prompt_is_pure_ascii():
    """LLM-authored prose carries non-breaking hyphens and en dashes that are
    invisible on screen and tokenize differently."""
    assert all(ord(c) < 128 for c in load_prompt())


def test_prompt_states_budgets_inside_the_json_shape():
    """Budget placement is load-bearing: stated only in a distant description
    block, labels degraded to 8-word noun phrases."""
    tail = load_prompt().strip().rsplit("{", 1)[-1]
    assert "22-28 words" in tail


def test_prompt_asks_for_all_three_tiers():
    prompt = load_prompt()
    for tier in ("summary", "abstraction", "label"):
        assert f'"{tier}"' in prompt or f"{tier} " in prompt


def test_prompt_states_the_measured_label_budget():
    """22-28 words is 32-40 measured tokens, inside the 35-50 ceiling.

    The ceiling moved from 70 to 50: at 70 the observed output was p50 41,
    p90 50, max 61, so the band never bound and the vectors were longer than
    they needed to be.
    """
    assert "22-28 words" in load_prompt()


def test_prompt_forbids_backslashes():
    """A backslash silently corrupts the JSON string via legal escapes."""
    assert "backslash" in load_prompt().lower()


# --------------------------------------------------------------------------
# ExtractiveSummarizer
# --------------------------------------------------------------------------

def test_produces_all_three_tiers(summarizer):
    got = summarizer.summarize("s1", "Semantic Projection", BODY)
    assert got.summary and got.abstraction and got.label


def test_is_marked_deterministic(summarizer):
    assert summarizer.summarize("s1", "T", BODY).deterministic


def test_is_actually_deterministic(summarizer):
    a = summarizer.summarize("s1", "T", BODY)
    b = summarizer.summarize("s1", "T", BODY)
    assert a == b


def test_label_respects_its_word_budget(summarizer):
    got = summarizer.summarize("s1", "Semantic Projection", BODY)
    assert len(got.label.split()) <= TIER_WORDS["label"][1]


def test_label_leads_with_the_title(summarizer):
    """The concept must be named, not merely described."""
    got = summarizer.summarize("s1", "Semantic Projection", BODY)
    assert got.label.startswith("Semantic Projection")


def test_every_tier_fits_the_embedding_window(summarizer):
    """The contract that replaced "summary is longer than label".

    All three budgets now fit 70 tokens, because the embedder averages over
    whatever it is given and a 116-token summary dilutes the concept it was
    meant to carry. Length no longer distinguishes the tiers; role and form
    do, and `check_tier_independence` is what enforces that.
    """
    got = summarizer.summarize("s1", "T", BODY)
    for tier, (lo, hi) in TIER_WORDS.items():
        words = len(getattr(got, tier).split())
        assert words <= hi, f"{tier}: {words} words, budget {lo}-{hi}"
        assert hi * TOKENS_PER_WORD <= EMBEDDING_TOKEN_WINDOW[1] + 1, (
            f"{tier} budget of {hi} words is "
            f"{hi * TOKENS_PER_WORD:.0f} tokens, over the window")


def test_the_token_estimate_is_the_measured_one():
    """1.604 was an estimate carried since the design spec and is 11% high.
    Measured on 786 cached summaries with the embedder's own tokenizer."""
    assert TOKENS_PER_WORD == 1.441
    assert round(EMBEDDING_TOKEN_WINDOW[1] / TOKENS_PER_WORD) == 35


def test_tiers_cut_at_sentence_boundaries(summarizer):
    """A fragment embeds badly, and the whole corpus compares against it."""
    got = summarizer.summarize("s1", "T", BODY)
    assert got.summary.rstrip().endswith((".", "!", "?"))


def test_empty_body_falls_back_to_the_title(summarizer):
    got = summarizer.summarize("s1", "Concept Scoring", "")
    assert got.label == "Concept Scoring"
    assert got.warnings


def test_empty_body_is_still_usable(summarizer):
    assert summarizer.summarize("s1", "Concept Scoring", "").is_usable


def test_whitespace_is_normalised(summarizer):
    got = summarizer.summarize("s1", "T", "one\n\n  two\tthree")
    assert "\n" not in got.summary and "  " not in got.summary


# --------------------------------------------------------------------------
# SpanSummary
# --------------------------------------------------------------------------

def test_basis_text_is_the_label_tier():
    s = SpanSummary("s1", "T", summary="s", abstraction="a", label="l")
    assert BASIS_TIER == "label" and s.basis_text == "l"


def test_basis_text_falls_back_when_the_label_is_missing():
    """A missing label must not yield an empty basis vector."""
    s = SpanSummary("s1", "T", summary="s", abstraction="a", label="")
    assert s.basis_text == "a"
    assert SpanSummary("s1", "T", summary="s").basis_text == "s"


def test_a_summary_with_an_error_is_not_usable():
    assert not SpanSummary("s1", "T", label="l", error="timeout").is_usable


def test_an_empty_summary_is_not_usable():
    assert not SpanSummary("s1", "T").is_usable


def test_tier_fit_reports_under_ok_and_over():
    short = SpanSummary("s1", "T", label="one two")
    good = SpanSummary("s1", "T", label=" ".join(["w"] * 25))
    long = SpanSummary("s1", "T", label=" ".join(["w"] * 99))
    assert short.tier_fit()["label"] == "under"
    assert good.tier_fit()["label"] == "ok"
    assert long.tier_fit()["label"] == "over"


def test_word_counts_are_reported():
    s = SpanSummary("s1", "T", label="a b c")
    assert s.word_counts()["label"] == 3


# --------------------------------------------------------------------------
# SummaryCache
# --------------------------------------------------------------------------

def test_key_is_stable_for_identical_input():
    assert summary_key("t", "b", "m", "p") == summary_key("t", "b", "m", "p")


def test_key_changes_with_the_body():
    assert summary_key("t", "b1", "m", "p") != summary_key("t", "b2", "m", "p")


def test_key_changes_with_the_model():
    assert summary_key("t", "b", "m1", "p") != summary_key("t", "b", "m2", "p")


def test_key_changes_with_the_prompt():
    """New instructions produce different output; an old summary must not be
    served against a new prompt."""
    assert summary_key("t", "b", "m", "p1") != summary_key("t", "b", "m", "p2")


def test_cache_round_trips(tmp_path):
    cache = SummaryCache(tmp_path / "s.json")
    s = SpanSummary("s1", "T", summary="x", abstraction="y", label="z")
    cache.put("k", s)
    cache.flush()
    assert SummaryCache(tmp_path / "s.json").get("k") == s


def test_cache_miss_returns_none(tmp_path):
    assert SummaryCache(tmp_path / "s.json").get("absent") is None


def test_cache_survives_a_corrupt_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text("this is not json", encoding="utf-8")
    assert SummaryCache(path).get("k") is None


def test_cache_write_is_atomic(tmp_path):
    """A crash mid-write must not leave a half-file that poisons later runs."""
    path = tmp_path / "s.json"
    cache = SummaryCache(path)
    cache.put("k", SpanSummary("s1", "T", label="z"))
    cache.flush()
    assert json.loads(path.read_text())["k"]["label"] == "z"
    assert not path.with_name(path.name + ".tmp").exists()


def test_flush_without_changes_writes_nothing(tmp_path):
    path = tmp_path / "s.json"
    SummaryCache(path).flush()
    assert not path.exists()


# --------------------------------------------------------------------------
# summarize_tree — integration
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.docmodel_tree import build_tree          # noqa: E402
from conceptdrill.hierarchy.summarize import SummaryRun, summarize_tree  # noqa: E402


def _sec(sid, caption, level, flow, **p):
    return {"id": sid, "type": "Section",
            "props": {"caption": caption, "level": level, "flow_index": flow, **p}}


def _par(pid, text, flow, parent):
    return {"id": pid, "type": "Paragraph",
            "props": {"text": text, "flow_index": flow, "parent_section": parent}}


@pytest.fixture
def tree():
    return build_tree({"meta": {"bibkey": "t"}, "objects": [
        _sec("s1", "Method", 2, 1),
        _par("p1", "The method embeds each span once.", 2, "s1"),
        _sec("s2", "Scoring", 3, 3),
        _par("p2", "Scoring combines structural weight and coverage.", 4, "s2"),
    ]})


def test_every_section_is_summarised(tree, summarizer):
    run = summarize_tree(tree, summarizer)
    assert set(run.summaries) == {"s1", "s2"}


def test_level_filter_restricts_the_run(tree, summarizer):
    run = summarize_tree(tree, summarizer, levels={2})
    assert set(run.summaries) == {"s1"}


def test_a_parent_is_summarised_from_its_own_paragraphs_only(tree, summarizer):
    """The unit is the smallest set of paragraphs under one heading.

    Given `\\span{A} P0 \\subsection{B} P1`, A is P0 -- not P0+P1. Sending
    the subtree summarised every child's text twice, once under itself and
    once under its parent, and made parent and child share source text, which
    manufactures parent-child merges in the basis.
    """
    run = summarize_tree(tree, summarizer)
    parent = run.summaries["s1"].summary
    child = run.summaries["s2"].summary
    assert "Scoring combines" in child
    assert "Scoring combines" not in parent


def test_a_heading_with_no_paragraphs_of_its_own_is_not_summarised(summarizer):
    """A title is not a concept, and there is nothing else there. Recorded,
    not silently skipped."""
    t = build_tree({"objects": [
        _sec("s1", "Parent", 2, 1),
        _sec("s2", "Child", 3, 2),
        _par("p1", "Only the child has paragraphs of its own.", 3, "s2")]})
    run = summarize_tree(t, summarizer)
    assert set(run.summaries) == {"s2"}
    assert run.empty == ("s1",)
    assert run.stats()["empty"] == 1


def test_every_section_is_accounted_for_as_summarised_or_empty(summarizer):
    t = build_tree({"objects": [
        _sec("s1", "Parent", 2, 1),
        _sec("s2", "Child", 3, 2),
        _par("p1", "child text here", 3, "s2")]})
    run = summarize_tree(t, summarizer)
    assert len(run.summaries) + len(run.empty) == len(t)


def test_degraded_title_is_restored_for_the_model(summarizer):
    t = build_tree({"objects": [_sec("s1", "\\ALG\\ Application", 2, 1),
                                _par("p1", "some body text here", 2, "s1")]})
    run = summarize_tree(t, summarizer)
    assert "\\ALG" in run.summaries["s1"].title


def test_run_reports_determinism(tree, summarizer):
    assert summarize_tree(tree, summarizer).deterministic


def test_cache_is_populated_then_reused(tree, summarizer, tmp_path):
    cache = SummaryCache(tmp_path / "c.json")
    first = summarize_tree(tree, summarizer, cache=cache)
    assert first.generated == 2 and first.cached == 0

    second = summarize_tree(tree, summarizer, cache=SummaryCache(tmp_path / "c.json"))
    assert second.cached == 2 and second.generated == 0
    assert second.summaries["s1"] == first.summaries["s1"]


def test_failures_are_recorded_and_do_not_stop_the_run(tree):
    class Broken:
        name, deterministic = "broken", False

        def summarize(self, sid, title, body):
            return SpanSummary(sid, title, error="boom")

    run = summarize_tree(tree, Broken())
    assert len(run.summaries) == 2
    assert set(run.failed) == {"s1", "s2"}
    assert run.usable() == {}


def test_unusable_summaries_are_not_cached(tree, tmp_path):
    class Broken:
        name, deterministic = "broken", False

        def summarize(self, sid, title, body):
            return SpanSummary(sid, title, error="boom")

    cache = SummaryCache(tmp_path / "c.json")
    summarize_tree(tree, Broken(), cache=cache)
    assert len(cache) == 0


def test_stats_report_the_basis_word_budget(tree, summarizer):
    stats = summarize_tree(tree, summarizer).stats()
    assert stats["spans"] == 2
    assert "label_word_budget" in stats


def test_progress_callback_sees_every_section(tree, summarizer):
    seen = []
    summarize_tree(tree, summarizer, progress=lambda n, s: seen.append(n.id))
    assert seen == ["s1", "s2"]


def test_empty_tree_yields_an_empty_run(summarizer):
    run = summarize_tree(build_tree({"objects": []}), summarizer)
    assert run.summaries == {} and run.failed == ()


# --------------------------------------------------------------------------
# Truncation must never be invisible
# --------------------------------------------------------------------------

def test_the_hash_embedder_has_no_token_report():
    """token_report belongs to the transformer path; the passthrough must not
    invent one for a backend that does not tokenize."""
    from conceptdrill.embeddings import get_embedder
    e = get_embedder("hash", cache=False)
    assert getattr(e, "token_report", None) is None or e.token_report([]) == {}


def test_budgets_stay_inside_the_window():
    """A future edit to TIER_WORDS that breaks the 70-token ceiling should
    fail here rather than silently produce diluted vectors."""
    for tier, (lo, hi) in TIER_WORDS.items():
        assert lo < hi, tier
        assert hi * TOKENS_PER_WORD <= EMBEDDING_TOKEN_WINDOW[1] + 1, (
            f"{tier}: {hi} words is {hi * TOKENS_PER_WORD:.0f} tokens")


def test_cache_revives_sibling_concepts_as_objects(tmp_path):
    """asdict flattens nested summaries to dicts and tuples to lists. A naive
    SpanSummary(**raw) returns siblings as dicts -- objects with no .label
    that every consumer then fails on."""
    from conceptdrill.hierarchy.summarize import SummaryCache
    inner = SpanSummary(span_id="s1", title="T", label="second phrase")
    outer = SpanSummary(span_id="s1", title="T", label="first phrase",
                           siblings=(inner,))
    cache = SummaryCache(tmp_path / "c.json")
    cache.put("k", outer)
    cache.flush()

    reread = SummaryCache(tmp_path / "c.json").get("k")
    assert reread == outer
    assert isinstance(reread.siblings, tuple)
    assert isinstance(reread.siblings[0], SpanSummary)
    assert [c.label for c in reread.concepts] == ["first phrase", "second phrase"]


def test_noop_macros_are_stripped_before_speaking():
    r"""SRE read `\ensuremath{X \rightarrow Y}\xspace` as "X right arrow Y
    backslash xspace": a spacing macro became two words inside the corpus's
    central claim."""
    from conceptdrill.hierarchy.mathtext import strip_noop_macros
    assert strip_noop_macros(r"\ensuremath{X \rightarrow Y}\xspace") == r"X \rightarrow Y"
    assert strip_noop_macros(r"A \quad B") == "A B"
    assert strip_noop_macros(r"\mbox{text} here") == "text here"


def test_stripping_leaves_real_macros_alone():
    from conceptdrill.hierarchy.mathtext import strip_noop_macros
    assert strip_noop_macros(r"\sum_{i} \alpha_i") == r"\sum_{i} \alpha_i"
