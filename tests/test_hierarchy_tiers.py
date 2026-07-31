"""Tier independence, and keeping the fixture out of measurements.

The `ExtractiveSummarizer` tests here are not incidental. They pin down *why*
it is a fixture: it fails the independence contract by construction, and a test
that asserts so is what stops it drifting back into a measurement path.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from conceptdrill.hierarchy.summarize import (MAX_TIER_JACCARD, ExtractiveSummarizer,
                                              SpanSummary, TierDegeneracy,
                                              TitleOnlySummarizer,
                                              assert_tier_independence,
                                              check_tier_independence, jaccard,
                                              token_set)

REPO = Path(__file__).resolve().parents[1]

BODY = ("Temporal query intent classification decides whether a keyword query "
        "has a temporal dimension. It matters for retrieval. Systems classify "
        "queries into past, recency, future and atemporal categories. The task "
        "was introduced at NTCIR. Features come from query linguistics.")


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "hierarchy_run", REPO / "tools" / "hierarchy_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------
# The measure
# --------------------------------------------------------------------------

def test_token_set_is_lowercased_alphanumerics():
    assert token_set("The $tau$ Function, 2.") == {"the", "tau", "function", "2"}


def test_jaccard_of_identical_text_is_one():
    assert jaccard("alpha beta", "beta alpha") == 1.0


def test_jaccard_of_disjoint_text_is_zero():
    assert jaccard("alpha beta", "gamma delta") == 0.0


def test_jaccard_with_an_empty_side_is_zero():
    """An absent tier must not be scored as maximally different."""
    assert jaccard("", "anything at all") == 0.0


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

def summary(**kw):
    base = dict(span_id="s1", title="T")
    base.update(kw)
    return SpanSummary(**base)


def test_independent_tiers_pass():
    got = summary(
        summary="The system classifies keyword queries by temporal intent, "
                "assigning each to past, recency, future or atemporal.",
        abstraction="Assigning search requests to time orientations using "
                    "linguistic and corpus evidence.",
        label="temporal orientation assignment for information retrieval "
              "requests")
    assert check_tier_independence(got) == []


def test_a_prefix_relation_is_caught():
    got = summary(summary="alpha beta gamma delta epsilon zeta",
                  abstraction="alpha beta gamma",
                  label="wholly different words entirely")
    problems = check_tier_independence(got)
    assert any("prefix" in p for p in problems)


def test_high_overlap_without_a_prefix_is_caught():
    """Same words, reordered: the prefix test alone would pass this."""
    got = summary(summary="alpha beta gamma delta",
                  abstraction="delta gamma beta alpha",
                  label="entirely unrelated vocabulary here")
    problems = check_tier_independence(got)
    assert any("Jaccard" in p for p in problems)
    assert not any("prefix" in p for p in problems)


def test_the_threshold_is_exclusive():
    """Exactly at the threshold passes; above it fails."""
    got = summary(summary="a b c d e", abstraction="a b c f g")
    assert jaccard(got.summary, got.abstraction) < MAX_TIER_JACCARD
    assert check_tier_independence(got) == []


def test_a_single_tier_is_never_compared():
    assert check_tier_independence(summary(label="only this one")) == []


def test_absent_tiers_are_not_compared():
    got = summary(label="a canonical noun phrase", abstraction="", summary="")
    assert check_tier_independence(got) == []


def test_assert_raises_with_the_section_id(monkeypatch):
    got = summary(summary="alpha beta gamma", abstraction="alpha beta")
    with pytest.raises(TierDegeneracy, match="s1"):
        assert_tier_independence(got)


def test_assert_is_silent_when_independent():
    assert_tier_independence(summary(label="a canonical noun phrase")) is None


# --------------------------------------------------------------------------
# Why ExtractiveSummarizer is a fixture
# --------------------------------------------------------------------------

def test_the_extractive_fixture_fails_the_contract_by_construction():
    """Its three tiers are cut points on one string. This is the whole reason
    it is out of the measurement path."""
    got = ExtractiveSummarizer().summarize("s1", "1 Introduction", BODY)
    problems = check_tier_independence(got)
    assert problems, "the fixture is supposed to be degenerate"
    assert any("prefix" in p or "Jaccard" in p for p in problems)


def test_the_extractive_fixture_is_flagged_unsafe():
    assert ExtractiveSummarizer.measurement_safe is False


def test_the_ablation_arm_is_flagged_safe_and_as_an_ablation():
    assert TitleOnlySummarizer.measurement_safe is True
    assert TitleOnlySummarizer.is_ablation is True


def test_the_ablation_arm_emits_only_a_label():
    got = TitleOnlySummarizer().summarize("s1", "3 Related Work", BODY)
    assert got.label == "3 Related Work"
    assert got.abstraction == "" and got.summary == ""
    assert check_tier_independence(got) == []


def test_the_ablation_arm_ignores_the_body_entirely():
    """If it read the body it would not be an ablation."""
    a = TitleOnlySummarizer().summarize("s1", "Method", "one body")
    b = TitleOnlySummarizer().summarize("s1", "Method", "a completely different body")
    assert a.label == b.label


def test_a_titleless_section_says_so_rather_than_inventing_one():
    got = TitleOnlySummarizer().summarize("s1", "", BODY)
    assert got.label == ""
    assert got.warnings == ("span has no title",)


# --------------------------------------------------------------------------
# The runner refuses substitutes
# --------------------------------------------------------------------------

def test_the_runner_refuses_the_fixture():
    runner = _load_runner()
    with pytest.raises(runner.NotMeasurementSafe, match="unknown summariser"):
        runner.make_summarizer("extractive", "")


def test_the_runner_refuses_an_unknown_summariser():
    runner = _load_runner()
    with pytest.raises(runner.NotMeasurementSafe):
        runner.make_summarizer("something-else", "")


def test_the_runner_accepts_the_ablation_arm():
    runner = _load_runner()
    assert isinstance(runner.make_summarizer("title-only", ""), TitleOnlySummarizer)


def test_no_credentials_raises_rather_than_falling_back(monkeypatch, tmp_path):
    """The clause that matters: unreachable LLM must stop the run, not
    silently produce something else."""
    runner = _load_runner()
    monkeypatch.delenv("NOVITA_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)          # no .env to read
    with pytest.raises(RuntimeError, match="no API key"):
        runner.make_summarizer("novita", "")


# --------------------------------------------------------------------------
# One entry per concept
# --------------------------------------------------------------------------

import json  # noqa: E402

from conceptdrill.hierarchy.novita import NovitaSummarizer  # noqa: E402


def reply_with(*entries):
    return json.dumps({"concepts": list(entries)})


def entry(label, abstraction="an abstraction", summary="a summary"):
    return {"label": label, "abstraction": abstraction, "summary": summary}


def summarize(reply):
    return NovitaSummarizer(lambda a, b: reply, model="stub").summarize(
        "s1", "Title", "body")


def test_a_single_concept_yields_no_siblings():
    got = summarize(reply_with(entry("one canonical noun phrase")))
    assert got.label == "one canonical noun phrase"
    assert got.siblings == ()
    assert len(got.concepts) == 1


def test_several_concepts_become_several_summaries():
    """A span defining three ideas used to yield one label -- a compromise
    matching none of them, and one basis row where CES wants three."""
    got = summarize(reply_with(entry("first phrase"), entry("second phrase"),
                               entry("third phrase")))
    assert [c.label for c in got.concepts] == ["first phrase", "second phrase",
                                               "third phrase"]


def test_the_dominant_concept_is_first():
    got = summarize(reply_with(entry("dominant"), entry("secondary")))
    assert got.label == "dominant"
    assert got.siblings[0].label == "secondary"


def test_siblings_never_nest():
    """Otherwise `concepts` would need a recursive walk and a caller could
    silently miss half of them."""
    got = summarize(reply_with(entry("a"), entry("b"), entry("c")))
    assert all(s.siblings == () for s in got.siblings)


def test_every_concept_carries_its_own_tiers():
    got = summarize(reply_with(
        entry("first phrase", "first abstraction", "first summary"),
        entry("second phrase", "second abstraction", "second summary")))
    second = got.siblings[0]
    assert second.abstraction == "second abstraction"
    assert second.summary == "second summary"


def test_every_concept_is_sanitised():
    """Sanitising the dominant entry and not the rest would put invisible
    characters into the basis through the side door."""
    got = summarize(reply_with(entry("clean phrase"),
                               entry("phrase with a‑non‑breaking hyphen")))
    assert "‑" not in got.siblings[0].label


def test_each_concept_is_independently_usable():
    got = summarize(reply_with(entry("first phrase"), entry("second phrase")))
    for c in got.concepts:
        assert c.is_usable
        assert c.span_id == "s1"


def test_the_flat_single_object_reply_still_parses():
    """Cached replies from the previous prompt must not become errors."""
    got = summarize(json.dumps({"summary": "s", "abstraction": "a", "label": "l"}))
    assert got.label == "l" and got.siblings == ()


def test_an_empty_concepts_list_is_an_error_not_a_silent_pass():
    got = summarize(json.dumps({"concepts": []}))
    assert got.error


def test_a_non_dict_entry_is_ignored_rather_than_crashing():
    got = summarize(json.dumps({"concepts": ["not an object",
                                             entry("real phrase")]}))
    assert got.label == "real phrase"


# --------------------------------------------------------------------------
# Input truncation must be visible
# --------------------------------------------------------------------------

from conceptdrill.hierarchy.novita import (DEFAULT_MAX_BODY,  # noqa: E402
                                           DEFAULT_MAX_TOKENS,
                                           MODEL_CONTEXT_TOKENS,
                                           MODEL_MAX_OUTPUT_TOKENS)


def test_the_completion_ceiling_is_the_models_own_maximum():
    """A ceiling costs nothing unspent, so sitting below the provider limit
    only converts long replies into failed spans."""
    assert DEFAULT_MAX_TOKENS == MODEL_MAX_OUTPUT_TOKENS == 32_768


def test_the_body_cap_is_a_small_fraction_of_the_context():
    """It was 6000 characters against a 262144-token context -- 0.8% of what
    the model can read."""
    assert DEFAULT_MAX_BODY == 200_000
    assert DEFAULT_MAX_BODY / 3.1 < MODEL_CONTEXT_TOKENS - MODEL_MAX_OUTPUT_TOKENS


def test_a_body_within_the_cap_reports_nothing_dropped():
    s = NovitaSummarizer(lambda a, b: "{}", model="stub", max_body=100)
    prompt, dropped = s.build_user_prompt_traced("T", "x" * 50)
    assert dropped == 0
    assert prompt.endswith("x" * 50)


def test_a_body_over_the_cap_reports_how_much_was_dropped():
    """Cutting the input silently is how a span came to be summarised from
    its opening pages with nothing in the record to say so."""
    s = NovitaSummarizer(lambda a, b: "{}", model="stub", max_body=100)
    prompt, dropped = s.build_user_prompt_traced("T", "x" * 250)
    assert dropped == 150
    assert prompt.count("x") == 100


def test_input_truncation_reaches_the_summary_as_a_warning():
    reply = reply_with(entry("a canonical noun phrase about indexing"))
    s = NovitaSummarizer(lambda a, b: reply, model="stub", max_body=100)
    got = s.summarize("s1", "T", "x" * 250)
    assert any("input truncated" in w and "150 characters" in w
               for w in got.warnings)


def test_every_concept_carries_the_input_truncation_warning():
    """Not just the dominant one: all of them were built from the cut body."""
    reply = reply_with(entry("first phrase"), entry("second phrase"))
    s = NovitaSummarizer(lambda a, b: reply, model="stub", max_body=100)
    got = s.summarize("s1", "T", "x" * 250)
    assert all(any("input truncated" in w for w in c.warnings)
               for c in got.concepts)


def test_a_failed_call_still_reports_input_truncation():
    def boom(a, b):
        raise RuntimeError("network")
    s = NovitaSummarizer(boom, model="stub", max_body=100)
    got = s.summarize("s1", "T", "x" * 250)
    assert got.error
    assert any("input truncated" in w for w in got.warnings)
