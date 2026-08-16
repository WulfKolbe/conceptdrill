"""Extraction, and the guards that keep a fixture out of a measurement."""
from __future__ import annotations

import json

import pytest

from conceptdrill.mathlayer import (ExtractionFailed, FixtureExtractor,
                                    LLMExtractor, NotMeasurementSafe,
                                    mathconcept_record, parse_reply,
                                    render_input, require_measurement_safe,
                                    symbol)

REPLY = json.dumps({"concepts": [
    {"concept": "Linear Map",
     "spoken": "T colon V right arrow W",
     "latex": [r"T : V \to W"],
     "symbols": [{"latex": "T", "role": "instance", "status": "observed"}],
     "related": [{"relation": "domain", "target": "vector space",
                  "status": "observed"}]},
]})


def chat_returning(reply):
    def chat(system, user):
        chat.seen = (system, user)
        return reply
    chat.params = {"model": "test", "temperature": 0.2}
    return chat


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_a_reply_becomes_a_validated_record():
    records = parse_reply(REPLY, span_id="s3", doc_id="fixture")
    assert len(records) == 1
    assert records[0]["concept"] == "linear map"          # canonicalised
    assert records[0]["evidence"]["span_ids"] == ["s3"]
    assert records[0]["confidence"] > 0


def test_a_fenced_reply_is_accepted():
    assert parse_reply(f"```json\n{REPLY}\n```")[0]["concept"] == "linear map"


def test_latex_survives_parsing_byte_for_byte():
    reply = json.dumps({"concepts": [{
        "concept": "linear map",
        "latex": [r"T \colon V \rightarrow W"],
        "symbols": [{"latex": r"T \colon V \rightarrow W",
                     "status": "observed"}],
        "related": []}]})
    record = parse_reply(reply)[0]
    assert record["latex"] == [r"T \colon V \rightarrow W"]
    assert record["symbols"][0]["latex"] == r"T \colon V \rightarrow W"


def test_a_concept_using_an_invented_relation_is_dropped_not_repaired():
    """Repairing it would mean choosing a status the model did not give."""
    reply = json.dumps({"concepts": [
        {"concept": "good", "symbols": [], "related":
         [{"relation": "domain", "target": "x", "status": "observed"}]},
        {"concept": "bad", "symbols": [], "related":
         [{"relation": "is-basically", "target": "x", "status": "observed"}]},
    ]})
    assert [r["concept"] for r in parse_reply(reply)] == ["good"]


def test_a_concept_using_an_invented_status_is_dropped():
    reply = json.dumps({"concepts": [
        {"concept": "bad", "related": [],
         "symbols": [{"latex": "T", "status": "definitely"}]}]})
    with pytest.raises(ExtractionFailed):
        parse_reply(reply)


@pytest.mark.parametrize("reply", ["", "   ", "not json at all",
                                   '{"other": []}', '{"concepts": []}'])
def test_an_unusable_reply_raises_rather_than_returning_empty(reply):
    """Silently returning no concepts is indistinguishable from a span that
    genuinely has none."""
    with pytest.raises(ExtractionFailed):
        parse_reply(reply)


def test_a_nameless_concept_is_skipped():
    reply = json.dumps({"concepts": [
        {"concept": "  ", "symbols": [], "related": []},
        {"concept": "basis", "symbols": [], "related": []}]})
    assert [r["concept"] for r in parse_reply(reply)] == ["basis"]


# --------------------------------------------------------------------------
# What the model is shown
# --------------------------------------------------------------------------

def test_latex_is_shown_separately_from_the_spoken_prose():
    r"""Speech cannot distinguish `\to` from `\rightarrow`, so the LaTeX has to
    reach the model by another route."""
    shown = render_input("Preliminaries",
                         "Let T colon V right arrow W be linear.",
                         [r"T : V \to W"])
    assert "right arrow" in shown
    assert r"T : V \to W" in shown


def test_no_latex_section_when_there_are_no_formulas():
    assert "LATEX" not in render_input("T", "prose only", [])


def test_the_prompt_names_exactly_the_schema_relations():
    """A prompt naming a relation the schema rejects would produce concepts
    that are silently dropped at parse time."""
    from conceptdrill.mathlayer import RELATIONS
    from conceptdrill.mathlayer.extract import PROMPT
    for kind in RELATIONS:
        assert kind in PROMPT


# --------------------------------------------------------------------------
# The extractors
# --------------------------------------------------------------------------

def test_the_llm_extractor_passes_prompt_and_input_through():
    chat = chat_returning(REPLY)
    extractor = LLMExtractor(chat, model="test")
    records = extractor.extract("s3", "Preliminaries",
                                "Let T colon V right arrow W be linear.",
                                latex=[r"T : V \to W"], doc_id="fixture")
    system, user = chat.seen
    assert "STATUS IS NOT OPTIONAL" in system
    assert r"T : V \to W" in user
    assert records[0]["concept"] == "linear map"


def test_an_empty_body_raises_rather_than_calling_the_model():
    def chat(system, user):
        raise AssertionError("must not be called")
    with pytest.raises(ExtractionFailed, match="empty body"):
        LLMExtractor(chat).extract("s1", "T", "   ")


def test_the_cache_signature_covers_decoding_parameters():
    """A cache built at one temperature must not serve a run made at another."""
    a = LLMExtractor(chat_returning(REPLY), model="test").cache_signature
    other = chat_returning(REPLY)
    other.params = {"model": "test", "temperature": 0.9}
    b = LLMExtractor(other, model="test").cache_signature
    assert a != b


def test_the_real_extractor_is_measurement_safe():
    assert require_measurement_safe(LLMExtractor(chat_returning(REPLY)))


def test_the_fixture_extractor_is_refused_by_a_measurement():
    """ExtractiveSummarizer used as a stand-in voided a 416-document corpus.
    The guard is mechanical here for the same reason."""
    fixture = FixtureExtractor([mathconcept_record(concept="basis")])
    assert fixture.measurement_safe is False
    with pytest.raises(NotMeasurementSafe):
        require_measurement_safe(fixture)


def test_the_fixture_extractor_replays_what_it_was_given():
    stored = [mathconcept_record(concept="vector space",
                                 symbols=[symbol("V", status="observed")])]
    got = FixtureExtractor(stored).extract("s1", "T", "body")
    assert got[0]["concept"] == "vector space"


def test_an_empty_fixture_raises():
    with pytest.raises(ExtractionFailed):
        FixtureExtractor([]).extract("s1", "T", "body")


def test_the_package_does_not_import_the_ces_machinery():
    """The layer must not reach the basis, projection or corpus store. Step 15
    measured CES retrieval as worse than uninformative for mathematics."""
    import conceptdrill.mathlayer as ml
    from pathlib import Path
    root = Path(ml.__file__).parent
    for source in root.glob("*.py"):
        text = source.read_text(encoding="utf-8")
        for banned in ("from ..hierarchy.basis", "from ..hierarchy.project",
                       "from ..hierarchy.corpus", "from ..embeddings",
                       "import numpy"):
            assert banned not in text, f"{source.name} reaches for {banned}"
