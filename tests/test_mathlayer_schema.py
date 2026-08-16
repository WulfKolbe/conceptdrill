"""The MathConcept contract.

The contract tests come first because they are the contract; the arithmetic of
`confidence` follows because it is one consequence of it.
"""
from __future__ import annotations

import pytest

from conceptdrill.mathlayer import (RELATIONS, SchemaViolation, anchor,
                                    assert_record, check_record, concept_id,
                                    confidence, evidence, mathconcept_record,
                                    relation, symbol)


def observed_map():
    """The fixture's `linear map`, as the layer would build it."""
    return mathconcept_record(
        concept="linear map",
        spoken="T colon V right arrow W",
        latex=[r"T : V \to W", r"T \colon V \rightarrow W"],
        symbols=[symbol("T", role="instance", status="observed")],
        related=[relation("domain", "vector space", status="observed")],
        lean=[anchor("LinearMap", status="supported", hits=11118)],
        evidence=evidence(doc_id="fixture", span_ids=["s3"],
                          quote=r"Let T : V \to W be linear."),
    )


# --------------------------------------------------------------------------
# The contract
# --------------------------------------------------------------------------

def test_an_unknown_field_raises_rather_than_being_kept():
    """A typo must not produce a record with a hole in it. See runlog.py."""
    with pytest.raises(SchemaViolation, match="unknown MathConcept fields"):
        mathconcept_record(concept="basis", confidance=0.5)


def test_every_field_is_present_even_when_unset():
    from conceptdrill.mathlayer import MATHCONCEPT_FIELDS
    record = mathconcept_record(concept="basis")
    assert set(record) == set(MATHCONCEPT_FIELDS)


def test_a_missing_list_becomes_empty_not_none():
    record = mathconcept_record(concept="basis")
    for field in ("latex", "symbols", "related", "lean", "loogle"):
        assert record[field] == []


def test_a_concept_needs_a_name():
    with pytest.raises(SchemaViolation, match="non-empty concept"):
        mathconcept_record(concept="   ")


def test_identity_is_content_addressed_not_positional():
    """A positional id transposes the moment a document is re-read."""
    assert concept_id("linear map") == concept_id("  Linear   Map  ")
    assert concept_id("linear map") != concept_id("linear order")
    assert concept_id("linear map").startswith("mc_")


def test_a_hand_written_id_that_disagrees_is_caught():
    record = observed_map()
    record["id"] = "mc_000000000000"
    assert any("content address" in p for p in check_record(record))


def test_the_name_is_canonicalised_on_the_record():
    assert mathconcept_record(concept="  Vector   Space ")["concept"] == \
        "vector space"


# --------------------------------------------------------------------------
# Status is not optional
# --------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["seen", "true", "", "OBSERVED"])
def test_an_invented_symbol_status_raises(bad):
    with pytest.raises(SchemaViolation, match="status"):
        symbol("T", status=bad)


def test_an_invented_relation_raises():
    with pytest.raises(SchemaViolation, match="relation"):
        relation("is-basically", "vector space")


def test_every_documented_relation_is_accepted():
    for kind in RELATIONS:
        assert relation(kind, "x")["relation"] == kind


def test_a_symbol_needs_latex():
    with pytest.raises(SchemaViolation, match="latex"):
        symbol("   ", status="observed")


def test_latex_is_preserved_exactly_and_never_normalised():
    r"""`\to` and `\rightarrow` speak identically, so the spelling is the only
    surviving evidence of what the author wrote."""
    for form in (r"T : V \to W", r"T \colon V \rightarrow W"):
        assert symbol(form, status="observed")["latex"] == form


def test_an_invented_anchor_status_raises():
    with pytest.raises(SchemaViolation, match="anchor status"):
        anchor("Module", status="probably")


def test_unverified_is_the_default_for_an_anchor():
    """Measured: two of five names asserted from memory did not exist."""
    assert anchor("Module")["status"] == "unverified"


# --------------------------------------------------------------------------
# Confidence
# --------------------------------------------------------------------------

def test_confidence_is_maximal_when_every_check_passes():
    assert confidence(observed_map()) == 1.0


def test_an_implied_relation_scores_below_an_observed_one():
    implied = mathconcept_record(
        concept="field",
        symbols=[symbol("K", status="observed")],
        related=[relation("related-to", "vector space", status="implied")],
        lean=[anchor("Field", status="supported", hits=9093)])
    assert confidence(implied) == pytest.approx(0.8)
    assert confidence(implied) < confidence(observed_map())


def test_a_concept_absent_from_the_text_scores_low():
    """`basis` is reachable from `n-dimensional` and is not in the fixture."""
    predicted = mathconcept_record(
        concept="basis",
        symbols=[symbol(r"\mathcal{B}", status="predicted")],
        related=[relation("related-to", "dimension", status="external")],
        lean=[anchor("Module.Basis", status="supported", hits=1493)])
    assert confidence(predicted) == pytest.approx(0.3)


def test_the_strongest_relation_decides_not_the_average():
    """One observed relation establishes the text talks about the concept;
    averaging would let background `external` facts dilute it away."""
    record = mathconcept_record(
        concept="linear map",
        symbols=[symbol("T", status="observed")],
        related=[relation("domain", "vector space", status="observed"),
                 relation("related-to", "category", status="external"),
                 relation("related-to", "matrix", status="external")],
        lean=[anchor("LinearMap", status="supported", hits=1)])
    assert confidence(record) == 1.0


def test_no_expected_anchor_renormalises_instead_of_scoring_zero():
    """The domain of a map is a type index, not a Mathlib name. That is a fact
    about Mathlib, not a failed lookup -- the absorbed/not_integrated
    distinction from runlog.py, applied here."""
    domain = mathconcept_record(
        concept="domain",
        symbols=[symbol("V", status="observed")],
        related=[relation("domain", "linear map", status="implied")],
        lean=[anchor("whatever", status="not_expected")])
    assert confidence(domain) == pytest.approx(0.75)

    failed = mathconcept_record(
        concept="domain",
        symbols=[symbol("V", status="observed")],
        related=[relation("domain", "linear map", status="implied")],
        lean=[anchor("Nonexistent", status="unresolved")])
    assert confidence(failed) < confidence(domain)


def test_an_unresolved_anchor_costs_confidence():
    supported = observed_map()
    broken = mathconcept_record(**{**{k: v for k, v in observed_map().items()
                                      if k not in ("confidence", "id")},
                                   "lean": [anchor("Basis", status="unresolved",
                                                   error="unknown identifier")]})
    assert confidence(broken) < confidence(supported)


def test_an_unchecked_anchor_sits_between_supported_and_rejected():
    """A verification pass that cannot LOWER a score is not a check.

    If `unverified` scored the same as `unresolved`, running `verify()` could
    only leave confidence flat or raise it, and discovering that a proposed
    name does not exist would be invisible. It is also the
    absorbed/not_integrated distinction: "nobody looked" and "we looked and it
    is not there" are different states.
    """
    def with_anchor(status):
        return confidence(mathconcept_record(
            concept="basis",
            symbols=[symbol("V", status="observed")],
            related=[relation("related-to", "dimension", status="observed")],
            lean=[anchor("Module.Basis", status=status)]))

    assert with_anchor("unresolved") < with_anchor("unverified") \
        < with_anchor("supported")


def test_the_best_anchor_decides_not_the_worst():
    """One name that resolves is formal support; a wrong guess beside it is
    not evidence against the concept."""
    record = mathconcept_record(
        concept="basis",
        symbols=[symbol("V", status="observed")],
        related=[relation("related-to", "dimension", status="observed")],
        lean=[anchor("Basis", status="unresolved", error="unknown identifier"),
              anchor("Module.Basis", status="supported", hits=1493)])
    assert confidence(record) == 1.0


def test_confidence_is_derived_when_not_supplied():
    assert observed_map()["confidence"] == 1.0


@pytest.mark.parametrize("score", [-0.1, 1.5, "high", None])
def test_a_confidence_outside_the_unit_interval_is_reported(score):
    record = observed_map()
    record["confidence"] = score
    assert any("confidence" in p for p in check_record(record))


# --------------------------------------------------------------------------
# Whole records
# --------------------------------------------------------------------------

def test_a_well_formed_record_reports_nothing():
    assert check_record(observed_map()) == []


def test_assert_record_raises_on_a_bad_one():
    record = observed_map()
    record["related"] = [{"relation": "is-basically", "target": "x",
                          "status": "observed"}]
    with pytest.raises(SchemaViolation):
        assert_record(record)


def test_ces_row_ids_are_recorded_but_carry_no_weight():
    """Step 15: CES retrieval was worse than uninformative for mathematics, so
    provenance must not become a score."""
    without = mathconcept_record(
        concept="linear map",
        symbols=[symbol("T", status="observed")],
        related=[relation("domain", "vector space", status="observed")],
        lean=[anchor("LinearMap", status="supported", hits=1)])
    with_rows = mathconcept_record(
        concept="linear map",
        symbols=[symbol("T", status="observed")],
        related=[relation("domain", "vector space", status="observed")],
        lean=[anchor("LinearMap", status="supported", hits=1)],
        evidence=evidence(ces_row_ids=["row_a", "row_b", "row_c"]))
    assert confidence(with_rows) == confidence(without)
