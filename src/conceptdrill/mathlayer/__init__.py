"""An experimental layer above CES: mathematical concepts and formal anchors.

CES answers what a span is about. It does not answer which mathematical object
the span discusses, and cannot: a basis row is a 20-28 word English noun phrase
embedded by a sentence encoder, and `\\dim V = n` is not one.

This package reads span text, spoken mathematics and formula LaTeX, and emits
`MathConcept` records naming concepts, the symbols standing for them, the
relations the text licenses, and candidate Mathlib declarations.

**It imports nothing from `conceptdrill.hierarchy` except `sanitize`.** No
basis, no projection, no corpus store. That is not tidiness: step 15 of the
experiment measured CES retrieval as worse than uninformative for mathematics
— one basis row was top-1 for four of six mathematical continuations, and the
wrong answers outscored the right ones — so a CES coordinate must never rank,
gate or select a MathConcept. `evidence.ces_row_ids` is provenance and nothing
here reads it.

See `docs/math-layer.md` for the design and
`experiments/math-layer/neighbourhood.md` for the measurements.
"""
from .extract import (Extractor, ExtractionFailed, FixtureExtractor,
                      LLMExtractor, NotMeasurementSafe, parse_reply,
                      render_input, require_measurement_safe)
from .loogle import (HttpLoogleClient, LoogleUnavailable, verified_against,
                     verify, verify_name)
from .schema import (ANCHOR_STATUS, EVIDENCE_STATUS, MATHCONCEPT_FIELDS,
                     RELATIONS, SchemaViolation, anchor, assert_record,
                     canonical_concept, check_record, concept_id, confidence,
                     evidence, mathconcept_record, relation, symbol)
from .store import MathConceptStore, StoreMismatch, merge

__all__ = [
    "ANCHOR_STATUS", "EVIDENCE_STATUS", "MATHCONCEPT_FIELDS", "RELATIONS",
    "Extractor", "ExtractionFailed", "FixtureExtractor", "HttpLoogleClient",
    "LLMExtractor", "LoogleUnavailable", "MathConceptStore",
    "NotMeasurementSafe", "SchemaViolation", "StoreMismatch",
    "anchor", "assert_record", "canonical_concept", "check_record",
    "concept_id", "confidence", "evidence", "mathconcept_record", "merge",
    "parse_reply", "relation", "render_input", "require_measurement_safe",
    "symbol", "verified_against", "verify", "verify_name",
]
