"""The ConceptDrill facade, determinism, storage, the agent API, and the CLI."""
from __future__ import annotations

import json

import numpy as np
import pytest

from conceptdrill import ConceptDrill
from conceptdrill.abstractor import CallableAbstractor, NullAbstractor
from conceptdrill.api import (request_concept_generation, request_embedding,
                              request_projection, request_storage,
                              request_verification)
from conceptdrill.cli import main
from conceptdrill.routing import DORMANT_TYPES, model_for_type, routing_table
from conceptdrill.storage import (content_hash, read_sidecar, sidecar_path,
                                  verify_sidecar)


@pytest.fixture
def drill(mock_document, embedder):
    return ConceptDrill(mock_document, embedder=embedder, max_concepts=25)


# --------------------------------------------------------------------------
# The class interface from the spec
# --------------------------------------------------------------------------

def test_builds_a_concept_space(drill):
    assert len(drill) > 0
    assert drill.space.matrix.shape[0] == len(drill.space)


def test_project_text_returns_one_score_per_concept(drill):
    vector = drill.project_text("semantic projection of a paragraph")
    assert vector.shape == (len(drill.space),)
    assert np.all(np.abs(vector) <= 1.0 + 1e-5)


def test_explain_text_returns_ranked_pairs(drill):
    hits = drill.explain_text("concept scoring and coverage", top_k=3)
    assert len(hits) == 3
    assert all(isinstance(name, str) for name, _ in hits)
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_explain_text_finds_the_relevant_concept(drill):
    """The end-to-end sanity check: text about a document topic must surface a
    concept from that topic."""
    names = [n.lower() for n, _ in drill.explain_text(
        "the concept space is built by scoring candidates", top_k=8)]
    assert any("concept" in n for n in names)


def test_get_concept_space_info_is_complete(drill):
    info = drill.get_concept_space_info()
    for key in ("size", "dimension", "levels", "sources", "embedding_model",
                "embedding_revision", "similarity_metric", "document",
                "build", "scorer"):
        assert key in info
    assert info["build"]["n_candidates"] >= info["size"]
    assert info["scorer"]["weights"]


def test_batch_projection_matches_single(drill):
    texts = ["semantic projection", "concept scoring"]
    batched = drill.project_spans(texts, top_k=3)
    for text, expected in zip(texts, batched):
        assert drill.explain_text(text, top_k=3) == expected


def test_refine_expands_in_place(mock_document, embedder):
    # Deliberately under-sized so there is an unselected pool left to draw from.
    small = ConceptDrill(mock_document, embedder=embedder, max_concepts=8)
    before = len(small)
    small.refine("concept scoring", desired_size=before + 3)
    assert len(small) == before + 3


def test_refine_stops_when_the_pool_is_exhausted(drill):
    """Asking for more concepts than exist must be a no-op, not an error."""
    before = len(drill)
    drill.refine("concept scoring", desired_size=before + 100)
    assert len(drill) == before


# --------------------------------------------------------------------------
# Determinism — the headline requirement
# --------------------------------------------------------------------------

def test_two_builds_agree(mock_document, embedder):
    from conceptdrill.embeddings import get_embedder
    a = ConceptDrill(mock_document, embedder=get_embedder("hash", cache=False, dim=128))
    b = ConceptDrill(mock_document, embedder=get_embedder("hash", cache=False, dim=128))
    assert [c.id for c in a.space.concepts] == [c.id for c in b.space.concepts]
    assert [c.score for c in a.space.concepts] == [c.score for c in b.space.concepts]


def test_content_hash_is_reproducible(mock_document_path, tmp_path):
    out = tmp_path / "one.json"
    other = tmp_path / "two.json"
    for target in (out, other):
        rc = main(["project", str(mock_document_path), "--model", "hash",
                   "--output", str(target)])
        assert rc == 0
    assert (read_sidecar(out)["content_hash"]
            == read_sidecar(other)["content_hash"])


def test_content_hash_ignores_the_timestamp():
    base = {"a": 1, "created_at": "2020-01-01"}
    later = {"a": 1, "created_at": "2026-07-28"}
    assert content_hash(base) == content_hash(later)


def test_content_hash_notices_real_changes():
    assert content_hash({"a": 1}) != content_hash({"a": 2})


def test_verify_detects_tampering(mock_document_path, tmp_path):
    out = tmp_path / "sidecar.json"
    main(["project", str(mock_document_path), "--model", "hash",
          "--output", str(out)])
    assert verify_sidecar(out)[0]

    payload = read_sidecar(out)
    payload["projections"][0]["object_id"] = "tampered"
    out.write_text(json.dumps(payload), encoding="utf-8")
    assert not verify_sidecar(out)[0]


# --------------------------------------------------------------------------
# The source document is never touched
# --------------------------------------------------------------------------

def test_input_file_is_left_byte_identical(mock_document_path, tmp_path):
    before = mock_document_path.read_bytes()
    main(["project", str(mock_document_path), "--model", "hash",
          "--output", str(tmp_path / "out.json")])
    assert mock_document_path.read_bytes() == before


def test_sidecar_path_does_not_double_the_extension(tmp_path):
    p = sidecar_path(tmp_path / "model.docmodel.json")
    assert p.name == "model.docmodel.conceptdrill.json"


# --------------------------------------------------------------------------
# Abstractor
# --------------------------------------------------------------------------

def test_null_abstractor_is_marked_deterministic(drill):
    assert drill.get_concept_space_info()["build"]["abstractor_deterministic"]


def test_callable_abstractor_is_marked_non_deterministic(mock_document, embedder):
    drill = ConceptDrill(mock_document, embedder=embedder,
                         abstractor=CallableAbstractor(lambda p: "a concept"))
    assert not drill.get_concept_space_info()["build"]["abstractor_deterministic"]


def test_callable_abstractor_falls_back_when_the_model_fails():
    def broken(prompt):
        raise RuntimeError("model unavailable")

    abstractor = CallableAbstractor(broken)
    # Falls through to the deterministic structural description.
    assert "summation" in abstractor.describe_equation(r"\sum_i x_i")


def test_title_shortening_prefers_a_clause_boundary():
    """Scientific titles put the contribution before the colon and the
    qualification after it, so the head is the better concept name."""
    out = NullAbstractor().shorten_title(
        "Conceptualizing Embedding Spaces: A Framework for Concept Extraction",
        max_words=5)
    assert out == "Conceptualizing Embedding Spaces"


def test_title_within_the_limit_is_left_alone():
    title = "Deep Residual Learning for Image Recognition"
    assert NullAbstractor().shorten_title(title, max_words=8) == title


def test_title_without_a_clause_boundary_is_truncated():
    out = NullAbstractor().shorten_title(
        "one two three four five six seven eight nine ten", max_words=4)
    assert out == "one two three four"


# --------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------

def test_routing_follows_the_spec():
    assert model_for_type("paragraph") == "sentencebert"
    assert model_for_type("heading") == "sentencebert"
    assert model_for_type("equation") == "mathbert"
    assert model_for_type("formula") == "mathbert"
    assert model_for_type("bibliography") == "sentencebert"


def test_code_routes_to_codebert_but_is_marked_dormant():
    assert model_for_type("code") == "codebert"
    assert model_for_type("algorithm") == "codebert"
    assert "code" in DORMANT_TYPES
    assert routing_table()["code"]["dormant"] is True


def test_explicit_override_wins():
    assert model_for_type("equation", override="sentencebert") == "sentencebert"


def test_unknown_type_falls_back_to_the_default():
    assert model_for_type("brand-new-type") == "sentencebert"


# --------------------------------------------------------------------------
# Agent API
# --------------------------------------------------------------------------

def test_request_concept_generation(mock_document_path):
    result = request_concept_generation(mock_document_path, model="hash")
    assert result["status"] == "completed"
    assert result["concepts"]


def test_request_embedding(tmp_path):
    result = request_embedding(["alpha", "beta"], model="hash",
                               cache_dir=str(tmp_path))
    assert result["status"] == "completed"
    assert result["count"] == 2
    assert result["dimension"] > 0


def test_request_projection_for_a_span(mock_document_path):
    result = request_projection(mock_document_path, model="hash",
                                text="semantic projection", top_k=3)
    assert result["status"] == "completed"
    assert len(result["concepts"]) == 3


def test_request_storage_reports_completed_then_updated(mock_document_path, tmp_path):
    out = tmp_path / "sidecar.json"
    first = request_storage(mock_document_path, output=out, model="hash")
    assert first["status"] == "completed"

    second = request_storage(mock_document_path, output=out, model="hash")
    assert second["status"] == "updated"
    assert second["changed"] is False       # same input, same hash


def test_request_verification(mock_document_path, tmp_path):
    out = tmp_path / "sidecar.json"
    request_storage(mock_document_path, output=out, model="hash")
    assert request_verification(out)["matches"] is True


def test_failures_are_returned_not_raised():
    result = request_projection("/nonexistent/path.json", model="hash")
    assert result["status"] == "failed"
    assert result["error_type"]
    assert "error" in result


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def test_cli_project_writes_a_sidecar(mock_document_path, tmp_path, capsys):
    out = tmp_path / "out.json"
    assert main(["project", str(mock_document_path), "--model", "hash",
                 "--output", str(out)]) == 0
    payload = read_sidecar(out)
    assert payload["format"] == "conceptdrill"
    assert payload["projections"]
    assert payload["concept_spaces"]["hash"]["concepts"]


def test_cli_project_dry_run_writes_nothing(mock_document_path, tmp_path, capsys):
    out = tmp_path / "out.json"
    assert main(["project", str(mock_document_path), "--model", "hash",
                 "--output", str(out), "--dry-run"]) == 0
    assert not out.exists()
    assert "would_write" in capsys.readouterr().out


def test_cli_multiple_models_produce_independent_projections(
        mock_document_path, tmp_path):
    """Repeating --model is what makes multi-model agreement a read over stored
    output rather than new machinery."""
    out = tmp_path / "out.json"
    assert main(["project", str(mock_document_path), "--output", str(out),
                 "--model", "hash", "--model", "hash"]) == 0
    # Deduplicated: the same model twice is one projection set.
    payload = read_sidecar(out)
    models = {p["embedding_model"] for p in payload["projections"]}
    assert models == {"hash"}
    assert set(payload["concept_spaces"]) == {"hash"}


def test_every_projection_resolves_against_its_own_model_space(mock_document,
                                                              embedder):
    """Each model builds its own vocabulary, so a sidecar holding two models'
    projections must store both spaces — otherwise concept ids dangle."""
    from conceptdrill.embeddings import get_embedder
    from conceptdrill.storage import build_payload, resolve_concept

    # Two genuinely different spaces: different dims mine different vocabularies.
    a = ConceptDrill(mock_document, embedder=embedder, max_concepts=8)
    b = ConceptDrill(mock_document,
                     embedder=get_embedder("hash", cache=False, dim=64),
                     max_concepts=20)
    b.embedder.name = "other"          # distinct key in the payload

    pa, _ = a.project_document(top_k=5)
    pb, _ = b.project_document(top_k=5)
    assert {c.id for c in a.space.concepts} != {c.id for c in b.space.concepts}

    payload = build_payload(
        source_path="x.json",
        spaces={"hash": a.space, "other": b.space},
        projections=[*pa, *pb], store_embeddings=False)

    assert set(payload["concept_spaces"]) == {"hash", "other"}
    for projection in payload["projections"]:
        space = payload["concept_spaces"][projection["embedding_model"]]
        ids = {c["id"] for c in space["concepts"]}
        for hit in projection["concepts"]:
            assert hit["concept_id"] in ids, "concept id must resolve"
            assert resolve_concept(payload, projection, hit["concept_id"])


def test_cli_explain_flat_form(mock_document_path, capsys):
    """The spec's invocation: conceptdrill --input X --text Y --top 5"""
    rc = main(["--input", str(mock_document_path), "--text",
               "Deep learning for graphs", "--top", "5", "--model", "hash"])
    assert rc == 0
    assert len(capsys.readouterr().out.strip().splitlines()) == 6   # header + 5


def test_cli_explain_subcommand_json(mock_document_path, capsys):
    rc = main(["explain", str(mock_document_path), "--text", "concept scoring",
               "--top", "3", "--model", "hash", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload["concepts"]) == 3
    assert payload["concepts"][0]["rank"] == 1


def test_cli_concepts_listing(mock_document_path, capsys):
    rc = main(["concepts", str(mock_document_path), "--model", "hash",
               "--top", "5", "--metrics"])
    assert rc == 0
    assert "concepts from" in capsys.readouterr().out


def test_cli_concepts_json(mock_document_path, capsys):
    rc = main(["concepts", str(mock_document_path), "--model", "hash", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["concepts"][0]["metrics"]


def test_cli_verify(mock_document_path, tmp_path, capsys):
    out = tmp_path / "out.json"
    main(["project", str(mock_document_path), "--model", "hash",
          "--output", str(out)])
    assert main(["verify", str(out)]) == 0
    assert "OK" in capsys.readouterr().out


def test_cli_routing_marks_dormant_types(capsys):
    assert main(["routing"]) == 0
    out = capsys.readouterr().out
    assert "codebert" in out
    assert "dormant" in out


def test_cli_missing_file_returns_two(capsys):
    assert main(["project", "/no/such/file.json"]) == 2


def test_cli_weight_override(mock_document_path, capsys):
    rc = main(["concepts", str(mock_document_path), "--model", "hash",
               "--weight", "structural=0.9", "--json"])
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["concepts"]


def test_cli_bad_weight_is_rejected(mock_document_path):
    with pytest.raises(SystemExit):
        main(["concepts", str(mock_document_path), "--weight", "nonsense=1"])


def test_cli_no_arguments_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


# --------------------------------------------------------------------------
# End to end on a real DocModel
# --------------------------------------------------------------------------

def test_docmodel_end_to_end(docmodel_path, tmp_path):
    out = tmp_path / "dm.conceptdrill.json"
    assert main(["project", str(docmodel_path), "--model", "hash",
                 "--output", str(out)]) == 0
    payload = read_sidecar(out)
    assert payload["projections"]
    # Pages, the Document root, and Citations must appear as skips.
    skipped_types = {s["object_type"] for s in payload["skipped"]}
    assert "Page" in skipped_types
    assert "Citation" in skipped_types


def test_docmodel_projections_reference_real_object_ids(docmodel_path, tmp_path):
    out = tmp_path / "dm.json"
    main(["project", str(docmodel_path), "--model", "hash", "--output", str(out)])
    ids = {p["object_id"] for p in read_sidecar(out)["projections"]}
    assert {"p1", "p2"} <= ids
