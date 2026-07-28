"""Shared fixtures.

Two things are pinned so the suite is deterministic and offline:

  * the `hash` embedder — no model download, no network;
  * the `regex` NLP tier — stanza and spaCy mine *different* noun phrases, so a
    suite that used whichever happened to be installed would assert different
    vocabularies on different machines. Pinning it also keeps the run fast:
    stanza costs seconds of model overhead per batch.

`test_nlp.py` covers the neural tiers explicitly, skipping when unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Must be set before conceptdrill.nlp resolves and caches the backend.
os.environ.setdefault("CONCEPTDRILL_NLP_BACKEND", "regex")

from conceptdrill.document import Document          # noqa: E402
from conceptdrill.embeddings import get_embedder    # noqa: E402


@pytest.fixture(autouse=True)
def _pinned_nlp_backend(monkeypatch):
    """Guard against a test leaving the backend override changed."""
    monkeypatch.setenv("CONCEPTDRILL_NLP_BACKEND", "regex")


@pytest.fixture
def embedder():
    """Offline deterministic embedder, cache disabled so tests never share state."""
    return get_embedder("hash", cache=False, dim=128)


@pytest.fixture
def mock_document_json():
    """A small paper with every structure the generators mine.

    Deliberately repetitive: noun-phrase extraction needs at least three
    occurrences of a phrase before it becomes a candidate.
    """
    return {
        "meta": {"title": "Semantic Projection for Document Models",
                 "bibkey": "mock2026"},
        "sections": [
            {"id": "s1", "title": "1 Introduction", "level": 1},
            {"id": "s2", "title": "2 Method", "level": 1},
            {"id": "s2a", "title": "2.1 Semantic Projection", "level": 2,
             "parent": "s2"},
            {"id": "s2b", "title": "2.2 Concept Scoring", "level": 2,
             "parent": "s2"},
            {"id": "s3", "title": "3 Evaluation", "level": 1},
            {"id": "s4", "title": "Glossary", "level": 1},
        ],
        "blocks": [
            {"id": "b1", "type": "paragraph", "section": "s1",
             "text": "Semantic projection maps a document object into a concept "
                     "space. We present a semantic projection method that builds "
                     "the concept space from the document itself. Concept space "
                     "construction has previously relied on an external ontology."},
            {"id": "b2", "type": "paragraph", "section": "s1",
             "text": "A Convolutional Neural Network (CNN) is often used as the "
                     "encoder. The CNN produces a latent vector for each input. "
                     "Our semantic projection approach is encoder agnostic."},
            {"id": "b3", "type": "paragraph", "section": "s2a",
             "text": "The semantic projection operator computes cosine similarity "
                     "between a latent vector and every concept vector. Concept "
                     "vectors are precomputed once. The concept space is a matrix."},
            {"id": "b4", "type": "definition", "section": "s2a",
             "name": "Concept Projection",
             "text": "Definition 1 (Concept Projection). The concept projection "
                     "of a text span is the vector of similarities between the "
                     "span and every concept in the concept space."},
            {"id": "b5", "type": "paragraph", "section": "s2b",
             "text": "Concept scoring combines structural importance, coverage, "
                     "and purity. Concept scoring rejects a concept that matches "
                     "every paragraph. Concept scoring is a weighted sum."},
            {"id": "b6", "type": "equation", "section": "s2b",
             "text": r"\mathcal{L}_{c} = \sum_{i=1}^{K} y_i \log p_i"},
            {"id": "b7", "type": "equation", "section": "s2b",
             "text": r"\hat{\theta} = \arg\min_{\theta} \frac{1}{N}\sum_i "
                     r"\|f(x_i;\theta) - y_i\|^2"},
            {"id": "b8", "type": "paragraph", "section": "s3",
             "text": "We evaluate the concept space on 120,000 query documents. "
                     "The evaluation shows that concept scoring improves "
                     "precision. Evaluation used a held out split."},
            {"id": "b9", "type": "table", "section": "s3",
             "text": "Precision at k for the semantic projection baseline",
             "caption": "Precision at k"},
            {"id": "b10", "type": "paragraph", "section": "s4",
             "text": "Concept space — the set of concepts used for projection.\n"
                     "Latent vector — the embedding produced by the encoder.\n"
                     "Structural weight — the prior assigned by provenance."},
        ],
        "bibliography": [
            {"id": "r1", "title": "Attention Is All You Need",
             "year": 2017, "citations": 100000, "label": "vaswani2017"},
            {"id": "r2", "title": "Deep Residual Learning for Image Recognition",
             "year": 2016, "citations": 200000, "label": "he2016"},
            {"id": "r3",
             "title": "Conceptualizing Embedding Spaces for Large Language Model "
                      "Interpretability: A Framework for Concept Extraction",
             "year": 2022, "citations": 40, "label": "ces2022"},
            {"id": "r4", "title": "Support Vector Networks",
             "year": 1995, "citations": 60000, "label": "cortes1995"},
        ],
    }


@pytest.fixture
def mock_document(mock_document_json):
    return Document.from_generic(mock_document_json, source_path="mock.json")


@pytest.fixture
def mock_document_path(tmp_path, mock_document_json):
    path = tmp_path / "paper.json"
    path.write_text(json.dumps(mock_document_json), encoding="utf-8")
    return path


@pytest.fixture
def docmodel_json():
    """A minimal Semantic Compiler DocModel, matching the real shape."""
    return {
        "meta": {"bibkey": "dm2026", "source": "tesseract"},
        "streams": {},
        "alignments": {},
        "objects": [
            {"id": "dm2026", "type": "Document",
             "props": {"bibkey": "dm2026", "total_pages": 2},
             "realizations": [], "children": ["sec1"], "parent": None},
            {"id": "pg1", "type": "Page",
             "props": {"page_number": 1, "is_blank": False},
             "realizations": [], "children": [], "parent": None},
            {"id": "sec1", "type": "Section",
             "props": {"title": "Deep Hashing", "level": 1},
             "realizations": [], "children": [], "parent": None},
            {"id": "p1", "type": "Paragraph",
             "props": {"text": "We present a deep hashing model for image "
                               "similarity search. The deep hashing model learns "
                               "binary hash codes. Deep hashing is efficient.",
                       "page": 1},
             "realizations": [], "children": [], "parent": "sec1"},
            {"id": "p2", "type": "Paragraph",
             "props": {"text": "Binary hash codes are compared with the Hamming "
                               "distance. The Hamming distance is cheap to "
                               "compute. Hash codes are short."},
             "realizations": [], "children": [], "parent": "sec1"},
            {"id": "eq1", "type": "Equation",
             "props": {"latex": r"d_H(a,b) = \sum_i a_i \oplus b_i",
                       "numbered": True},
             "realizations": [], "children": [], "parent": None},
            {"id": "tbl1", "type": "Table",
             "props": {"caption": "Search latencies for different thresholds",
                       "latex_code": "\\begin{tabular}{rr}\\hline k & ms \\\\\\hline"
                                     " 10 & 87.9 \\\\\\hline\\end{tabular}"},
             "realizations": [], "children": [], "parent": None},
            {"id": "cit1", "type": "Citation",
             "props": {"citekey": "T514", "page": 2},
             "realizations": [], "children": [], "parent": None},
            {"id": "li1", "type": "ListItem",
             "props": {"marker": "1.",
                       "content": "Cao, Z., Long, M., Wang, J.: Hashnet: Deep "
                                  "learning to hash by continuation. In: Proc. "
                                  "ICCV (2017)"},
             "realizations": [], "children": [], "parent": None},
        ],
    }


@pytest.fixture
def docmodel_path(tmp_path, docmodel_json):
    path = tmp_path / "model.docmodel.json"
    path.write_text(json.dumps(docmodel_json), encoding="utf-8")
    return path
