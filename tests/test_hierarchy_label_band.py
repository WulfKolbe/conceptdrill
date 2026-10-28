"""The label band is asked for and checked from ONE source.

## The contract

The band a label is CHECKED against must equal the band the prompt ASKED for,
at every ceiling, with no literal band anywhere outside `bands_for`:

    for ceiling in (35, 50, 70):
        effective_label_band(ceiling) == bands_for(ceiling)["label"]

`bands_for` is the single source. `TIER_WORDS` and `LABEL_WORDS` are its value
at the default ceiling and are convenience aliases, not second sources.

## Why it needs a test

The two numbers drifted apart in the tree and nothing noticed: the prompt asked
for `bands_for(50)["label"]` while `LABEL_WORDS` and `tools/gate5_prompt.py`
both checked a hard-coded 30-42. A label of 26 words satisfied the prompt it
was generated from and failed the gate measuring it, so Gate 5's pass rate was
a measurement of the gap between two constants rather than of the prompt.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.labelcheck import (LABEL_WORDS, check_label,
                                               check_tiers,
                                               effective_label_band)
from conceptdrill.hierarchy.summarize import (EMBEDDING_TOKEN_WINDOW,
                                              TIER_WORDS, bands_for)

CEILINGS = (35, 50, 70)


@pytest.mark.parametrize("ceiling", CEILINGS)
def test_the_checked_band_is_the_asked_band(ceiling):
    """The postcondition itself."""
    assert effective_label_band(ceiling) == bands_for(ceiling)["label"]


def test_the_default_check_band_is_the_default_prompt_band():
    """`LABEL_WORDS` is the alias, so it must equal what the default ceiling
    asks for. This is the assertion the drift broke."""
    assert LABEL_WORDS == bands_for(EMBEDDING_TOKEN_WINDOW[1])["label"]


def test_the_alias_agrees_with_the_tier_table():
    assert LABEL_WORDS == TIER_WORDS["label"]


def test_omitting_the_ceiling_gives_the_default_band():
    assert effective_label_band() == LABEL_WORDS
    assert effective_label_band(None) == LABEL_WORDS


@pytest.mark.parametrize("ceiling", CEILINGS)
def test_a_label_at_either_edge_of_its_band_passes(ceiling):
    """Behavioural, not just arithmetic: a label exactly at the bound the
    prompt asked for must not be reported as out of band."""
    lo, hi = effective_label_band(ceiling)
    for n in (lo, hi):
        label = " ".join(["concept"] * n)
        assert [p for p in check_label(label, words=effective_label_band(ceiling))
                if "words" in p or "below" in p or "above" in p] == []


@pytest.mark.parametrize("ceiling", CEILINGS)
def test_a_label_outside_its_band_is_reported(ceiling):
    lo, hi = effective_label_band(ceiling)
    band = effective_label_band(ceiling)
    assert any("below" in p
               for p in check_label(" ".join(["concept"] * (lo - 1)), words=band))
    assert any("above" in p
               for p in check_label(" ".join(["concept"] * (hi + 1)), words=band))


def test_a_higher_ceiling_gives_a_wider_band():
    """Guards the direction: a bigger token budget must not shrink the band."""
    assert effective_label_band(70)[1] > effective_label_band(50)[1]
    assert effective_label_band(50)[1] > effective_label_band(35)[1]


@pytest.mark.parametrize("ceiling", CEILINGS)
def test_check_tiers_checks_the_label_against_its_own_ceiling(ceiling):
    """`check_tiers` must not fall back to the default band: a label exactly
    at the edge of the band its ceiling asked for passes, one word past fails."""
    lo, hi = effective_label_band(ceiling)
    edge = check_tiers(" ".join(["concept"] * hi), "x", "y", ceiling=ceiling)
    assert edge["label"] == []
    over = check_tiers(" ".join(["concept"] * (hi + 1)), "x", "y",
                       ceiling=ceiling)
    assert any(f"above {hi}" in p for p in over["label"])


def test_check_tiers_without_a_ceiling_uses_the_default_band():
    hi = LABEL_WORDS[1]
    assert check_tiers(" ".join(["concept"] * hi), "x", "y")["label"] == []


def test_no_band_literal_survives_in_the_gate_or_the_checker():
    """The literals this work removed must not come back.

    Reads the sources rather than the behaviour: a reintroduced `30 <= n <= 42`
    would pass every test above while restoring exactly the bug.
    """
    import re
    from pathlib import Path

    import conceptdrill.hierarchy.labelcheck as lc
    root = Path(lc.__file__).resolve().parents[3]
    for relative in ("src/conceptdrill/hierarchy/labelcheck.py",
                     "tools/gate5_prompt.py"):
        source = (root / relative).read_text(encoding="utf-8")
        code = "\n".join(line for line in source.splitlines()
                         if not line.lstrip().startswith("#"))
        assert not re.search(r"\b30\s*<=|\b<=\s*42\b|\(\s*30\s*,\s*42\s*\)",
                             code), f"{relative} holds a literal label band"


def test_the_gate_passes_a_band_to_the_check_rather_than_taking_the_default():
    """The gate knows its ceiling, so it must hand the band to `check_label`.

    Relying on the default would make the gate correct only at the default
    ceiling, which is exactly how the band drifted the first time. Read from
    the source because the alternative is running the gate, which costs an LLM.
    """
    from pathlib import Path

    import conceptdrill.hierarchy.labelcheck as lc
    root = Path(lc.__file__).resolve().parents[3]
    source = (root / "tools" / "gate5_prompt.py").read_text(encoding="utf-8")
    assert "check_label(summary.label, words=" in source, (
        "gate5_prompt.py calls check_label without a band")
    assert "effective_label_band(" in source, (
        "gate5_prompt.py does not derive its band from the single source")
