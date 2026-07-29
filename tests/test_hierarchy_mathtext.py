"""Unit tests for formula-to-text.

The LaTeX cases are taken from the drilled CES paper, not invented.
"""
from __future__ import annotations

import pytest

from conceptdrill.hierarchy.mathtext import (LATEX_PROP_CANDIDATES,
                                             SPOKEN_PROP_CANDIDATES,
                                             fallback_speak, latex_from_props,
                                             math_text, protect_identifiers,
                                             spoken_from_props)

REAL_OBJ = r"Obj(c_1) \subseteq Obj(c_2)"
REAL_AVG = r"AVERAGE_{s \in siblings(c,p)}\frac{|parents(c)|}{|parents(p)|}"
REAL_LOSS = r"\mathcal{L}_{c} = \sum_{i=1}^{K} y_i \log p_i"
REAL_SET = r"C^i=\{c \in C | d(c) = i\}"


class FakeSpeaker:
    def __init__(self, reply="spoken form"):
        self.reply, self.seen = reply, []

    def speak_math(self, latex):
        self.seen.append(latex)
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


# --------------------------------------------------------------------------
# protect_identifiers
# --------------------------------------------------------------------------

def test_multi_letter_identifier_is_wrapped():
    """Unwrapped, SRE reads Obj as 'O b j' -- three meaningless tokens."""
    assert r"\text{Obj}" in protect_identifiers(REAL_OBJ)


def test_several_identifiers_are_wrapped():
    out = protect_identifiers(REAL_AVG)
    assert r"\text{AVERAGE}" in out and r"\text{siblings}" in out


def test_single_letters_are_left_alone():
    """x and y are variables, not words."""
    assert protect_identifiers("x + y = z") == "x + y = z"


def test_two_letter_runs_are_left_alone():
    """'ab' is far more often two variables than one identifier."""
    assert protect_identifiers("ab = c") == "ab = c"


def test_latex_commands_are_not_wrapped():
    out = protect_identifiers(r"\alpha + \beta")
    assert r"\text{alpha}" not in out


def test_already_wrapped_text_is_untouched():
    """Applying this twice must be harmless."""
    src = r"\text{Obj}(c)"
    assert protect_identifiers(src) == src


def test_protection_is_idempotent():
    once = protect_identifiers(REAL_OBJ)
    assert protect_identifiers(once) == once


def test_empty_input():
    assert protect_identifiers("") == ""


# --------------------------------------------------------------------------
# fallback_speak — offline, deterministic
# --------------------------------------------------------------------------

def test_fallback_names_operators():
    out = fallback_speak(REAL_LOSS)
    assert "the sum of" in out and "equals" in out and "log" in out


def test_fallback_handles_set_notation():
    assert "in" in fallback_speak(REAL_SET)


def test_fallback_expands_subset():
    assert "subset of or equal to" in fallback_speak(REAL_OBJ)


def test_fallback_spells_greek():
    assert "tau" in fallback_speak(r"\tau + \alpha")


def test_fallback_leaves_no_latex_syntax():
    for tex in (REAL_OBJ, REAL_AVG, REAL_LOSS, REAL_SET):
        out = fallback_speak(tex)
        assert "\\" not in out, tex
        assert "{" not in out and "}" not in out, tex
        assert "$" not in out, tex


def test_fallback_is_deterministic():
    assert fallback_speak(REAL_LOSS) == fallback_speak(REAL_LOSS)


def test_fallback_of_empty_is_empty():
    assert fallback_speak("") == "" and fallback_speak("   ") == ""


def test_fallback_keeps_identifiers():
    """Identifiers are the concept signal; only the syntax should go."""
    assert "Obj" in fallback_speak(REAL_OBJ)


# --------------------------------------------------------------------------
# Prop resolution
# --------------------------------------------------------------------------

def test_spoken_prop_is_preferred():
    """Once the docmodel carries spoken math, it wins: the compiler has more
    context than this package does."""
    text, source = math_text({"spoken": "L equals the sum", "latex": REAL_LOSS})
    assert text == "L equals the sum" and source == "docmodel"


def test_every_candidate_prop_is_recognised():
    for key in SPOKEN_PROP_CANDIDATES:
        assert spoken_from_props({key: "said"}) == "said"


def test_latex_candidates_are_tried_in_order():
    assert latex_from_props({"latex_raw": "b"}) == "b"
    assert latex_from_props({"latex": "a", "latex_raw": "b"}) == "a"


def test_blank_spoken_prop_falls_through():
    text, source = math_text({"spoken": "   ", "latex": REAL_LOSS})
    assert source == "fallback" and text


# --------------------------------------------------------------------------
# math_text resolution order
# --------------------------------------------------------------------------

def test_speaker_is_used_when_no_spoken_prop():
    speaker = FakeSpeaker("the spoken form")
    text, source = math_text({"latex": REAL_LOSS}, speaker=speaker)
    assert text == "the spoken form" and source == "speech"


def test_speaker_receives_protected_latex():
    speaker = FakeSpeaker()
    math_text({"latex": REAL_OBJ}, speaker=speaker)
    assert r"\text{Obj}" in speaker.seen[0]


def test_protection_can_be_disabled():
    speaker = FakeSpeaker()
    math_text({"latex": REAL_OBJ}, speaker=speaker, protect=False)
    assert r"\text{" not in speaker.seen[0]


def test_speaker_failure_degrades_to_the_fallback():
    """A backend failure must cost quality, not the run."""
    text, source = math_text({"latex": REAL_LOSS},
                             speaker=FakeSpeaker(RuntimeError("node died")))
    assert source == "fallback" and text


def test_empty_speaker_reply_degrades_to_the_fallback():
    text, source = math_text({"latex": REAL_LOSS}, speaker=FakeSpeaker(""))
    assert source == "fallback" and text


def test_no_speaker_uses_the_fallback():
    text, source = math_text({"latex": REAL_LOSS})
    assert source == "fallback" and "the sum of" in text


def test_trivial_formula_is_dropped():
    """This paper is full of single-symbol Formula objects like T and k. As
    prose they contribute a stray letter and no concept signal."""
    assert math_text({"latex": "T"}) == ("", "none")
    assert math_text({"latex": "k"}) == ("", "none")


def test_threshold_is_configurable():
    text, source = math_text({"latex": "T"}, min_latex_chars=1)
    assert source == "fallback" and text == "T"


def test_object_with_no_math_at_all():
    assert math_text({}) == ("", "none")


def test_source_is_always_one_of_the_documented_values():
    for props in ({"spoken": "s"}, {"latex": REAL_LOSS}, {}, {"latex": "T"}):
        assert math_text(props)[1] in {"docmodel", "speech", "fallback", "none"}
