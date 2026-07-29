"""Unit tests for model-reply JSON recovery.

Each defect class is tested on its own, then in combination, because a repair
that fixes one defect while breaking another is worse than no repair.
"""
from __future__ import annotations

from conceptdrill.hierarchy.replyparse import (control_corruption,
                                               iter_json_objects, parse_reply,
                                               repair_escapes, repair_keys,
                                               strip_fences)

GOOD = '{"summary": "a summary", "abstraction": "an idea", "label": "a label"}'


# --------------------------------------------------------------------------
# Clean input must pass through untouched
# --------------------------------------------------------------------------

def test_clean_json_parses():
    assert parse_reply(GOOD)["label"] == "a label"


def test_clean_json_is_not_mangled_by_repairs():
    """A legal \\n escape must stay a newline, not become a literal backslash-n."""
    assert parse_reply('{"summary": "line one\\nline two"}')["summary"] == \
        "line one\nline two"


def test_unicode_survives():
    assert parse_reply('{"label": "the \\u03c4 function"}')["label"] == \
        "the τ function"


# --------------------------------------------------------------------------
# Defect 1: markdown fences
# --------------------------------------------------------------------------

def test_json_fence_is_stripped():
    assert parse_reply(f"```json\n{GOOD}\n```")["label"] == "a label"


def test_bare_fence_is_stripped():
    assert parse_reply(f"```\n{GOOD}\n```")["summary"] == "a summary"


def test_strip_fences_leaves_unfenced_text_alone():
    assert strip_fences(GOOD) == GOOD


# --------------------------------------------------------------------------
# Defect 2: raw LaTeX backslashes (the common one on scientific text)
# --------------------------------------------------------------------------

def test_raw_latex_backslash_is_recovered():
    """`\\sum` is not a legal JSON escape and kills a strict parse."""
    got = parse_reply('{"summary": "the loss \\sum_i y_i log p_i"}')
    assert got is not None
    assert "sum" in got["summary"]


def test_several_latex_commands_recover():
    got = parse_reply(r'{"label": "\mathcal{L} and \alpha and \sum"}')
    assert got is not None
    assert "mathcal" in got["label"] and "alpha" in got["label"]


# --------------------------------------------------------------------------
# The defect that CANNOT be repaired: LaTeX eaten by a legal JSON escape
# --------------------------------------------------------------------------

def test_tau_is_silently_eaten_by_the_tab_escape():
    """`\t` is a LEGAL JSON escape, so `\tau` parses to TAB + 'au'. Nothing
    fails and nothing can repair it -- the reply was valid JSON all along."""
    got = parse_reply(r'{"label": "the \tau function"}')
    assert got is not None
    assert "tau" not in got["label"]
    assert "\t" in got["label"]


def test_control_corruption_detects_the_eaten_command():
    got = parse_reply(r'{"label": "the \tau function"}')
    assert control_corruption(got["label"])


def test_control_corruption_is_quiet_on_clean_prose():
    assert control_corruption("A perfectly ordinary summary sentence.") == ()


def test_control_corruption_ignores_newlines():
    """Multi-line summaries are legitimate; flagging \n would fire always."""
    assert control_corruption("line one\nline two") == ()


def test_control_corruption_flags_every_eaten_escape():
    """\\beta -> backspace+'eta', \\frac -> formfeed+'rac', \\rho -> CR+'ho'."""
    assert control_corruption("the \beta parameter") == ("\\b...",)
    assert control_corruption("the \frac term") == ("\\f...",)
    assert control_corruption("the \rho value") == ("\\r...",)


def test_control_corruption_reports_all_offenders():
    assert set(control_corruption("\b and \f")) == {"\\b...", "\\f..."}


def test_control_corruption_handles_empty_input():
    assert control_corruption("") == () and control_corruption(None) == ()


def test_repair_escapes_leaves_legal_escapes_alone():
    assert repair_escapes(r'"a\nb"') == r'"a\nb"'
    assert repair_escapes(r'"aéb"') == r'"aéb"'


def test_repair_escapes_doubles_an_illegal_one():
    assert repair_escapes(r'"\sum"') == r'"\\sum"'


# --------------------------------------------------------------------------
# Defect 3: unquoted keys
# --------------------------------------------------------------------------

def test_unquoted_keys_are_recovered():
    got = parse_reply('{summary: "s", abstraction: "a", label: "l"}')
    assert got == {"summary": "s", "abstraction": "a", "label": "l"}


def test_repair_keys_does_not_touch_quoted_keys():
    assert repair_keys(GOOD) == GOOD


def test_repair_keys_does_not_touch_a_colon_inside_a_value():
    """'Concept Spaces: A Framework' must not gain quotes mid-sentence."""
    src = '{"label": "Embedding Spaces: A Framework"}'
    assert parse_reply(repair_keys(src))["label"] == "Embedding Spaces: A Framework"


# --------------------------------------------------------------------------
# Defect 4: surrounding prose and extra objects
# --------------------------------------------------------------------------

def test_trailing_prose_is_ignored():
    assert parse_reply(f"Here is the result:\n{GOOD}\nHope that helps!")["label"] \
        == "a label"


def test_first_object_wins_when_several_are_present():
    assert parse_reply(f'{GOOD}\n{{"label": "second"}}')["label"] == "a label"


def test_a_list_reply_yields_its_first_object():
    assert parse_reply(f"[{GOOD}]")["label"] == "a label"


def test_brace_inside_a_string_does_not_confuse_the_scanner():
    got = parse_reply('prose {"label": "uses {braces} inside"} more prose')
    assert got["label"] == "uses {braces} inside"


def test_iter_json_objects_finds_each_block():
    blocks = list(iter_json_objects('{"a": 1} and {"b": 2}'))
    assert blocks == ['{"a": 1}', '{"b": 2}']


def test_iter_json_objects_ignores_unbalanced_tail():
    assert list(iter_json_objects('{"a": 1} and {"b":')) == ['{"a": 1}']


# --------------------------------------------------------------------------
# Combined defects and total failure
# --------------------------------------------------------------------------

def test_fence_plus_latex_plus_unquoted_keys():
    reply = '```json\n{summary: "uses \\tau here", label: "l"}\n```'
    got = parse_reply(reply)
    assert got is not None and got["label"] == "l"


def test_unparseable_reply_returns_none_rather_than_raising():
    assert parse_reply("I'm afraid I can't do that.") is None


def test_empty_reply_returns_none():
    assert parse_reply("") is None and parse_reply("   ") is None


def test_none_input_returns_none():
    assert parse_reply(None) is None


def test_a_bare_scalar_is_not_a_result():
    assert parse_reply('"just a string"') is None
    assert parse_reply("42") is None
