import pytest

from rag_eval.judge import Judge, _parse_verdict
from rag_eval.scorer_rules import Verdict


# ---------- Verdict enum ----------


def test_verdict_has_unparseable():
    assert Verdict.UNPARSEABLE.value == "UNPARSEABLE"


# ---------- _parse_verdict: happy path ----------


def test_parse_clean_verdict_line_correct():
    assert _parse_verdict("VERDICT: CORRECT") == Verdict.CORRECT


def test_parse_all_three_valid_tokens():
    assert _parse_verdict("VERDICT: CORRECT") == Verdict.CORRECT
    assert _parse_verdict("VERDICT: INCORRECT") == Verdict.INCORRECT
    assert _parse_verdict("VERDICT: REFUSED") == Verdict.REFUSED


def test_parse_allows_preamble_before_verdict_line():
    text = (
        "Let me analyze this carefully.\n\n"
        "The reference says 15 days and the answer says 15 days.\n\n"
        "VERDICT: CORRECT"
    )
    assert _parse_verdict(text) == Verdict.CORRECT


def test_parse_trailing_period_tolerated():
    # Small models add punctuation. Allow a single trailing period.
    assert _parse_verdict("VERDICT: CORRECT.") == Verdict.CORRECT


def test_parse_extra_whitespace_around_token():
    assert _parse_verdict("VERDICT:   CORRECT   ") == Verdict.CORRECT


def test_parse_repeated_same_verdict_is_ok():
    # Same verdict twice is not "conflicting" — it's redundant reassurance.
    text = "VERDICT: CORRECT\n\nJust to be clear:\nVERDICT: CORRECT"
    assert _parse_verdict(text) == Verdict.CORRECT


# ---------- _parse_verdict: UNPARSEABLE cases ----------


def test_parse_missing_verdict_line_is_unparseable():
    assert _parse_verdict("The answer looks reasonable to me.") == Verdict.UNPARSEABLE


def test_parse_verdict_word_buried_in_prose_is_unparseable():
    # "The verdict is CORRECT" is chatty preamble, not a VERDICT: line.
    # This is the case that would silently coerce into CORRECT under a
    # loose "first CORRECT wins" parser — exactly what we do not want.
    assert (
        _parse_verdict("The verdict is CORRECT because the facts match.")
        == Verdict.UNPARSEABLE
    )


def test_parse_lowercase_label_is_unparseable():
    assert _parse_verdict("verdict: CORRECT") == Verdict.UNPARSEABLE


def test_parse_lowercase_token_is_unparseable():
    # The prompt says uppercase; the parser enforces it. A mixed-case token
    # is non-compliance and must not silently coerce.
    assert _parse_verdict("VERDICT: correct") == Verdict.UNPARSEABLE
    assert _parse_verdict("VERDICT: Correct") == Verdict.UNPARSEABLE


def test_parse_unknown_token_is_unparseable():
    assert _parse_verdict("VERDICT: MAYBE") == Verdict.UNPARSEABLE
    assert _parse_verdict("VERDICT: UNSURE") == Verdict.UNPARSEABLE


def test_parse_verdict_never_self_selects_unparseable():
    # The judge should never write "VERDICT: UNPARSEABLE" — that value is
    # only ever emitted by the parser. If the model tries, treat as
    # non-compliance.
    assert _parse_verdict("VERDICT: UNPARSEABLE") == Verdict.UNPARSEABLE


def test_parse_conflicting_verdict_lines_is_unparseable():
    text = (
        "First analysis:\nVERDICT: CORRECT\n\n"
        "On reflection:\nVERDICT: INCORRECT"
    )
    assert _parse_verdict(text) == Verdict.UNPARSEABLE


def test_parse_trailing_content_after_verdict_is_unparseable():
    # "VERDICT: CORRECT and here's why" — extra content after the token
    # violates "nothing after the VERDICT line."
    assert _parse_verdict("VERDICT: CORRECT and here's why") == Verdict.UNPARSEABLE


def test_parse_empty_string_is_unparseable():
    assert _parse_verdict("") == Verdict.UNPARSEABLE


def test_parse_only_whitespace_is_unparseable():
    assert _parse_verdict("\n\n   \n") == Verdict.UNPARSEABLE


# ---------- Judge class (fast, with stub generator) ----------


class _StubGenerator:
    """Stand-in for the real Generator; returns a canned response."""

    def __init__(self, canned_response: str, model_name: str = "stub-judge"):
        self.canned_response = canned_response
        self.model_name = model_name
        self.calls: list[list[dict]] = []

    def generate(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.canned_response


def test_judge_returns_parsed_verdict_from_generator_output():
    j = Judge(generator=_StubGenerator("VERDICT: CORRECT"))
    assert j.judge("q?", "ref", "ctx", "ans") == Verdict.CORRECT


def test_judge_returns_unparseable_when_generator_output_lacks_verdict_line():
    j = Judge(generator=_StubGenerator("I'm not sure what to say."))
    assert j.judge("q?", "ref", "ctx", "ans") == Verdict.UNPARSEABLE


def test_judge_passes_all_four_inputs_to_generator():
    stub = _StubGenerator("VERDICT: CORRECT")
    Judge(generator=stub).judge(
        "What is 2+2?", "4", "math context", "The answer is 4."
    )
    assert len(stub.calls) == 1
    body = stub.calls[0][1]["content"]
    assert "What is 2+2?" in body
    assert "4" in body
    assert "math context" in body
    assert "The answer is 4." in body


def test_judge_signals_unanswerable_when_reference_is_none():
    stub = _StubGenerator("VERDICT: CORRECT")
    Judge(generator=stub).judge("weird question?", None, "ctx", "I don't know.")
    body = stub.calls[0][1]["content"]
    assert "UNANSWERABLE" in body


def test_judge_exposes_model_name_from_generator():
    j = Judge(generator=_StubGenerator("VERDICT: CORRECT", model_name="my-model"))
    assert j.model_name == "my-model"


# ---------- Slow: real Qwen judge on three canned cases ----------


@pytest.mark.slow
def test_real_judge_correct_case():
    j = Judge()
    verdict = j.judge(
        question="How many PTO days do new hires get in their first year?",
        reference="15 days in the first year, accruing at 1.25 days per month.",
        context=(
            "Full-time employees accrue 15 days of paid time off during "
            "their first year of employment, accruing at a rate of 1.25 "
            "days per month."
        ),
        generated="New hires get 15 days of PTO in their first year, accruing at 1.25 days per month.",
    )
    assert verdict == Verdict.CORRECT


@pytest.mark.slow
def test_real_judge_incorrect_case():
    j = Judge()
    verdict = j.judge(
        question="How many PTO days do new hires get?",
        reference="15 days in the first year.",
        context="Employees get 15 days of PTO their first year.",
        generated="New hires get 30 days of PTO in their first year.",
    )
    assert verdict == Verdict.INCORRECT


@pytest.mark.slow
def test_real_judge_refused_case():
    j = Judge()
    verdict = j.judge(
        question="How many PTO days do new hires get?",
        reference="15 days in the first year.",
        context="Employees get 15 days.",
        generated="I don't know.",
    )
    assert verdict == Verdict.REFUSED
