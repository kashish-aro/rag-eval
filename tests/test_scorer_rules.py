from rag_eval.scorer_rules import (
    Verdict,
    _contains_refusal,
    _extract_number_facts,
    _normalize_number,
    score_rule_based,
)


# ---------- helpers: _normalize_number ----------


def test_normalize_strips_commas():
    assert _normalize_number("1,200") == "1200"


def test_normalize_preserves_decimal():
    assert _normalize_number("1.25") == "1.25"


def test_normalize_strips_trailing_decimal_zeros():
    assert _normalize_number("1.250") == "1.25"
    assert _normalize_number("1.0") == "1"
    assert _normalize_number("100.00") == "100"


def test_normalize_plain_integer_unchanged():
    assert _normalize_number("15") == "15"


# ---------- helpers: _extract_number_facts ----------


def test_extract_currency_number():
    assert "1200" in _extract_number_facts("The stipend is $1,200")


def test_extract_decimal_number():
    assert "1.25" in _extract_number_facts("Rate of 1.25 per month")


def test_extract_number_words():
    assert "15" in _extract_number_facts("fifteen days")
    assert "5" in _extract_number_facts("five days")
    assert "90" in _extract_number_facts("ninety-day expiration")


def test_extract_multiple_numbers():
    assert _extract_number_facts("15 days and 1.25 per month") == {"15", "1.25"}


def test_extract_word_and_digit_same_number_deduped():
    # "fifteen 15 days" should produce {"15"} not two entries.
    assert _extract_number_facts("fifteen 15 days") == {"15"}


def test_extract_from_empty_string():
    assert _extract_number_facts("") == set()


# ---------- helpers: _contains_refusal ----------


def test_refusal_exact_match():
    assert _contains_refusal("I don't know.")


def test_refusal_embedded_in_longer_text():
    assert _contains_refusal("The answer is: I don't know.")


def test_refusal_case_insensitive():
    assert _contains_refusal("I DON'T KNOW.")


def test_non_refusal_is_not_refusal():
    assert not _contains_refusal("The answer is 15 days.")
    assert not _contains_refusal("")


# ---------- score_rule_based: currency normalization ----------


def test_currency_dollar_sign_matches_dollar_sign_reference():
    assert (
        score_rule_based("The stipend is $1,200.", "The stipend is $1,200.")
        == Verdict.CORRECT
    )


def test_bare_number_matches_dollar_sign_reference():
    assert (
        score_rule_based("The stipend is $1,200.", "The stipend is 1200.")
        == Verdict.CORRECT
    )


def test_dollars_word_matches_dollar_sign_reference():
    assert (
        score_rule_based("The stipend is $1,200.", "The stipend is 1,200 dollars.")
        == Verdict.CORRECT
    )


# ---------- score_rule_based: number-word normalization ----------


def test_spelled_out_number_matches_digit_reference():
    assert score_rule_based("15 days.", "The accrual is fifteen days.") == Verdict.CORRECT


def test_bare_digit_matches_number_plus_unit_reference():
    # Per the user's requirement: "just 15" matches "15 days".
    assert score_rule_based("15 days.", "15") == Verdict.CORRECT


# ---------- score_rule_based: multi-fact strictness (the over-credit guard) ----------


def test_partial_answer_missing_second_number_is_incorrect():
    # Reference has TWO numeric facts (15 and 1.25); answer has one.
    # This is the primary over-credit protection: a partial answer that
    # matches on one fact but not the other must NOT be CORRECT.
    verdict = score_rule_based(
        "15 days in the first year, accruing at 1.25 days per month.",
        "Just 15 days.",
    )
    assert verdict == Verdict.INCORRECT


def test_partial_answer_missing_second_dollar_amount_is_incorrect():
    verdict = score_rule_based(
        "$1,200 stipend, expiring after 90 days.",
        "The stipend is $1,200.",
    )
    assert verdict == Verdict.INCORRECT


def test_all_facts_present_is_correct():
    verdict = score_rule_based(
        "15 days in the first year, accruing at 1.25 days per month.",
        "New hires get 15 days per year, which is 1.25 days a month.",
    )
    assert verdict == Verdict.CORRECT


# ---------- score_rule_based: wrong / empty answers ----------


def test_wrong_number_is_incorrect():
    assert score_rule_based("15 days.", "The accrual is 20 days.") == Verdict.INCORRECT


def test_empty_answer_on_answerable_is_incorrect():
    assert score_rule_based("15 days.", "") == Verdict.INCORRECT


# ---------- score_rule_based: refusal on answerable ----------


def test_answerable_refused_verdict_is_refused():
    # REFUSED is distinct from INCORRECT so the aggregator can distinguish
    # "gave up" (fixable with better retrieval/prompting) from "guessed
    # wrong" (fixable elsewhere).
    assert score_rule_based("15 days.", "I don't know.") == Verdict.REFUSED


# ---------- score_rule_based: unanswerable ----------


def test_unanswerable_with_refusal_is_correct():
    assert score_rule_based(None, "I don't know.") == Verdict.CORRECT


def test_unanswerable_with_refusal_in_longer_text_is_correct():
    assert score_rule_based(None, "Sorry, I don't know.") == Verdict.CORRECT


def test_unanswerable_with_confident_answer_is_incorrect():
    # Model hallucinated a specific fact for a question that has no answer.
    assert score_rule_based(None, "The match is 5%.") == Verdict.INCORRECT


def test_unanswerable_with_empty_answer_is_incorrect():
    # Empty is not a refusal — it's a non-response, treated as failure.
    assert score_rule_based(None, "") == Verdict.INCORRECT


# ---------- score_rule_based: design decisions documented in tests ----------


def test_extra_numbers_in_answer_do_not_penalize():
    # Design choice: mentioning extra numbers is not counted against the
    # answer. A verbose but correct answer should score CORRECT. Penalizing
    # verbosity would depress scores on well-formed answers.
    verdict = score_rule_based(
        "The accrual is 15 days.",
        "Employees get 15 days per year, or maybe 30 in special cases.",
    )
    assert verdict == Verdict.CORRECT


def test_reference_without_numbers_is_incorrect_conservatively():
    # A reference with no numeric fingerprint can't be verified. We return
    # INCORRECT rather than falsely CORRECT — under-crediting is preferable
    # to over-crediting per the design constraint. This is a known
    # limitation of the number-only strategy.
    assert score_rule_based("Yes.", "Yes, absolutely.") == Verdict.INCORRECT


# ---------- Verdict enum ----------


def test_verdict_is_json_serializable_as_string():
    # Verdict inherits from str so json.dumps gives the string value
    # directly, without needing a custom encoder.
    import json

    assert json.dumps(Verdict.CORRECT) == '"CORRECT"'
