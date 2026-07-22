import re
from enum import Enum

from rag_eval.prompts import REFUSAL_STRING


class Verdict(str, Enum):
    """Faithfulness verdict per (reference, generated) pair.

    Semantics: "did the model do the right thing for this question type."
      CORRECT     — matched reference, or refused an unanswerable question
      INCORRECT   — wrong facts, or answered when it should have refused
      REFUSED     — model gave up on an answerable question
      UNPARSEABLE — the scorer could not extract a verdict (only ever
                    emitted by LLM-based scorers, e.g. judge.py; the rule-
                    based scorer never returns this)

    REFUSED is distinct from INCORRECT so the aggregator can measure
    "give up" separately from "guess wrong" — different failure modes
    warrant different fixes upstream. UNPARSEABLE is a real outcome, not
    an error: a rising rate means the judge is drifting and its numbers
    should not be trusted.
    """

    CORRECT = "CORRECT"
    INCORRECT = "INCORRECT"
    REFUSED = "REFUSED"
    UNPARSEABLE = "UNPARSEABLE"


# Digit strings with optional thousands separators and optional decimal.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
    "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60",
    "seventy": "70", "eighty": "80", "ninety": "90",
}


def _normalize_number(num_str: str) -> str:
    """Strip commas and trailing decimal zeros: '1,200' -> '1200', '1.250' -> '1.25'."""
    s = num_str.replace(",", "")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return s


def _extract_number_facts(text: str) -> set[str]:
    """Return the set of normalized numeric facts in text (digits + spelled-out words)."""
    lower = text.lower()
    facts: set[str] = set()
    for m in _NUMBER_RE.finditer(lower):
        facts.add(_normalize_number(m.group()))
    for word in re.findall(r"\b[a-z]+\b", lower):
        if word in _NUMBER_WORDS:
            facts.add(_NUMBER_WORDS[word])
    return facts


def _contains_refusal(text: str) -> bool:
    """Case-insensitive substring match on the canonical refusal string.

    Tight coupling to REFUSAL_STRING from prompts.py is intentional: the
    system prompt tells the generator to emit that exact string, so if the
    constant changes both the prompt contract and this check must move
    together.
    """
    return REFUSAL_STRING.lower() in text.lower()


def score_rule_based(reference: str | None, generated: str) -> Verdict:
    """Faithfulness verdict for one (reference, generated) pair.

    Answerable (reference is not None):
      - Refused → REFUSED
      - Every numeric fact in reference present in answer → CORRECT
      - Otherwise → INCORRECT (this includes partial-fact answers; the
        "don't over-credit" rule is what makes this scorer standalone-usable
        if the LLM judge fails its kappa gate)

    Unanswerable (reference is None):
      - Refused → CORRECT
      - Otherwise → INCORRECT (hallucinated on a question with no answer)
    """
    refused = _contains_refusal(generated)

    if reference is None:
        return Verdict.CORRECT if refused else Verdict.INCORRECT

    if refused:
        return Verdict.REFUSED

    ref_facts = _extract_number_facts(reference)
    ans_facts = _extract_number_facts(generated)

    if not ref_facts:
        # No numeric fingerprint to verify against. Conservative: INCORRECT.
        # Under-crediting is preferable to over-crediting; a scorer that
        # says CORRECT when it can't actually check is worse than one that
        # says INCORRECT and forces the reader to look at the case.
        return Verdict.INCORRECT

    return Verdict.CORRECT if ref_facts <= ans_facts else Verdict.INCORRECT
