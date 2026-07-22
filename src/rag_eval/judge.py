import re

from rag_eval.generator import Generator
from rag_eval.prompts import render_judge_prompt
from rag_eval.scorer_rules import Verdict

DEFAULT_JUDGE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_JUDGE_MAX_NEW_TOKENS = 300

# VERDICT: <TOKEN> as its own line. Uppercase label, uppercase token, optional
# trailing period (small models add punctuation), nothing else on the line.
# MULTILINE anchors ^ and $ at line boundaries so a preamble is fine as long
# as the verdict line stands alone at the end. `[A-Z]+` deliberately rejects
# lowercase or mixed-case tokens — a scorer that silently coerces
# "verdict: correct" into a verdict is exactly the failure mode we want to
# avoid.
_VERDICT_LINE_RE = re.compile(r"^VERDICT:\s*([A-Z]+)\.?\s*$", re.MULTILINE)

_VALID_JUDGE_VERDICTS = {Verdict.CORRECT, Verdict.INCORRECT, Verdict.REFUSED}


def _parse_verdict(text: str) -> Verdict:
    """Extract the verdict from a judge's response.

    Returns UNPARSEABLE (a first-class outcome, not an error) if:
      - no VERDICT: line is present at line-start,
      - the token is not in {CORRECT, INCORRECT, REFUSED},
      - the label or token is not uppercase,
      - the line has trailing content beyond an optional period,
      - multiple VERDICT: lines contain conflicting verdicts.

    UNPARSEABLE is tracked as a rate in downstream aggregation; a rising
    rate signals the judge is drifting and its numbers should not be
    trusted for that run.
    """
    matches = _VERDICT_LINE_RE.findall(text)
    if not matches:
        return Verdict.UNPARSEABLE

    unique_tokens = {m for m in matches}
    if len(unique_tokens) > 1:
        return Verdict.UNPARSEABLE

    token = unique_tokens.pop()
    try:
        verdict = Verdict(token)
    except ValueError:
        return Verdict.UNPARSEABLE

    if verdict not in _VALID_JUDGE_VERDICTS:
        # The judge should never emit UNPARSEABLE itself, only the parser.
        return Verdict.UNPARSEABLE

    return verdict


class Judge:
    """LLM-as-judge over Qwen 2.5 Instruct.

    Wraps a Generator (greedy, deterministic) with the judge prompt and the
    verdict parser. Generator is injectable so a single Qwen instance can be
    shared between the answer-generator and the judge — loading two copies
    is 6GB otherwise.
    """

    def __init__(
        self,
        generator: Generator | None = None,
        model_name: str = DEFAULT_JUDGE_MODEL,
        max_new_tokens: int = DEFAULT_JUDGE_MAX_NEW_TOKENS,
    ):
        self.generator = generator or Generator(
            model_name=model_name,
            max_new_tokens=max_new_tokens,
        )
        self.model_name = self.generator.model_name

    def judge(
        self,
        question: str,
        reference: str | None,
        context: str,
        generated: str,
    ) -> Verdict:
        messages = render_judge_prompt(question, reference, context, generated)
        response = self.generator.generate(messages)
        return _parse_verdict(response)
