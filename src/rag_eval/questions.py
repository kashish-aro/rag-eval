import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

REQUIRED_FIELDS = frozenset(
    {"id", "question", "answer", "relevant_doc_ids", "answerable"}
)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answer: str | None
    relevant_doc_ids: tuple[str, ...]
    answerable: bool


@dataclass(frozen=True)
class QuestionSet:
    questions: tuple[Question, ...]

    def __iter__(self) -> Iterator[Question]:
        return iter(self.questions)

    def __len__(self) -> int:
        return len(self.questions)

    def answerable_only(self) -> "QuestionSet":
        return QuestionSet(tuple(q for q in self.questions if q.answerable))

    def unanswerable_only(self) -> "QuestionSet":
        return QuestionSet(tuple(q for q in self.questions if not q.answerable))


def load_questions(path: str | Path, corpus_doc_ids: set[str]) -> QuestionSet:
    """Load and validate questions.jsonl, failing loudly on any invariant break.

    questions.jsonl is authoritative ground truth. If a future edit produces
    duplicate ids, an inconsistent answerable flag, or a doc_id that no longer
    exists in the corpus, loading raises rather than letting the eval score
    against a broken set. corpus_doc_ids is passed in (not derived here) so
    callers control which corpus the questions are validated against.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"questions file not found: {p}")

    loaded: list[Question] = []
    seen_ids: set[str] = set()

    with p.open(encoding="utf-8") as f:
        for lineno, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue

            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"line {lineno}: invalid JSON — {e}") from e

            missing = REQUIRED_FIELDS - set(row.keys())
            if missing:
                raise ValueError(
                    f"line {lineno}: missing required fields {sorted(missing)}"
                )

            qid = row["id"]
            question_text = row["question"]
            answer = row["answer"]
            relevant = row["relevant_doc_ids"]
            answerable = row["answerable"]

            if not isinstance(qid, str) or not qid:
                raise ValueError(
                    f"line {lineno}: id must be a non-empty string, got {qid!r}"
                )
            # isinstance(x, bool) is True for bools; check bool first to
            # reject True/False slipping in where a string is expected.
            if not isinstance(question_text, str) or not question_text:
                raise ValueError(
                    f"line {lineno} ({qid}): question must be a non-empty string"
                )
            if answer is not None and not isinstance(answer, str):
                raise ValueError(
                    f"line {lineno} ({qid}): answer must be a string or null, "
                    f"got {type(answer).__name__}"
                )
            if not isinstance(relevant, list) or not all(
                isinstance(x, str) for x in relevant
            ):
                raise ValueError(
                    f"line {lineno} ({qid}): relevant_doc_ids must be a list of strings"
                )
            if not isinstance(answerable, bool):
                raise ValueError(
                    f"line {lineno} ({qid}): answerable must be a bool, "
                    f"got {type(answerable).__name__}"
                )

            if qid in seen_ids:
                raise ValueError(f"line {lineno}: duplicate id {qid!r}")
            seen_ids.add(qid)

            has_answer = answer is not None
            has_docs = bool(relevant)
            if not (answerable == has_answer == has_docs):
                raise ValueError(
                    f"line {lineno} ({qid}): inconsistent answerable flag — "
                    f"answerable={answerable}, answer_is_not_none={has_answer}, "
                    f"relevant_doc_ids_nonempty={has_docs}. All three must agree."
                )

            unknown = [d for d in relevant if d not in corpus_doc_ids]
            if unknown:
                raise ValueError(
                    f"line {lineno} ({qid}): relevant_doc_ids reference docs "
                    f"not present in corpus: {unknown}"
                )

            loaded.append(
                Question(
                    id=qid,
                    question=question_text,
                    answer=answer,
                    relevant_doc_ids=tuple(relevant),
                    answerable=answerable,
                )
            )

    if not loaded:
        raise ValueError(f"no questions loaded from {p}")

    return QuestionSet(tuple(loaded))
