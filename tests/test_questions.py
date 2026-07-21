import json
from pathlib import Path

import pytest

from rag_eval.corpus import load_corpus
from rag_eval.questions import Question, QuestionSet, load_questions

REPO_ROOT = Path(__file__).parent.parent
CORPUS_PATH = REPO_ROOT / "corpus"
QUESTIONS_PATH = REPO_ROOT / "questions.jsonl"

TEST_CORPUS_IDS = {"doc_01", "doc_02", "doc_03"}


def _valid_answerable(**overrides):
    row = {
        "id": "q001",
        "question": "why?",
        "answer": "because",
        "relevant_doc_ids": ["doc_01"],
        "answerable": True,
    }
    row.update(overrides)
    return row


def _valid_unanswerable(**overrides):
    row = {
        "id": "q002",
        "question": "what?",
        "answer": None,
        "relevant_doc_ids": [],
        "answerable": False,
    }
    row.update(overrides)
    return row


def _write(tmp_path, rows) -> Path:
    p = tmp_path / "questions.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return p


# ---------- happy path against the real fixtures ----------


def test_loads_all_20_from_fixture():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    qs = load_questions(QUESTIONS_PATH, corpus_ids)
    assert len(qs) == 20


def test_split_is_16_answerable_4_unanswerable():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    qs = load_questions(QUESTIONS_PATH, corpus_ids)
    assert len(qs.answerable_only()) == 16
    assert len(qs.unanswerable_only()) == 4


def test_unanswerable_have_null_answer_and_empty_relevant_docs():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    qs = load_questions(QUESTIONS_PATH, corpus_ids)
    for q in qs.unanswerable_only():
        assert q.answer is None
        assert q.relevant_doc_ids == ()


def test_answerable_have_answer_and_relevant_docs():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    qs = load_questions(QUESTIONS_PATH, corpus_ids)
    for q in qs.answerable_only():
        assert q.answer is not None
        assert q.relevant_doc_ids  # non-empty tuple


def test_question_is_frozen():
    q = Question(
        id="q", question="?", answer="!", relevant_doc_ids=("doc_01",), answerable=True
    )
    with pytest.raises(Exception):
        q.answer = "changed"  # type: ignore[misc]


def test_relevant_doc_ids_is_tuple_not_list():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    qs = load_questions(QUESTIONS_PATH, corpus_ids)
    for q in qs:
        assert isinstance(q.relevant_doc_ids, tuple)


def test_questionset_iterable_and_sized():
    qs = QuestionSet(
        (
            Question(
                id="a", question="?", answer="x", relevant_doc_ids=("d",), answerable=True
            ),
        )
    )
    assert len(qs) == 1
    assert [q.id for q in qs] == ["a"]


# ---------- failure modes ----------


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="questions file not found"):
        load_questions(tmp_path / "nope.jsonl", TEST_CORPUS_IDS)


def test_empty_file_raises(tmp_path):
    p = tmp_path / "questions.jsonl"
    p.write_text("")
    with pytest.raises(ValueError, match="no questions loaded"):
        load_questions(p, TEST_CORPUS_IDS)


def test_malformed_json_raises(tmp_path):
    p = tmp_path / "questions.jsonl"
    p.write_text("{not json\n")
    with pytest.raises(ValueError, match="line 1.*invalid JSON"):
        load_questions(p, TEST_CORPUS_IDS)


def test_missing_field_raises(tmp_path):
    row = _valid_answerable()
    del row["answerable"]
    p = _write(tmp_path, [row])
    with pytest.raises(ValueError, match="missing required fields.*answerable"):
        load_questions(p, TEST_CORPUS_IDS)


def test_wrong_type_answer_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(answer=123)])
    with pytest.raises(ValueError, match="answer must be a string or null"):
        load_questions(p, TEST_CORPUS_IDS)


def test_wrong_type_relevant_doc_ids_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(relevant_doc_ids="doc_01")])
    with pytest.raises(ValueError, match="relevant_doc_ids must be a list of strings"):
        load_questions(p, TEST_CORPUS_IDS)


def test_wrong_type_answerable_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(answerable="yes")])
    with pytest.raises(ValueError, match="answerable must be a bool"):
        load_questions(p, TEST_CORPUS_IDS)


def test_duplicate_id_raises(tmp_path):
    p = _write(
        tmp_path,
        [
            _valid_answerable(id="q001"),
            _valid_answerable(id="q001", question="different text"),
        ],
    )
    with pytest.raises(ValueError, match="duplicate id 'q001'"):
        load_questions(p, TEST_CORPUS_IDS)


def test_answerable_true_but_null_answer_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(answer=None)])
    with pytest.raises(ValueError, match="inconsistent answerable flag"):
        load_questions(p, TEST_CORPUS_IDS)


def test_answerable_true_but_empty_relevant_docs_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(relevant_doc_ids=[])])
    with pytest.raises(ValueError, match="inconsistent answerable flag"):
        load_questions(p, TEST_CORPUS_IDS)


def test_answerable_false_but_non_null_answer_raises(tmp_path):
    p = _write(tmp_path, [_valid_unanswerable(answer="something")])
    with pytest.raises(ValueError, match="inconsistent answerable flag"):
        load_questions(p, TEST_CORPUS_IDS)


def test_answerable_false_but_non_empty_relevant_docs_raises(tmp_path):
    p = _write(tmp_path, [_valid_unanswerable(relevant_doc_ids=["doc_01"])])
    with pytest.raises(ValueError, match="inconsistent answerable flag"):
        load_questions(p, TEST_CORPUS_IDS)


def test_unknown_doc_id_raises(tmp_path):
    p = _write(tmp_path, [_valid_answerable(relevant_doc_ids=["doc_99"])])
    with pytest.raises(ValueError, match="not present in corpus.*doc_99"):
        load_questions(p, TEST_CORPUS_IDS)


def test_error_message_includes_line_number(tmp_path):
    # Row 1 valid, row 2 broken — check the reported line is 2, not 1.
    p = _write(
        tmp_path,
        [
            _valid_answerable(id="q001"),
            _valid_answerable(id="q002", answerable="not-a-bool"),
        ],
    )
    with pytest.raises(ValueError, match="line 2"):
        load_questions(p, TEST_CORPUS_IDS)
