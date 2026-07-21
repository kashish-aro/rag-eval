"""Guards against silent corpus / questions.jsonl drift.

Ground truth in questions.jsonl references docs by filename stem. If a corpus
file is renamed or removed, retrieval metrics silently compute wrong values:
questions referencing the missing id register as retrieval failures regardless
of what the retriever actually returned. This test surfaces that class of drift
before an eval run masks it as poor retrieval quality.

Deliberately does not use the future question loader (Step 3). Parsing
questions.jsonl directly means a bug in the loader cannot hide a mismatch here.
"""
import json
from pathlib import Path

from rag_eval.corpus import load_corpus

REPO_ROOT = Path(__file__).parent.parent
CORPUS_PATH = REPO_ROOT / "corpus"
QUESTIONS_PATH = REPO_ROOT / "questions.jsonl"


def _referenced_doc_ids() -> set[str]:
    ids: set[str] = set()
    with QUESTIONS_PATH.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids.update(row["relevant_doc_ids"])
    return ids


def test_every_referenced_doc_id_exists_in_corpus():
    corpus_ids = {d.doc_id for d in load_corpus(CORPUS_PATH)}
    referenced = _referenced_doc_ids()
    missing = referenced - corpus_ids
    assert not missing, (
        f"questions.jsonl references doc_ids not in corpus/: {sorted(missing)}. "
        "Either restore the corpus files or fix questions.jsonl."
    )
