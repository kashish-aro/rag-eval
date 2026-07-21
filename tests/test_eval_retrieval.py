import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from rag_eval.chunker import WholeDocChunker
from rag_eval.embedder import Embedder
from rag_eval.eval_retrieval import run_retrieval_eval

REPO_ROOT = Path(__file__).parent.parent
CORPUS_PATH = REPO_ROOT / "corpus"
QUESTIONS_PATH = REPO_ROOT / "questions.jsonl"


class _StubModel:
    """Deterministic hash-seeded encoder so fast tests never load ST."""

    def __init__(self, dim: int = 8, tag: str = "eval"):
        self.dim = dim
        self.tag = tag

    def encode(self, texts):
        vectors = []
        for t in texts:
            digest = hashlib.sha256((self.tag + t).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.RandomState(seed)
            vectors.append(rng.standard_normal(self.dim).astype(np.float32))
        return np.stack(vectors)


def _tiny_corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir(parents=True, exist_ok=True)
    (corpus / "doc_a.md").write_text("# Doc A\n\nAlpha content about apples.\n")
    (corpus / "doc_b.md").write_text("# Doc B\n\nBeta content about bananas.\n")
    (corpus / "doc_c.md").write_text("# Doc C\n\nGamma content about cherries.\n")
    return corpus


def _tiny_questions(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    p = tmp_path / "questions.jsonl"
    p.write_text(
        '{"id": "q1", "question": "apples?", "answer": "yes", '
        '"relevant_doc_ids": ["doc_a"], "answerable": true}\n'
        '{"id": "q2", "question": "bananas?", "answer": "yes", '
        '"relevant_doc_ids": ["doc_b"], "answerable": true}\n'
        '{"id": "u1", "question": "durian?", "answer": null, '
        '"relevant_doc_ids": [], "answerable": false}\n'
    )
    return p


def _run(tmp_path, *, chunker=None, embedder=None, k=2) -> dict:
    corpus = _tiny_corpus(tmp_path)
    questions = _tiny_questions(tmp_path)
    chunker = chunker or WholeDocChunker()
    embedder = embedder or Embedder(
        "stub-model", cache_dir=tmp_path / "cache", model=_StubModel()
    )
    out_path = run_retrieval_eval(
        corpus_path=corpus,
        questions_path=questions,
        chunker=chunker,
        embedder=embedder,
        k=k,
        runs_dir=tmp_path / "runs",
    )
    assert out_path.exists()
    return json.loads(out_path.read_text())


# ---------- fast tests ----------


def test_run_writes_file_and_returns_path(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    questions = _tiny_questions(tmp_path)
    embedder = Embedder(
        "stub-model", cache_dir=tmp_path / "cache", model=_StubModel()
    )
    runs = tmp_path / "runs"

    out_path = run_retrieval_eval(
        corpus_path=corpus,
        questions_path=questions,
        chunker=WholeDocChunker(),
        embedder=embedder,
        k=2,
        runs_dir=runs,
    )

    assert out_path.is_file()
    assert out_path.parent == runs
    assert out_path.suffix == ".json"


def test_config_section_records_chunker_embedder_and_k(tmp_path):
    report = _run(
        tmp_path,
        chunker=WholeDocChunker(),
        embedder=Embedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=tmp_path / "cache",
            model=_StubModel(),
        ),
        k=3,
    )
    assert report["config"] == {
        "chunker": "WholeDocChunker",
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2",
        "k": 3,
    }


def test_config_reflects_the_k_actually_used(tmp_path):
    # Swap k, config must follow — otherwise the report is meaningless
    # for a comparison across k values.
    r1 = _run(tmp_path / "a", k=1)
    r2 = _run(tmp_path / "b", k=3)
    assert r1["config"]["k"] == 1
    assert r2["config"]["k"] == 3


def test_aggregate_section_has_all_fields(tmp_path):
    report = _run(tmp_path)
    agg = report["aggregate"]
    assert set(agg.keys()) == {
        "k",
        "n_answerable",
        "n_unanswerable",
        "any_hit_rate",
        "mean_recall",
        "mean_reciprocal_rank",
    }


def test_per_question_has_one_entry_per_question(tmp_path):
    # Tiny fixtures: 2 answerable + 1 unanswerable = 3 rows expected.
    report = _run(tmp_path)
    assert len(report["per_question"]) == 3
    ids = [p["question_id"] for p in report["per_question"]]
    assert set(ids) == {"q1", "q2", "u1"}


def test_per_question_row_shape(tmp_path):
    report = _run(tmp_path)
    row = report["per_question"][0]
    assert set(row.keys()) == {
        "question_id",
        "answerable",
        "retrieved_doc_ids",
        "relevant_doc_ids",
        "any_hit_at_k",
        "recall_at_k",
        "reciprocal_rank",
    }


def test_unanswerable_row_carries_none_metrics_but_retrieved_docs(tmp_path):
    report = _run(tmp_path)
    u1 = next(p for p in report["per_question"] if p["question_id"] == "u1")
    assert u1["answerable"] is False
    assert u1["any_hit_at_k"] is None
    assert u1["recall_at_k"] is None
    assert u1["reciprocal_rank"] is None
    assert len(u1["retrieved_doc_ids"]) > 0


def test_filename_encodes_chunker_and_embedder_and_k(tmp_path):
    corpus = _tiny_corpus(tmp_path)
    questions = _tiny_questions(tmp_path)
    out = run_retrieval_eval(
        corpus_path=corpus,
        questions_path=questions,
        chunker=WholeDocChunker(),
        embedder=Embedder(
            "sentence-transformers/all-MiniLM-L6-v2",
            cache_dir=tmp_path / "cache",
            model=_StubModel(),
        ),
        k=5,
        runs_dir=tmp_path / "runs",
    )
    name = out.name
    assert "wholedoc" in name
    assert "all-minilm-l6-v2" in name
    assert "k5" in name


def test_generated_at_is_iso_utc(tmp_path):
    report = _run(tmp_path)
    ts = report["generated_at"]
    assert ts.endswith("+00:00") or ts.endswith("Z")


# ---------- slow test: real ST end-to-end ----------


@pytest.mark.slow
def test_real_pipeline_on_fixtures(tmp_path):
    embedder = Embedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=tmp_path / "cache",
    )
    out_path = run_retrieval_eval(
        corpus_path=CORPUS_PATH,
        questions_path=QUESTIONS_PATH,
        chunker=WholeDocChunker(),
        embedder=embedder,
        k=3,
        runs_dir=tmp_path / "runs",
    )
    report = json.loads(out_path.read_text())

    assert report["config"]["chunker"] == "WholeDocChunker"
    assert report["config"]["k"] == 3
    assert report["aggregate"]["n_answerable"] == 16
    assert report["aggregate"]["n_unanswerable"] == 4
    assert len(report["per_question"]) == 20
