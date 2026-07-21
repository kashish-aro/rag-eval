import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rag_eval.corpus import load_corpus
from rag_eval.metrics_retrieval import aggregate_retrieval_metrics
from rag_eval.questions import load_questions
from rag_eval.retriever import Retriever

DEFAULT_RUNS_DIR = Path("runs")


def _slug(s: str) -> str:
    """Path-safe short slug for the filename. Not for the config record."""
    return s.rsplit("/", 1)[-1].lower().replace("chunker", "")


def run_retrieval_eval(
    corpus_path: str | Path,
    questions_path: str | Path,
    chunker: Any,
    embedder: Any,
    k: int,
    runs_dir: str | Path = DEFAULT_RUNS_DIR,
) -> Path:
    """Wire the retrieval pipeline end-to-end and write a run JSON.

    The written JSON records chunker, embedder_model, and k explicitly in the
    config section so a later config comparison can identify which pipeline
    produced which numbers. The comparison is the whole point of the harness;
    a report without its config is useless.
    """
    docs = load_corpus(corpus_path)
    corpus_ids = {d.doc_id for d in docs}
    questions = load_questions(questions_path, corpus_ids)

    chunks = chunker.chunk(docs)
    chunk_embeddings = embedder.embed([c.text for c in chunks])
    retriever = Retriever(chunks, chunk_embeddings)

    question_list = list(questions)
    query_embeddings = embedder.embed([q.question for q in question_list])

    results = [
        (q, retriever.query(query_embeddings[i], k))
        for i, q in enumerate(question_list)
    ]
    metrics = aggregate_retrieval_metrics(results, k)

    chunker_name = type(chunker).__name__
    config = {
        "chunker": chunker_name,
        "embedder_model": embedder.model_name,
        "k": k,
    }
    output = {
        "config": config,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "aggregate": {
            "k": metrics.k,
            "n_answerable": metrics.n_answerable,
            "n_unanswerable": metrics.n_unanswerable,
            "any_hit_rate": metrics.any_hit_rate,
            "mean_recall": metrics.mean_recall,
            "mean_reciprocal_rank": metrics.mean_reciprocal_rank,
        },
        "per_question": [
            {
                "question_id": p.question_id,
                "answerable": p.answerable,
                "retrieved_doc_ids": list(p.retrieved_doc_ids),
                "relevant_doc_ids": list(p.relevant_doc_ids),
                "any_hit_at_k": p.any_hit_at_k,
                "recall_at_k": p.recall_at_k,
                "reciprocal_rank": p.reciprocal_rank,
            }
            for p in metrics.per_question
        ],
    }

    runs_dir_p = Path(runs_dir)
    runs_dir_p.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    filename = (
        f"retrieval-{_slug(chunker_name)}-{_slug(embedder.model_name)}-k{k}-{ts}.json"
    )
    out_path = runs_dir_p / filename
    out_path.write_text(json.dumps(output, indent=2))
    return out_path


# Re-export so eval_retrieval-owned test files do not need to import asdict.
__all__ = ["run_retrieval_eval", "DEFAULT_RUNS_DIR"]
