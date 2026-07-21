from dataclasses import dataclass

from rag_eval.questions import Question
from rag_eval.retriever import Retrieved


@dataclass(frozen=True)
class PerQuestionMetrics:
    question_id: str
    answerable: bool
    # Collapsed top-k doc_ids in rank order — for both answerable and
    # unanswerable questions, so a reader can see what retrieval returned
    # even for questions we could not score.
    retrieved_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    # None for unanswerable questions: metrics are undefined when there is
    # nothing relevant to retrieve. Scoring these as zero would depress the
    # aggregate; scoring them as one would inflate it. Neither is honest.
    any_hit_at_k: bool | None
    recall_at_k: float | None
    reciprocal_rank: float | None


@dataclass(frozen=True)
class AggregateRetrievalMetrics:
    k: int
    n_answerable: int
    n_unanswerable: int
    any_hit_rate: float
    mean_recall: float
    mean_reciprocal_rank: float
    per_question: tuple[PerQuestionMetrics, ...]


def collapse_to_docs(retrieved: list[Retrieved], k: int) -> tuple[str, ...]:
    """Take top-k retrieved chunks and return unique doc_ids in first-seen order.

    Metrics score at the document level; the retriever operates on chunks.
    Without this collapse, two chunks of the same doc landing in top-k would
    silently inflate hit@k and recall@k. dict insertion order (Py 3.7+) makes
    the collapse deterministic.
    """
    if k <= 0:
        raise ValueError(f"k must be >= 1, got {k}")
    seen: dict[str, None] = {}
    for r in retrieved[:k]:
        seen.setdefault(r.doc_id, None)
    return tuple(seen.keys())


def any_hit_at_k(
    retrieved: list[Retrieved], relevant: set[str], k: int
) -> bool:
    if not relevant:
        raise ValueError("any_hit_at_k undefined for empty relevant set")
    return any(d in relevant for d in collapse_to_docs(retrieved, k))


def recall_at_k(
    retrieved: list[Retrieved], relevant: set[str], k: int
) -> float:
    if not relevant:
        raise ValueError("recall_at_k undefined for empty relevant set")
    docs = collapse_to_docs(retrieved, k)
    hits = sum(1 for d in docs if d in relevant)
    return hits / len(relevant)


def reciprocal_rank(
    retrieved: list[Retrieved], relevant: set[str], k: int
) -> float:
    """1 / (rank of first relevant doc within collapsed top-k), else 0."""
    if not relevant:
        raise ValueError("reciprocal_rank undefined for empty relevant set")
    for rank, d in enumerate(collapse_to_docs(retrieved, k), start=1):
        if d in relevant:
            return 1.0 / rank
    return 0.0


def aggregate_retrieval_metrics(
    question_results: list[tuple[Question, list[Retrieved]]],
    k: int,
) -> AggregateRetrievalMetrics:
    """Aggregate metrics over answerable questions only; carry per-question detail.

    Unanswerable questions do not contribute to the aggregate rates — there
    is nothing to retrieve, so the metrics are undefined rather than zero.
    Their per-question rows appear with `None` metric values but a populated
    `retrieved_doc_ids` field, so a reader can inspect what retrieval
    returned for them.
    """
    per_q: list[PerQuestionMetrics] = []
    hits: list[bool] = []
    recalls: list[float] = []
    rrs: list[float] = []

    for question, retrieved in question_results:
        collapsed = collapse_to_docs(retrieved, k) if retrieved else ()

        if question.answerable:
            relevant_set = set(question.relevant_doc_ids)
            hit = any_hit_at_k(retrieved, relevant_set, k)
            rec = recall_at_k(retrieved, relevant_set, k)
            rr = reciprocal_rank(retrieved, relevant_set, k)
            hits.append(hit)
            recalls.append(rec)
            rrs.append(rr)
            per_q.append(
                PerQuestionMetrics(
                    question_id=question.id,
                    answerable=True,
                    retrieved_doc_ids=collapsed,
                    relevant_doc_ids=question.relevant_doc_ids,
                    any_hit_at_k=hit,
                    recall_at_k=rec,
                    reciprocal_rank=rr,
                )
            )
        else:
            per_q.append(
                PerQuestionMetrics(
                    question_id=question.id,
                    answerable=False,
                    retrieved_doc_ids=collapsed,
                    relevant_doc_ids=question.relevant_doc_ids,
                    any_hit_at_k=None,
                    recall_at_k=None,
                    reciprocal_rank=None,
                )
            )

    n_ans = len(hits)
    return AggregateRetrievalMetrics(
        k=k,
        n_answerable=n_ans,
        n_unanswerable=len(per_q) - n_ans,
        any_hit_rate=sum(hits) / n_ans if n_ans else 0.0,
        mean_recall=sum(recalls) / n_ans if n_ans else 0.0,
        mean_reciprocal_rank=sum(rrs) / n_ans if n_ans else 0.0,
        per_question=tuple(per_q),
    )
