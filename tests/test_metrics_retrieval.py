import pytest

from rag_eval.metrics_retrieval import (
    aggregate_retrieval_metrics,
    any_hit_at_k,
    collapse_to_docs,
    recall_at_k,
    reciprocal_rank,
)
from rag_eval.questions import Question
from rag_eval.retriever import Retrieved


def _r(chunk_id: str, doc_id: str, score: float = 1.0) -> Retrieved:
    return Retrieved(chunk_id=chunk_id, doc_id=doc_id, score=score)


def _q(qid: str, relevant: list[str], answerable: bool = True) -> Question:
    return Question(
        id=qid,
        question="?",
        answer="a" if answerable else None,
        relevant_doc_ids=tuple(relevant),
        answerable=answerable,
    )


# ---------- collapse_to_docs ----------


def test_collapse_no_duplicates():
    retrieved = [_r("a#0", "a"), _r("b#0", "b"), _r("c#0", "c")]
    assert collapse_to_docs(retrieved, k=3) == ("a", "b", "c")


def test_collapse_deduplicates_within_top_k():
    # Two chunks of the same doc in top-k must collapse to one doc entry.
    retrieved = [_r("a#0", "a"), _r("a#1", "a"), _r("b#0", "b")]
    assert collapse_to_docs(retrieved, k=3) == ("a", "b")


def test_collapse_respects_k_upper_bound():
    retrieved = [_r("a#0", "a"), _r("b#0", "b"), _r("c#0", "c")]
    assert collapse_to_docs(retrieved, k=2) == ("a", "b")


def test_collapse_ignores_duplicates_outside_top_k():
    # a#1 is a duplicate of a, but it sits outside the top-2 window.
    retrieved = [_r("a#0", "a"), _r("b#0", "b"), _r("a#1", "a")]
    assert collapse_to_docs(retrieved, k=2) == ("a", "b")


def test_collapse_k_zero_raises():
    with pytest.raises(ValueError, match="k must be"):
        collapse_to_docs([_r("a#0", "a")], k=0)


def test_collapse_preserves_first_seen_order():
    # b appears first (chunk b#0), then a. Order must be (b, a).
    retrieved = [_r("b#0", "b"), _r("a#0", "a"), _r("b#1", "b")]
    assert collapse_to_docs(retrieved, k=3) == ("b", "a")


# ---------- any_hit_at_k ----------


def test_any_hit_true_when_relevant_in_top_k():
    assert any_hit_at_k([_r("a#0", "a")], relevant={"a"}, k=1) is True


def test_any_hit_false_when_relevant_not_in_top_k():
    assert any_hit_at_k([_r("a#0", "a")], relevant={"b"}, k=1) is False


def test_any_hit_duplicate_chunks_are_one_hit_not_two():
    # The invariant the whole chunk_id contract is built for: two chunks of
    # doc_05 both in top-3 counts as one doc hit, not two. We express that as
    # a boolean here (True vs True is not distinguishing), but the important
    # property is that it does not silently inflate downstream metrics —
    # covered in the recall tests below.
    retrieved = [_r("doc_05#0", "doc_05"), _r("doc_05#1", "doc_05"), _r("doc_03#0", "doc_03")]
    assert any_hit_at_k(retrieved, relevant={"doc_05"}, k=3) is True


def test_any_hit_empty_relevant_raises():
    with pytest.raises(ValueError, match="undefined for empty relevant"):
        any_hit_at_k([_r("a#0", "a")], relevant=set(), k=1)


# ---------- recall_at_k ----------


def test_recall_single_relevant_hit():
    assert recall_at_k([_r("a#0", "a")], relevant={"a"}, k=1) == 1.0


def test_recall_single_relevant_miss():
    assert recall_at_k([_r("b#0", "b")], relevant={"a"}, k=1) == 0.0


def test_recall_multi_doc_one_of_two():
    retrieved = [_r("a#0", "a"), _r("c#0", "c")]
    assert recall_at_k(retrieved, relevant={"a", "b"}, k=2) == 0.5


def test_recall_multi_doc_both_of_two():
    retrieved = [_r("a#0", "a"), _r("b#0", "b")]
    assert recall_at_k(retrieved, relevant={"a", "b"}, k=2) == 1.0


def test_recall_chunk_duplication_does_not_inflate():
    # Three chunks of doc_a with relevant {a, b} at k=3.
    # Chunk-level counting would give 3/2 = 1.5, which is nonsense.
    # Doc-level counting after collapse gives 1/2 = 0.5. That's the point
    # of the whole chunk_id contract.
    retrieved = [_r("a#0", "a"), _r("a#1", "a"), _r("a#2", "a")]
    assert recall_at_k(retrieved, relevant={"a", "b"}, k=3) == 0.5


def test_recall_empty_relevant_raises():
    with pytest.raises(ValueError, match="undefined for empty relevant"):
        recall_at_k([_r("a#0", "a")], relevant=set(), k=1)


# ---------- reciprocal_rank ----------


def test_rr_first_position():
    retrieved = [_r("a#0", "a"), _r("b#0", "b")]
    assert reciprocal_rank(retrieved, relevant={"a"}, k=2) == 1.0


def test_rr_second_position():
    retrieved = [_r("b#0", "b"), _r("a#0", "a")]
    assert reciprocal_rank(retrieved, relevant={"a"}, k=2) == 0.5


def test_rr_not_found_within_k():
    retrieved = [_r("b#0", "b"), _r("c#0", "c")]
    assert reciprocal_rank(retrieved, relevant={"a"}, k=2) == 0.0


def test_rr_rank_is_at_doc_level_not_chunk_level():
    # b is the 2nd distinct doc after collapse (a takes positions 1-2 as
    # chunks, but only rank 1 as a doc). Rank of b is 2, not 3. MRR = 1/2.
    retrieved = [_r("a#0", "a"), _r("a#1", "a"), _r("b#0", "b")]
    assert reciprocal_rank(retrieved, relevant={"b"}, k=3) == 0.5


def test_rr_empty_relevant_raises():
    with pytest.raises(ValueError, match="undefined for empty relevant"):
        reciprocal_rank([_r("a#0", "a")], relevant=set(), k=1)


# ---------- aggregator ----------


def test_aggregator_hand_worked_example():
    # 3 answerable + 2 unanswerable, values computed by hand.
    #
    # q1: relevant {a},    retrieved [a#0, b#0, c#0]   -> hit=T, recall=1,   rr=1
    # q2: relevant {a, b}, retrieved [a#0, c#0, e#0]   -> hit=T, recall=0.5, rr=1
    # q3: relevant {x},    retrieved [a#0, b#0, c#0]   -> hit=F, recall=0,   rr=0
    # u1: unanswerable,    retrieved [a#0]             -> excluded
    # u2: unanswerable,    retrieved [b#0]             -> excluded
    #
    # Aggregates over 3 answerable:
    #   any_hit_rate       = 2/3
    #   mean_recall        = (1 + 0.5 + 0) / 3 = 0.5
    #   mean_reciprocal_rank = (1 + 1 + 0) / 3 = 2/3
    inputs = [
        (_q("q1", ["a"]), [_r("a#0", "a"), _r("b#0", "b"), _r("c#0", "c")]),
        (_q("q2", ["a", "b"]), [_r("a#0", "a"), _r("c#0", "c"), _r("e#0", "e")]),
        (_q("q3", ["x"]), [_r("a#0", "a"), _r("b#0", "b"), _r("c#0", "c")]),
        (_q("u1", [], answerable=False), [_r("a#0", "a")]),
        (_q("u2", [], answerable=False), [_r("b#0", "b")]),
    ]

    agg = aggregate_retrieval_metrics(inputs, k=3)

    assert agg.k == 3
    assert agg.n_answerable == 3
    assert agg.n_unanswerable == 2
    assert agg.any_hit_rate == pytest.approx(2 / 3)
    assert agg.mean_recall == pytest.approx(0.5)
    assert agg.mean_reciprocal_rank == pytest.approx(2 / 3)


def test_aggregator_per_question_layout():
    inputs = [
        (_q("q1", ["a"]), [_r("a#0", "a")]),
        (_q("q2", ["b"]), [_r("a#0", "a")]),
        (_q("u1", [], answerable=False), [_r("a#0", "a")]),
    ]
    agg = aggregate_retrieval_metrics(inputs, k=1)

    assert len(agg.per_question) == 3
    by_id = {p.question_id: p for p in agg.per_question}

    # q1: answerable, hit
    assert by_id["q1"].answerable is True
    assert by_id["q1"].any_hit_at_k is True
    assert by_id["q1"].recall_at_k == 1.0
    assert by_id["q1"].reciprocal_rank == 1.0
    assert by_id["q1"].retrieved_doc_ids == ("a",)

    # q2: answerable, miss
    assert by_id["q2"].answerable is True
    assert by_id["q2"].any_hit_at_k is False
    assert by_id["q2"].recall_at_k == 0.0
    assert by_id["q2"].reciprocal_rank == 0.0

    # u1: unanswerable — metric fields are None, retrieved_doc_ids still populated
    assert by_id["u1"].answerable is False
    assert by_id["u1"].any_hit_at_k is None
    assert by_id["u1"].recall_at_k is None
    assert by_id["u1"].reciprocal_rank is None
    assert by_id["u1"].retrieved_doc_ids == ("a",)


def test_aggregator_all_unanswerable_yields_zero_rates_without_dividing_by_zero():
    inputs = [
        (_q("u1", [], answerable=False), [_r("a#0", "a")]),
        (_q("u2", [], answerable=False), [_r("b#0", "b")]),
    ]
    agg = aggregate_retrieval_metrics(inputs, k=1)
    assert agg.n_answerable == 0
    assert agg.n_unanswerable == 2
    assert agg.any_hit_rate == 0.0
    assert agg.mean_recall == 0.0
    assert agg.mean_reciprocal_rank == 0.0
