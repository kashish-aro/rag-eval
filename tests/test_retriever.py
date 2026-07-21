import numpy as np
import pytest

from rag_eval.chunker import Chunk
from rag_eval.retriever import Retrieved, Retriever


def _chunk(chunk_id: str, doc_id: str = "doc_x", text: str = "t") -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, text=text)


# ---------- dataclass and construction ----------


def test_retrieved_is_frozen():
    r = Retrieved(chunk_id="a#0", doc_id="a", score=1.0)
    with pytest.raises(Exception):
        r.score = 0.0  # type: ignore[misc]


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="does not match"):
        Retriever(
            [_chunk("a#0"), _chunk("b#0")],
            np.array([[1.0, 0.0]], dtype=np.float64),
        )


def test_k_must_be_positive():
    r = Retriever([_chunk("a#0")], np.array([[1.0, 0.0]], dtype=np.float64))
    with pytest.raises(ValueError, match="k must be"):
        r.query(np.array([1.0, 0.0]), k=0)
    with pytest.raises(ValueError, match="k must be"):
        r.query(np.array([1.0, 0.0]), k=-1)


def test_k_larger_than_corpus_returns_all():
    r = Retriever(
        [_chunk("a#0"), _chunk("b#0")],
        np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
    )
    result = r.query(np.array([1.0, 0.0]), k=99)
    assert len(result) == 2


# ---------- hand-worked cosine ----------


def test_cosine_matches_hand_worked_example():
    # Deliberately un-normalized chunks so this also proves the
    # normalization path works. By hand:
    #   [1, 0] / ||.|| = [1, 0]
    #   [0, 1] / ||.|| = [0, 1]
    #   [1, 1] / ||.|| = [1/sqrt(2), 1/sqrt(2)]
    # Query [1, 0] normalizes to [1, 0]. Cosines:
    #   a: 1*1 + 0*0                     = 1.0
    #   b: 1*0 + 0*1                     = 0.0
    #   c: 1*(1/sqrt(2)) + 0*(1/sqrt(2)) = 1/sqrt(2)  ~ 0.70710678
    chunks = [_chunk("a#0"), _chunk("b#0"), _chunk("c#0")]
    embeddings = np.array(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)
    result = r.query(np.array([1.0, 0.0]), k=3)

    assert [x.chunk_id for x in result] == ["a#0", "c#0", "b#0"]
    assert result[0].score == pytest.approx(1.0)
    assert result[1].score == pytest.approx(1.0 / np.sqrt(2))
    assert result[2].score == pytest.approx(0.0)


def test_orthogonal_query_scores_zero():
    r = Retriever(
        [_chunk("a#0")],
        np.array([[1.0, 0.0]], dtype=np.float64),
    )
    result = r.query(np.array([0.0, 1.0]), k=1)
    assert result[0].score == pytest.approx(0.0)


def test_scores_are_descending():
    chunks = [_chunk(f"c_{i}#0") for i in range(5)]
    embeddings = np.array(
        [[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-1.0, 0.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)
    result = r.query(np.array([1.0, 0.0]), k=5)
    scores = [x.score for x in result]
    assert scores == sorted(scores, reverse=True)


# ---------- tie-breaking ----------


def test_ties_broken_by_chunk_id_ascending():
    # Three chunks with identical embeddings all score 1.0 against [1, 0].
    # Ids deliberately not in sorted order — result must be sorted by id.
    chunks = [_chunk("z#0"), _chunk("a#0"), _chunk("m#0")]
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)
    result = r.query(np.array([1.0, 0.0]), k=3)

    assert [x.chunk_id for x in result] == ["a#0", "m#0", "z#0"]


def test_tie_break_is_stable_across_repeated_calls():
    # The property the user flagged: same query, same retriever, same order,
    # every time. Without deterministic tie-break, downstream metrics drift.
    chunks = [_chunk("z#0"), _chunk("a#0"), _chunk("m#0")]
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)

    call_1 = [x.chunk_id for x in r.query(np.array([1.0, 0.0]), k=3)]
    call_2 = [x.chunk_id for x in r.query(np.array([1.0, 0.0]), k=3)]
    call_3 = [x.chunk_id for x in r.query(np.array([1.0, 0.0]), k=3)]

    assert call_1 == call_2 == call_3 == ["a#0", "m#0", "z#0"]


def test_partial_tie_breaks_only_within_tied_group():
    # a and z tied at 1.0, m at 0.0. Top pair broken by chunk_id, m last.
    chunks = [_chunk("z#0"), _chunk("a#0"), _chunk("m#0")]
    embeddings = np.array(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)
    result = r.query(np.array([1.0, 0.0]), k=3)

    assert [x.chunk_id for x in result] == ["a#0", "z#0", "m#0"]


# ---------- output shape ----------


def test_doc_id_preserved_in_output():
    r = Retriever(
        [_chunk("chunk_alpha#0", doc_id="doc_alpha")],
        np.array([[1.0, 0.0]], dtype=np.float64),
    )
    result = r.query(np.array([1.0, 0.0]), k=1)
    assert result[0].doc_id == "doc_alpha"


def test_k_equals_1_returns_single_top():
    chunks = [_chunk("a#0"), _chunk("b#0"), _chunk("c#0")]
    embeddings = np.array(
        [[0.5, 0.5], [1.0, 0.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    r = Retriever(chunks, embeddings)
    result = r.query(np.array([1.0, 0.0]), k=1)
    assert len(result) == 1
    assert result[0].chunk_id == "b#0"
