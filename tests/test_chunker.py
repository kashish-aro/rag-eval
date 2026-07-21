from pathlib import Path

import pytest

from rag_eval.chunker import Chunk, WholeDocChunker
from rag_eval.corpus import Document, load_corpus

CORPUS_PATH = Path(__file__).parent.parent / "corpus"


def test_chunk_is_frozen():
    c = Chunk(chunk_id="doc_01#0", doc_id="doc_01", text="x")
    with pytest.raises(Exception):
        c.text = "y"  # type: ignore[misc]


def test_empty_input_returns_empty_list():
    assert WholeDocChunker().chunk([]) == []


def test_two_docs_produce_two_chunks():
    docs = [
        Document(doc_id="a", text="alpha"),
        Document(doc_id="b", text="beta"),
    ]
    chunks = WholeDocChunker().chunk(docs)
    assert len(chunks) == 2
    assert chunks[0].text == "alpha"
    assert chunks[1].text == "beta"


def test_chunk_id_carries_doc_id():
    docs = [Document(doc_id="doc_05", text="x")]
    chunks = WholeDocChunker().chunk(docs)
    assert chunks[0].chunk_id == "doc_05#0"
    assert chunks[0].doc_id == "doc_05"


def test_chunk_id_and_doc_id_field_agree():
    # Contract for downstream chunk->doc collapse: extracting the doc_id from
    # the chunk_id prefix must always equal the explicit doc_id field.
    docs = [Document(doc_id=f"doc_{i:02d}", text=str(i)) for i in range(5)]
    chunks = WholeDocChunker().chunk(docs)
    for c in chunks:
        assert c.chunk_id.split("#")[0] == c.doc_id


def test_chunk_ids_are_unique():
    docs = [Document(doc_id=f"doc_{i:02d}", text=str(i)) for i in range(5)]
    chunks = WholeDocChunker().chunk(docs)
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_text_preserved_verbatim():
    text = "# heading\n\nline one\nline two\n\n  trailing spaces  \n"
    docs = [Document(doc_id="d", text=text)]
    chunks = WholeDocChunker().chunk(docs)
    assert chunks[0].text == text


def test_emission_order_is_deterministic():
    docs = load_corpus(CORPUS_PATH)
    a = [c.chunk_id for c in WholeDocChunker().chunk(docs)]
    b = [c.chunk_id for c in WholeDocChunker().chunk(docs)]
    assert a == b


def test_whole_corpus_one_chunk_per_doc_with_matching_doc_ids():
    # The 1:1 assertion the doc-collapse logic will rely on. When a real
    # splitting chunker replaces WholeDocChunker, this test moves / is
    # replaced, but the invariant "chunks cover exactly the corpus docs, no
    # more no less" stays.
    docs = load_corpus(CORPUS_PATH)
    chunks = WholeDocChunker().chunk(docs)

    assert len(chunks) == len(docs) == 18

    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}

    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    over = {k: v for k, v in counts.items() if v != 1}
    assert not over, f"WholeDocChunker emitted != 1 chunk for: {over}"


def test_whole_corpus_chunk_text_equals_source_doc_text():
    docs = load_corpus(CORPUS_PATH)
    chunks = WholeDocChunker().chunk(docs)
    by_id = {d.doc_id: d.text for d in docs}
    for c in chunks:
        assert c.text == by_id[c.doc_id]
