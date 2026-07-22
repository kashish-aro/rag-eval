from pathlib import Path

import pytest

from rag_eval.chunker import (
    Chunk,
    FixedWindowChunker,
    HeadingChunker,
    WholeDocChunker,
    reconstruct_from_chunks,
)
from rag_eval.corpus import Document, load_corpus

CORPUS_PATH = Path(__file__).parent.parent / "corpus"


# ---------- Chunk dataclass ----------


def test_chunk_is_frozen():
    c = Chunk(chunk_id="doc_01#0", doc_id="doc_01", text="x")
    with pytest.raises(Exception):
        c.text = "y"  # type: ignore[misc]


# ---------- reconstruct_from_chunks ----------


def test_reconstruct_no_overlap_is_plain_concat():
    assert reconstruct_from_chunks(["ab", "cd", "ef"], overlap=0) == "abcdef"


def test_reconstruct_with_overlap_drops_prefix_of_tail_chunks():
    # Two chunks of "abcdef" with size=4, overlap=2:
    # chunk 0 = "abcd", chunk 1 = "cdef" -> reconstruct: "abcd" + "ef" = "abcdef"
    assert reconstruct_from_chunks(["abcd", "cdef"], overlap=2) == "abcdef"


def test_reconstruct_empty():
    assert reconstruct_from_chunks([], overlap=0) == ""


def test_reconstruct_negative_overlap_raises():
    with pytest.raises(ValueError):
        reconstruct_from_chunks(["a"], overlap=-1)


# ---------- WholeDocChunker specific ----------


def test_whole_doc_empty_input():
    assert WholeDocChunker().chunk([]) == []


def test_whole_doc_two_docs_produce_two_chunks():
    docs = [Document(doc_id="a", text="alpha"), Document(doc_id="b", text="beta")]
    chunks = WholeDocChunker().chunk(docs)
    assert len(chunks) == 2
    assert chunks[0].text == "alpha"
    assert chunks[1].text == "beta"


def test_whole_doc_chunk_id_zero_index():
    chunks = WholeDocChunker().chunk([Document(doc_id="doc_05", text="x")])
    assert chunks[0].chunk_id == "doc_05#0"


def test_whole_doc_corpus_produces_one_chunk_per_doc():
    docs = load_corpus(CORPUS_PATH)
    chunks = WholeDocChunker().chunk(docs)
    assert len(chunks) == len(docs) == 18
    counts: dict[str, int] = {}
    for c in chunks:
        counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
    assert all(v == 1 for v in counts.values())


def test_whole_doc_params_empty():
    assert WholeDocChunker().params() == {}


# ---------- FixedWindowChunker specific ----------


def test_fixed_window_invalid_size_raises():
    with pytest.raises(ValueError, match="size must be"):
        FixedWindowChunker(size=0)
    with pytest.raises(ValueError, match="size must be"):
        FixedWindowChunker(size=-1)


def test_fixed_window_negative_overlap_raises():
    with pytest.raises(ValueError, match="overlap must be"):
        FixedWindowChunker(size=10, overlap=-1)


def test_fixed_window_overlap_ge_size_raises():
    with pytest.raises(ValueError, match="must be < size"):
        FixedWindowChunker(size=10, overlap=10)
    with pytest.raises(ValueError, match="must be < size"):
        FixedWindowChunker(size=10, overlap=11)


def test_fixed_window_doc_shorter_than_size_is_single_chunk():
    docs = [Document(doc_id="d", text="short")]
    chunks = FixedWindowChunker(size=100).chunk(docs)
    assert len(chunks) == 1
    assert chunks[0].text == "short"


def test_fixed_window_no_overlap_produces_disjoint_chunks():
    docs = [Document(doc_id="d", text="abcdefghij")]  # len 10
    chunks = FixedWindowChunker(size=4, overlap=0).chunk(docs)
    assert [c.text for c in chunks] == ["abcd", "efgh", "ij"]


def test_fixed_window_with_overlap_reconstructs():
    text = "abcdefghij"
    docs = [Document(doc_id="d", text=text)]
    chunker = FixedWindowChunker(size=4, overlap=1)
    chunks = chunker.chunk(docs)
    # step = 3: [0:4]="abcd", [3:7]="defg", [6:10]="ghij"
    assert [c.text for c in chunks] == ["abcd", "defg", "ghij"]
    assert reconstruct_from_chunks([c.text for c in chunks], overlap=1) == text


def test_fixed_window_params():
    p = FixedWindowChunker(size=400, overlap=50).params()
    assert p == {"size": 400, "overlap": 50}


def test_fixed_window_corpus_produces_multiple_chunks_per_doc():
    docs = load_corpus(CORPUS_PATH)
    chunks = FixedWindowChunker(size=400, overlap=50).chunk(docs)
    # 18 docs of ~500-1500 chars each with size=400 must produce >18 chunks
    assert len(chunks) > 18
    # Every doc appears in the output
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}


# ---------- HeadingChunker specific ----------


def test_heading_doc_without_heading_is_single_chunk():
    docs = [Document(doc_id="d", text="just some prose no headings")]
    chunks = HeadingChunker().chunk(docs)
    assert len(chunks) == 1
    assert chunks[0].text == "just some prose no headings"


def test_heading_doc_starting_with_heading_and_two_sections():
    text = "# Title\n\ntop text\n\n## Sec A\n\naaa\n\n## Sec B\n\nbbb\n"
    chunks = HeadingChunker().chunk([Document(doc_id="d", text=text)])
    # Splits at "# Title" (0), "## Sec A", "## Sec B" -> 3 chunks
    assert len(chunks) == 3
    assert chunks[0].text.startswith("# Title")
    assert chunks[1].text.startswith("## Sec A")
    assert chunks[2].text.startswith("## Sec B")
    # Concat recovers source (no overlap)
    assert "".join(c.text for c in chunks) == text


def test_heading_prelude_before_first_heading_becomes_chunk_0():
    text = "prelude line\n\n# First\n\ncontent\n"
    chunks = HeadingChunker().chunk([Document(doc_id="d", text=text)])
    assert len(chunks) == 2
    assert chunks[0].text == "prelude line\n\n"
    assert chunks[1].text.startswith("# First")


def test_heading_corpus_produces_more_than_one_chunk_for_multi_section_docs():
    docs = load_corpus(CORPUS_PATH)
    chunks = HeadingChunker().chunk(docs)
    assert len(chunks) > 18
    assert {c.doc_id for c in chunks} == {d.doc_id for d in docs}


def test_heading_params():
    assert HeadingChunker().params() == {}


# ---------- Parametrized shared contract ----------
#
# Every chunker must uphold: id contract, uniqueness, determinism, and
# reconstruction. FixedWindowChunker takes overlap; the others use 0.

_ALL_CHUNKERS = [
    (WholeDocChunker(), 0),
    (FixedWindowChunker(size=400, overlap=0), 0),
    (FixedWindowChunker(size=400, overlap=50), 50),
    (HeadingChunker(), 0),
]


@pytest.mark.parametrize("chunker,overlap", _ALL_CHUNKERS)
def test_chunk_id_prefix_equals_doc_id(chunker, overlap):
    docs = load_corpus(CORPUS_PATH)
    for c in chunker.chunk(docs):
        assert c.chunk_id.split("#")[0] == c.doc_id


@pytest.mark.parametrize("chunker,overlap", _ALL_CHUNKERS)
def test_chunk_ids_unique_corpus_wide(chunker, overlap):
    docs = load_corpus(CORPUS_PATH)
    ids = [c.chunk_id for c in chunker.chunk(docs)]
    assert len(set(ids)) == len(ids)


@pytest.mark.parametrize("chunker,overlap", _ALL_CHUNKERS)
def test_emission_deterministic(chunker, overlap):
    docs = load_corpus(CORPUS_PATH)
    a = [c.chunk_id for c in chunker.chunk(docs)]
    b = [c.chunk_id for c in chunker.chunk(docs)]
    assert a == b


@pytest.mark.parametrize("chunker,overlap", _ALL_CHUNKERS)
def test_reconstruction_invariant(chunker, overlap):
    # For each doc, concatenating (or reconstructing with overlap) the
    # chunker's output for that doc must yield the original text.
    docs = load_corpus(CORPUS_PATH)
    chunks = chunker.chunk(docs)
    by_doc: dict[str, list[str]] = {}
    for c in chunks:
        by_doc.setdefault(c.doc_id, []).append(c.text)
    for d in docs:
        assert (
            reconstruct_from_chunks(by_doc[d.doc_id], overlap=overlap) == d.text
        ), f"reconstruction failed for {d.doc_id} under {type(chunker).__name__}"


@pytest.mark.parametrize("chunker,overlap", _ALL_CHUNKERS)
def test_every_corpus_doc_appears(chunker, overlap):
    docs = load_corpus(CORPUS_PATH)
    doc_ids_out = {c.doc_id for c in chunker.chunk(docs)}
    assert doc_ids_out == {d.doc_id for d in docs}
