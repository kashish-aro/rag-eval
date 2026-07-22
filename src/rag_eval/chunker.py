import re
from dataclasses import dataclass

from rag_eval.corpus import Document


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text with an unambiguous back-pointer to its doc.

    chunk_id format: "{doc_id}#{i}" where i disambiguates multiple chunks from
    the same doc. Downstream retrieval metrics score at the document level,
    which requires collapsing chunk-hits to doc-hits; carrying doc_id in the
    chunk_id (in addition to the field) gives that collapse a stable invariant
    that survives any chunker swap.
    """

    chunk_id: str
    doc_id: str
    text: str


# Chunker interface convention (duck-typed, no ABC):
#
#   .chunk(documents) -> list[Chunk]
#   .params()        -> dict          # chunker-specific config for the run manifest
#
# Chunkers must uphold:
#   1. chunk.doc_id == chunk.chunk_id.split("#")[0]
#   2. chunk_id values are unique across the corpus
#   3. reconstruction invariant (see reconstruct_from_chunks below)
#   4. emission order is deterministic across calls


def reconstruct_from_chunks(chunk_texts: list[str], overlap: int = 0) -> str:
    """Rebuild source text from same-doc chunks emitted by a chunker.

    Generalizes the concatenation invariant to handle overlap: chunks[0] is
    kept whole; each subsequent chunk contributes only its non-overlapping
    tail. For overlap=0 this reduces to plain concatenation. This is the
    form all chunkers must satisfy against their input document text.
    """
    if not chunk_texts:
        return ""
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    result = chunk_texts[0]
    for c in chunk_texts[1:]:
        result += c[overlap:]
    return result


class WholeDocChunker:
    """Emits one chunk per document. Baseline chunker for the eval harness.

    Reconstruction invariant holds trivially with overlap=0 (single chunk
    per doc equals the whole doc).
    """

    def params(self) -> dict:
        return {}

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{d.doc_id}#0", doc_id=d.doc_id, text=d.text)
            for d in documents
        ]


class FixedWindowChunker:
    """Fixed-size character windows with configurable overlap.

    step = size - overlap. Chunks cover [0, size), [step, step+size), ... up to
    the end of the doc, with the last chunk possibly shorter. Reconstruction
    with the same overlap value recovers the source text exactly. The strict
    concatenation invariant is the overlap=0 special case; positive overlap
    duplicates text between adjacent chunks by definition, so we test against
    reconstruct_from_chunks(..., overlap) instead of raw concatenation.
    """

    def __init__(self, size: int, overlap: int = 0):
        if size <= 0:
            raise ValueError(f"size must be >= 1, got {size}")
        if overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {overlap}")
        if overlap >= size:
            raise ValueError(
                f"overlap ({overlap}) must be < size ({size}); otherwise the "
                "window never advances"
            )
        self.size = size
        self.overlap = overlap

    def params(self) -> dict:
        return {"size": self.size, "overlap": self.overlap}

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        for d in documents:
            text = d.text
            if not text:
                continue
            i = 0
            start = 0
            while start < len(text):
                end = min(start + self.size, len(text))
                chunks.append(
                    Chunk(
                        chunk_id=f"{d.doc_id}#{i}",
                        doc_id=d.doc_id,
                        text=text[start:end],
                    )
                )
                i += 1
                if end == len(text):
                    break
                start += step
        return chunks


_HEADING_RE = re.compile(r"^#{1,6} ", re.MULTILINE)


class HeadingChunker:
    """Splits a markdown document at heading boundaries (levels 1-6).

    Each chunk starts at a heading line and extends to the next heading
    (or end of doc). Text before the first heading, if any, becomes chunk 0.
    A doc with no headings is emitted as a single chunk. Concatenation
    recovers the source exactly (overlap=0, no discard).
    """

    def params(self) -> dict:
        return {}

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for d in documents:
            text = d.text
            starts = [m.start() for m in _HEADING_RE.finditer(text)]
            if not starts or starts[0] != 0:
                starts = [0] + starts
            boundaries = starts + [len(text)]
            for i, (s, e) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                chunks.append(
                    Chunk(
                        chunk_id=f"{d.doc_id}#{i}",
                        doc_id=d.doc_id,
                        text=text[s:e],
                    )
                )
        return chunks
