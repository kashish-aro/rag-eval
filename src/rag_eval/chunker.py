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


class WholeDocChunker:
    """Emits one chunk per document. Baseline chunker for the eval harness.

    Any future chunker (sentence, sliding-window, heading-split) must uphold
    the same contract so downstream code does not need to change:
      - chunk.doc_id == chunk.chunk_id.split("#")[0]
      - chunk_id values are unique across the corpus
      - concatenating same-doc chunks in emitted order recovers the source text
      - emission order is deterministic across calls
    """

    def chunk(self, documents: list[Document]) -> list[Chunk]:
        return [
            Chunk(chunk_id=f"{d.doc_id}#0", doc_id=d.doc_id, text=d.text)
            for d in documents
        ]
