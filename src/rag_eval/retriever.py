from dataclasses import dataclass

import numpy as np

from rag_eval.chunker import Chunk


@dataclass(frozen=True)
class Retrieved:
    chunk_id: str
    doc_id: str
    score: float


class Retriever:
    """Top-k cosine retrieval with deterministic ordering.

    Chunk embeddings are pre-normalized at construction so per-query cost is
    one normalization plus one matmul. Results are ordered by:

      1. score descending
      2. chunk_id ascending (deterministic tie-break)

    Rule 2 matters for reproducibility: with numpy's default (unstable)
    argsort, tied scores can permute across runs and every downstream metric
    computed from top-k becomes non-reproducible. Python's stable sorted() on
    the (-score, chunk_id) key locks the order.
    """

    def __init__(self, chunks: list[Chunk], embeddings: np.ndarray):
        if len(chunks) != embeddings.shape[0]:
            raise ValueError(
                f"chunk count ({len(chunks)}) does not match embedding "
                f"count ({embeddings.shape[0]})"
            )
        self.chunks = chunks
        self._chunk_ids = [c.chunk_id for c in chunks]
        self._doc_ids = [c.doc_id for c in chunks]
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        self._embeddings = embeddings / norms

    def query(self, query_embedding: np.ndarray, k: int) -> list[Retrieved]:
        if k <= 0:
            raise ValueError(f"k must be >= 1, got {k}")

        q = query_embedding / np.linalg.norm(query_embedding)
        scores = self._embeddings @ q

        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self._chunk_ids[i]),
        )
        top = order[: min(k, len(self.chunks))]
        return [
            Retrieved(
                chunk_id=self._chunk_ids[i],
                doc_id=self._doc_ids[i],
                score=float(scores[i]),
            )
            for i in top
        ]
