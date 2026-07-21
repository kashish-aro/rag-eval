import hashlib
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_CACHE_DIR = Path(".cache/embeddings")


def _cache_key(model_name: str, text: str) -> str:
    """Content-addressed key over (model_name, text).

    Model name is part of the key so a config-comparison run swapping
    embedders cannot read the previous model's vectors from cache. A null
    byte separates the fields so (model="ab", text="cd") does not collide
    with (model="a", text="bcd").
    """
    h = hashlib.sha256()
    h.update(model_name.encode("utf-8"))
    h.update(b"\x00")
    h.update(text.encode("utf-8"))
    return h.hexdigest()


class Embedder:
    """SentenceTransformer wrapped with an on-disk vector cache.

    Cached vectors are bit-identical to freshly computed ones (.npy preserves
    dtype and shape), so a warm-cache run and a cold run produce identical
    downstream retrieval results.

    The real sentence_transformers import is deferred to first embed() so
    tests that inject a stub via model= never pay the import cost.
    """

    def __init__(
        self,
        model_name: str,
        cache_dir: Path | None = None,
        model: Any = None,
    ):
        self.model_name = model_name
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = model

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self, text: str) -> Path:
        return self.cache_dir / f"{_cache_key(self.model_name, text)}.npy"

    def _load_cached(self, text: str) -> np.ndarray | None:
        p = self._cache_path(text)
        if p.exists():
            return np.load(p)
        return None

    def _save_cached(self, text: str, vector: np.ndarray) -> None:
        np.save(self._cache_path(text), vector)

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            raise ValueError("cannot embed empty list — dimensionality unknown")

        vectors: list[np.ndarray | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []

        for i, text in enumerate(texts):
            cached = self._load_cached(text)
            if cached is not None:
                vectors[i] = cached
            else:
                miss_idx.append(i)
                miss_texts.append(text)

        if miss_texts:
            fresh = np.asarray(self._get_model().encode(miss_texts))
            if fresh.ndim == 1:
                fresh = fresh.reshape(1, -1)
            for slot, vec in zip(miss_idx, fresh):
                self._save_cached(texts[slot], vec)
                vectors[slot] = vec

        return np.stack(vectors)  # type: ignore[arg-type]
