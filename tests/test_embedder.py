import hashlib

import numpy as np
import pytest

from rag_eval.embedder import Embedder, _cache_key


class StubModel:
    """Deterministic hash-based encoder used to test the cache without ST.

    Vector for a text is derived from sha256(tag + text). `tag` differentiates
    two stubs so tests can verify the cache returns the *original* model's
    vectors on a subsequent load, not the new stub's.
    """

    def __init__(self, dim: int = 8, tag: str = "stub"):
        self.dim = dim
        self.tag = tag
        self.calls: list[list[str]] = []

    def encode(self, texts):
        self.calls.append(list(texts))
        vectors = []
        for t in texts:
            # Seed derived from a hash so vectors are deterministic per-text,
            # and RNG supports arbitrary dim (sha256 alone caps at 8 float32s).
            digest = hashlib.sha256((self.tag + t).encode("utf-8")).digest()
            seed = int.from_bytes(digest[:4], "big")
            rng = np.random.RandomState(seed)
            vectors.append(rng.standard_normal(self.dim).astype(np.float32))
        return np.stack(vectors)


# ---------- _cache_key ----------


def test_cache_key_differs_across_models_for_same_text():
    assert _cache_key("model-a", "the same text") != _cache_key(
        "model-b", "the same text"
    )


def test_cache_key_stable_for_same_inputs():
    assert _cache_key("m", "t") == _cache_key("m", "t")


def test_cache_key_no_separator_collision():
    # A naive concat "model+text" would let (ab, cd) collide with (a, bcd).
    # The null-byte separator prevents that.
    assert _cache_key("ab", "cd") != _cache_key("a", "bcd")


# ---------- cache correctness with stub ----------


def test_same_text_two_models_creates_two_cache_entries(tmp_path):
    Embedder("model-a", cache_dir=tmp_path, model=StubModel(tag="a")).embed(["hello"])
    Embedder("model-b", cache_dir=tmp_path, model=StubModel(tag="b")).embed(["hello"])

    files = sorted(tmp_path.glob("*.npy"))
    assert len(files) == 2, [f.name for f in files]


def test_cache_hit_skips_model(tmp_path):
    stub = StubModel()
    e = Embedder("m", cache_dir=tmp_path, model=stub)

    e.embed(["hello"])
    assert stub.calls == [["hello"]]

    e.embed(["hello"])
    assert stub.calls == [["hello"]], "cache hit should not invoke the model"


def test_cached_vector_bit_identical_to_fresh(tmp_path):
    e = Embedder("m", cache_dir=tmp_path, model=StubModel())

    fresh = e.embed(["some text"])
    cached = e.embed(["some text"])

    assert np.array_equal(fresh, cached)
    assert fresh.dtype == cached.dtype


def test_mixed_hit_and_miss_computes_only_misses(tmp_path):
    stub = StubModel()
    e = Embedder("m", cache_dir=tmp_path, model=stub)

    e.embed(["first"])
    stub.calls.clear()

    result = e.embed(["first", "second"])

    assert stub.calls == [["second"]], "only 'second' should hit the model"
    assert result.shape == (2, stub.dim)


def test_output_order_matches_input_order(tmp_path):
    e = Embedder("m", cache_dir=tmp_path, model=StubModel(tag="ord"))
    e.embed(["b"])  # pre-cache one entry out of order

    result = e.embed(["a", "b", "c"])

    reference = StubModel(tag="ord").encode(["a", "b", "c"])
    assert np.array_equal(result, reference)


def test_new_embedder_reads_previous_cache(tmp_path):
    # Cache must survive a process boundary in effect: a new Embedder instance
    # over the same cache_dir returns the ORIGINAL model's vectors, not the
    # new stub's. This is the property that keeps re-runs reproducible.
    Embedder("m", cache_dir=tmp_path, model=StubModel(tag="v1")).embed(["persistent"])

    stub_v2 = StubModel(tag="v2")
    e_2 = Embedder("m", cache_dir=tmp_path, model=stub_v2)
    result = e_2.embed(["persistent"])

    assert stub_v2.calls == [], "v2 should not have been invoked"
    v1_reference = StubModel(tag="v1").encode(["persistent"])
    assert np.array_equal(result, v1_reference)


def test_empty_list_raises(tmp_path):
    e = Embedder("m", cache_dir=tmp_path, model=StubModel())
    with pytest.raises(ValueError, match="empty list"):
        e.embed([])


def test_cache_dir_created(tmp_path):
    nested = tmp_path / "a" / "b" / "cache"
    Embedder("m", cache_dir=nested, model=StubModel())
    assert nested.is_dir()


def test_output_shape_n_by_d(tmp_path):
    e = Embedder("m", cache_dir=tmp_path, model=StubModel(dim=16))
    result = e.embed(["one", "two", "three"])
    assert result.shape == (3, 16)


# ---------- slow: real model ----------


@pytest.mark.slow
def test_real_model_smoke_and_determinism(tmp_path):
    e = Embedder(
        "sentence-transformers/all-MiniLM-L6-v2",
        cache_dir=tmp_path,
    )
    v1 = e.embed(["hello world", "goodbye world"])
    assert v1.shape == (2, 384)

    v2 = e.embed(["hello world", "goodbye world"])
    assert np.array_equal(v1, v2), "cache must return bit-identical vectors"
