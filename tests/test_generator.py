import pytest

from rag_eval.generator import Generator
from rag_eval.prompts import render_generator_prompt


# ---------- fast tests (no model load) ----------


def test_constructor_stores_config():
    g = Generator(
        model_name="stub-model",
        seed=7,
        max_new_tokens=50,
        model=object(),
        tokenizer=object(),
    )
    assert g.model_name == "stub-model"
    assert g.seed == 7
    assert g.max_new_tokens == 50


def test_lazy_load_skipped_when_model_injected():
    # If both model and tokenizer are provided, _ensure_loaded must not try
    # to import transformers. Passing sentinel objects proves this — a real
    # import path would fail on them.
    g = Generator(model_name="stub", model=object(), tokenizer=object())
    g._ensure_loaded()  # must not raise, must not touch transformers
    assert g._model is not None
    assert g._tokenizer is not None


# ---------- slow tests (real Qwen) ----------


@pytest.mark.slow
def test_greedy_generation_is_deterministic():
    # The core determinism guarantee: same prompt twice must yield
    # bit-identical output. Without this, faithfulness comparisons across
    # configs carry noise the harness cannot correct for.
    g = Generator(max_new_tokens=64)
    messages = render_generator_prompt(
        "What is the PTO carryover cap?",
        [("doc_02", "Employees may carry over a maximum of 5 unused PTO days.")],
    )
    out1 = g.generate(messages)
    out2 = g.generate(messages)
    assert out1 == out2, f"non-deterministic: {out1!r} vs {out2!r}"


@pytest.mark.slow
def test_generation_returns_only_new_tokens_not_the_echoed_prompt():
    g = Generator(max_new_tokens=32)
    messages = render_generator_prompt(
        "What is 2+2?",
        [("doc_math", "Basic arithmetic: 2 plus 2 equals 4.")],
    )
    out = g.generate(messages)
    # The generator must strip the prompt echo. Any assistant reply is fine;
    # what we're checking is that we do NOT see the system prompt or the
    # user's question text verbatim in the output.
    assert "You are answering questions" not in out
    assert "Context:" not in out
    assert out  # non-empty
