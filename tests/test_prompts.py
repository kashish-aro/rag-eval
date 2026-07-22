from rag_eval.prompts import (
    JUDGE_SYSTEM_PROMPT,
    REFUSAL_STRING,
    SYSTEM_PROMPT,
    render_generator_prompt,
    render_judge_prompt,
)


def test_returns_two_messages_system_then_user():
    msgs = render_generator_prompt("q?", [("doc_a", "text")])
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_system_prompt_contains_refusal_contract():
    # The judge and rule-based scorer both parse refusals; the exact string
    # in the system prompt must match the constant so a downstream contract
    # change surfaces here.
    assert REFUSAL_STRING in SYSTEM_PROMPT
    assert "only from the provided context" in SYSTEM_PROMPT


def test_user_message_contains_the_question():
    msgs = render_generator_prompt("What is the accrual rate?", [])
    assert "What is the accrual rate?" in msgs[1]["content"]


def test_user_message_contains_each_chunk_content():
    msgs = render_generator_prompt(
        "q?",
        [("doc_02", "PTO accrues at 1.25 days per month."),
         ("doc_15", "Rate limit is 1000 rpm.")],
    )
    body = msgs[1]["content"]
    assert "PTO accrues at 1.25 days per month." in body
    assert "Rate limit is 1000 rpm." in body


def test_chunks_have_visible_doc_id_headers():
    msgs = render_generator_prompt("q?", [("doc_02", "x"), ("doc_15", "y")])
    body = msgs[1]["content"]
    assert "--- doc_02 ---" in body
    assert "--- doc_15 ---" in body


def test_multi_chunk_boundaries_are_delimited():
    # Chunks must not blur into each other; a blank line separator between
    # header blocks is enough for the model to see boundaries.
    msgs = render_generator_prompt(
        "q?", [("doc_a", "alpha"), ("doc_b", "beta")]
    )
    body = msgs[1]["content"]
    # Both chunks present in order
    idx_a = body.index("--- doc_a ---")
    idx_b = body.index("--- doc_b ---")
    assert idx_a < idx_b
    # A blank line separates them
    assert "\n\n--- doc_b ---" in body


def test_empty_context_still_renders():
    # An empty retrieval must not blow up; the system prompt handles refusal.
    msgs = render_generator_prompt("q?", [])
    assert len(msgs) == 2
    assert "q?" in msgs[1]["content"]
    # Some marker for "nothing here" so the model sees the empty context
    # explicitly rather than an ambiguous blank.
    assert "no context" in msgs[1]["content"].lower()


def test_rendering_is_deterministic():
    a = render_generator_prompt("q?", [("d", "t")])
    b = render_generator_prompt("q?", [("d", "t")])
    assert a == b


# ---------- judge prompt ----------


def test_judge_system_prompt_specifies_verdict_format():
    # The strict output contract that the parser depends on. If this text
    # changes without updating the parser (or vice versa), the whole judge
    # pipeline drifts silently.
    assert "VERDICT:" in JUDGE_SYSTEM_PROMPT
    assert "CORRECT" in JUDGE_SYSTEM_PROMPT
    assert "INCORRECT" in JUDGE_SYSTEM_PROMPT
    assert "REFUSED" in JUDGE_SYSTEM_PROMPT
    assert "uppercase" in JUDGE_SYSTEM_PROMPT.lower()


def test_judge_prompt_returns_two_messages():
    msgs = render_judge_prompt("q?", "ref", "ctx", "ans")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"


def test_judge_prompt_carries_all_four_inputs():
    msgs = render_judge_prompt(
        "How many days?", "15 days.", "Employees get 15 days.", "The answer is 15."
    )
    body = msgs[1]["content"]
    assert "How many days?" in body
    assert "15 days." in body
    assert "Employees get 15 days." in body
    assert "The answer is 15." in body


def test_judge_prompt_marks_unanswerable_when_reference_is_none():
    msgs = render_judge_prompt("q?", None, "ctx", "ans")
    body = msgs[1]["content"]
    assert "UNANSWERABLE" in body


def test_judge_prompt_deterministic():
    a = render_judge_prompt("q", "r", "c", "g")
    b = render_judge_prompt("q", "r", "c", "g")
    assert a == b
