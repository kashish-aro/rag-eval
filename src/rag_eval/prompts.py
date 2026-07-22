SYSTEM_PROMPT = """You are answering questions using only the provided context.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the context does not contain the answer, respond with exactly: I don't know.
- Keep answers concise and factual.
""".strip()


REFUSAL_STRING = "I don't know."


JUDGE_SYSTEM_PROMPT = """You are a strict judge deciding whether a generated answer is correct.

You will be given a question, a reference answer (or the marker UNANSWERABLE), the retrieved context the answerer had access to, and the generated answer. Choose exactly one verdict:

- CORRECT: the generated answer matches the reference answer's facts, or, for an unanswerable question, correctly refuses (says "I don't know").
- INCORRECT: the generated answer contains wrong facts, is missing key facts from the reference, or answers a question that has no answer.
- REFUSED: the answer says "I don't know" for a question that has a reference answer (the model gave up on something it should have answered).

You may reason briefly first, but you MUST end your response with EXACTLY this line and nothing after it:

VERDICT: <one of CORRECT, INCORRECT, REFUSED>

Do not write anything after the VERDICT line. Use uppercase for the verdict token.
""".strip()


def render_judge_prompt(
    question: str,
    reference: str | None,
    context: str,
    generated: str,
) -> list[dict]:
    """Build the Qwen-style message list for the LLM-as-judge.

    reference=None marks an unanswerable question; the user content signals
    this with an UNANSWERABLE marker so the judge knows the reference
    behavior is refusal, not a matching answer.
    """
    ref_text = (
        reference
        if reference is not None
        else "UNANSWERABLE (the correct behavior is to refuse with 'I don't know.')"
    )
    user_content = (
        f"Question: {question}\n\n"
        f"Reference answer: {ref_text}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Generated answer:\n{generated}\n\n"
        f"What is your verdict?"
    )
    return [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def render_generator_prompt(
    question: str,
    context_chunks: list[tuple[str, str]],
) -> list[dict]:
    """Build the Qwen-style message list for the generator.

    context_chunks is a list of (doc_id, text) in retrieval rank order. Each
    chunk is wrapped in a visible header so the model sees explicit
    boundaries and could in principle cite its source. Empty context still
    renders — the system prompt handles refusal.

    Returns a list of message dicts suitable for
    tokenizer.apply_chat_template(); we do not build the raw prompt string
    here because Qwen's chat template varies with model version, and going
    through the tokenizer is the reliable path.
    """
    if context_chunks:
        context = "\n\n".join(
            f"--- {doc_id} ---\n{text}" for doc_id, text in context_chunks
        )
    else:
        context = "(no context provided)"

    user_content = f"Context:\n\n{context}\n\nQuestion: {question}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
