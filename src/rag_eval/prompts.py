SYSTEM_PROMPT = """You are answering questions using only the provided context.

Rules:
- Answer only from the provided context. Do not use outside knowledge.
- If the context does not contain the answer, respond with exactly: I don't know.
- Keep answers concise and factual.
""".strip()


REFUSAL_STRING = "I don't know."


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
