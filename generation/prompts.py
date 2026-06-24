def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Build a prompt that includes retrieved chunks as context,
    with clear citation markers.
    """
    context_blocks = []
    for i, chunk in enumerate(chunks):
        context_blocks.append(
            f"[Source {i+1} | Company: {chunk['company'].upper()} | "
            f"Section: {chunk['section']}]\n{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""You are a financial analyst assistant. Answer the question below 
using ONLY the provided source excerpts from SEC 10-K filings.

For every claim you make, cite the source using its number in brackets, 
like [Source 1] or [Source 2]. If multiple sources support a claim, 
cite all of them.

If the sources do not contain enough information to answer the question, 
say so clearly rather than speculating.

---

SOURCES:
{context}

---

QUESTION: {query}

ANSWER:"""

    return prompt