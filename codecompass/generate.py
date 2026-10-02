"""Answer a question with Claude, grounded on retrieved chunks, with verifiable citations.

Each chunk is sent as a separate text *document* with citations enabled, so every
claim in the answer points back to an exact span of a retrieved chunk
(file, symbol and lines). Requires ``pip install anthropic`` and an API key.
"""

from __future__ import annotations

from dataclasses import dataclass

from .search import Hit

DEFAULT_MODEL = "claude-opus-5-5"

SYSTEM = (
    "You answer questions about a source code repository using only the code excerpts provided as documents. "
    "Explain how the code works, name the functions and files involved, and quote short snippets when useful. "
    "If the excerpts do not contain the answer, say so plainly instead of guessing."
)


@dataclass
class Answer:
    text: str
    citations: list[dict]  # {"source": "path::symbol (lines a-b)", "quote": "..."}


def _document(hit: Hit) -> dict:
    c = hit.chunk
    title = f"{c.path}" + (f"::{c.symbol}" if c.symbol else "") + f" (lines {c.start_line}-{c.end_line})"
    return {
        "type": "document",
        "source": {"type": "text", "media_type": "text/plain", "data": c.text},
        "title": title,
        "citations": {"enabled": True},
    }


def answer(question: str, hits: list[Hit], model: str = DEFAULT_MODEL, client=None) -> Answer:
    if client is None:
        import anthropic

        client = anthropic.Anthropic()
    content = [_document(h) for h in hits] + [{"type": "text", "text": question}]
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM,
        messages=[{"role": "user", "content": content}],
        # Retry on another model if a safety classifier declines the request.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        return Answer("The model declined to answer this question.", [])

    parts, citations = [], []
    for block in response.content:
        if block.type != "text":
            continue
        parts.append(block.text)
        for cit in getattr(block, "citations", None) or []:
            citations.append({"source": cit.document_title, "quote": cit.cited_text})
            parts.append(f" [{len(citations)}]")
    return Answer("".join(parts), citations)
