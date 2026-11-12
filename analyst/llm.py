"""Optional LLM layer. The app works fully without it (rule-based narrative)."""

from __future__ import annotations

import os

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")

SYSTEM_PROMPT = (
    "You are a senior investment research analyst writing a concise note for a "
    "retail investor. Use ONLY the numbers and findings provided. Do not invent "
    "prices, dates, news or statistics. Be balanced: state the key bull case, the "
    "key bear case, and the main risks. Keep it under 200 words, plain prose with "
    "no headings. End with one sentence noting this is educational and not "
    "financial advice."
)


def generate_narrative(prompt: str, api_key: str, model: str | None = None,
                       max_tokens: int = 700) -> str:
    """Call the Anthropic API. Raises on any failure so callers can fall back."""
    import anthropic  # lazy import: optional dependency

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model or DEFAULT_MODEL,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    parts = [b.text for b in resp.content if getattr(b, "type", "") == "text"]
    text = "".join(parts).strip()
    if not text:
        raise RuntimeError("LLM returned an empty response")
    return text
