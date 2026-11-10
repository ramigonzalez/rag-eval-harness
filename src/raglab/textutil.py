"""Small text helpers shared by chunkers and BM25."""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens. Punctuation is dropped so BM25 stays stable."""
    return _TOKEN.findall(text.lower())


def word_count(text: str) -> int:
    return len(text.split())


def split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE.split(text) if part.strip()]
    return parts or [text]
