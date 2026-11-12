"""Chunks of consecutive sentences, with a sentence overlap.

This is the retrieval unit used by sentence-window indexing. Neighboring
sentences are already inside the chunk, so the stored text is what gets scored.
"""

from __future__ import annotations

from raglab.chunking.base import Chunker
from raglab.chunking.registry import register_chunker
from raglab.textutil import split_sentences


@register_chunker
class SentenceWindowChunker(Chunker):
    name = "sentence_window"

    def __init__(self, window_sentences: int = 3, overlap_sentences: int = 1) -> None:
        if window_sentences < 1:
            raise ValueError("window_sentences must be >= 1")
        if overlap_sentences < 0 or overlap_sentences >= window_sentences:
            raise ValueError("overlap_sentences must satisfy 0 <= overlap < window")
        self.window_sentences = window_sentences
        self.overlap_sentences = overlap_sentences

    @classmethod
    def from_config(cls, options: dict) -> SentenceWindowChunker:
        unknown = set(options) - {"window_sentences", "overlap_sentences"}
        if unknown:
            raise ValueError(f"Unknown sentence_window options: {sorted(unknown)}")
        return cls(
            window_sentences=int(options.get("window_sentences", 3)),
            overlap_sentences=int(options.get("overlap_sentences", 1)),
        )

    def split_text(self, text: str) -> list[str]:
        sentences = split_sentences(text)
        if not sentences:
            return []
        if len(sentences) <= self.window_sentences:
            return [" ".join(sentences)]
        step = self.window_sentences - self.overlap_sentences
        chunks: list[str] = []
        start = 0
        while start < len(sentences):
            end = min(start + self.window_sentences, len(sentences))
            chunks.append(" ".join(sentences[start:end]))
            if end == len(sentences):
                break
            start += step
        return chunks
