"""Fixed word windows with overlap. Cuts wherever the window ends."""

from __future__ import annotations

from raglab.chunking.base import Chunker
from raglab.chunking.registry import register_chunker


@register_chunker
class FixedChunker(Chunker):
    name = "fixed"

    def __init__(self, chunk_size: int = 100, overlap: int = 25) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    @classmethod
    def from_config(cls, options: dict) -> FixedChunker:
        unknown = set(options) - {"chunk_size", "overlap"}
        if unknown:
            raise ValueError(f"Unknown fixed chunker options: {sorted(unknown)}")
        return cls(
            chunk_size=int(options.get("chunk_size", 100)),
            overlap=int(options.get("overlap", 25)),
        )

    def split_text(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        start = 0
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step
        return chunks
