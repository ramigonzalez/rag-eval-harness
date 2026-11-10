"""One chunk per document. The baseline the other chunkers are compared with."""

from __future__ import annotations

from raglab.chunking.base import Chunker
from raglab.chunking.registry import register_chunker


@register_chunker
class WholeChunker(Chunker):
    name = "whole"

    @classmethod
    def from_config(cls, options: dict) -> WholeChunker:
        if options:
            raise ValueError(f"whole chunker takes no options, got {sorted(options)}")
        return cls()

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        return [text] if text else []
