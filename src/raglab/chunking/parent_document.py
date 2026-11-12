"""Small child chunks that still point at the parent document.

Retrieval scores the child text. The chunk's ``doc_id`` is the parent, which is
what document-level metrics use and what a generator can expand back to.
"""

from __future__ import annotations

from raglab.chunking.base import Chunker
from raglab.chunking.fixed import FixedChunker
from raglab.chunking.registry import register_chunker


@register_chunker
class ParentDocumentChunker(Chunker):
    name = "parent_document"

    def __init__(self, chunk_size: int = 40, overlap: int = 10) -> None:
        self.child = FixedChunker(chunk_size=chunk_size, overlap=overlap)

    @classmethod
    def from_config(cls, options: dict) -> ParentDocumentChunker:
        unknown = set(options) - {"chunk_size", "overlap"}
        if unknown:
            raise ValueError(f"Unknown parent_document options: {sorted(unknown)}")
        return cls(
            chunk_size=int(options.get("chunk_size", 40)),
            overlap=int(options.get("overlap", 10)),
        )

    def split_text(self, text: str) -> list[str]:
        return self.child.split_text(text)
