"""Shared chunk packing."""

from __future__ import annotations

from raglab.types import Chunk, Document


def pack_chunks(doc: Document, parts: list[str]) -> list[Chunk]:
    chunks: list[Chunk] = []
    index = 0
    for part in parts:
        text = part.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::c{index}",
                doc_id=doc.doc_id,
                text=text,
                index=index,
            )
        )
        index += 1
    return chunks


class Chunker:
    """Split documents into chunks. Subclasses implement ``split_text`` or ``chunk_corpus``."""

    name = "base"

    def split_text(self, text: str) -> list[str]:
        raise NotImplementedError

    def chunk_document(self, document: Document) -> list[Chunk]:
        return pack_chunks(document, self.split_text(document.full_text))

    def chunk_corpus(
        self, documents: list[Document], embedder: object | None = None
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self.chunk_document(document))
        return chunks
