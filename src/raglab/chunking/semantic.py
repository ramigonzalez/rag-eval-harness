"""Group neighboring sentences until the topic shift is unusually large.

Adjacent sentences are embedded. A boundary is cut when its cosine distance is
above ``breakpoint_percentile`` of the distances in that document (the usual
semantic-chunking rule). Groups that are still longer than ``max_chunk_words``
are handed to the recursive splitter. Very short groups are merged forward.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from raglab.chunking.base import Chunker, pack_chunks
from raglab.chunking.recursive import RecursiveChunker
from raglab.chunking.registry import register_chunker
from raglab.textutil import split_sentences, word_count
from raglab.types import Chunk, Document

logger = logging.getLogger("raglab")


def _adjacent_distances(vectors: np.ndarray) -> np.ndarray:
    if len(vectors) < 2:
        return np.zeros(0, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / np.maximum(norms, 1e-12)
    similarity = np.sum(unit[:-1] * unit[1:], axis=1)
    return 1.0 - similarity


@register_chunker
class SemanticChunker(Chunker):
    name = "semantic"

    def __init__(
        self,
        breakpoint_percentile: float = 95,
        max_chunk_words: int = 220,
        min_chunk_words: int = 40,
        embedder: Any | None = None,
    ) -> None:
        if not 0 < breakpoint_percentile < 100:
            raise ValueError("breakpoint_percentile must be between 0 and 100")
        if max_chunk_words < 1:
            raise ValueError("max_chunk_words must be >= 1")
        if min_chunk_words < 1 or min_chunk_words > max_chunk_words:
            raise ValueError("min_chunk_words must satisfy 1 <= min <= max")
        self.breakpoint_percentile = float(breakpoint_percentile)
        self.max_chunk_words = int(max_chunk_words)
        self.min_chunk_words = int(min_chunk_words)
        self.embedder = embedder

    @classmethod
    def from_config(cls, options: dict) -> SemanticChunker:
        known = {"breakpoint_percentile", "max_chunk_words", "min_chunk_words"}
        unknown = set(options) - known
        if unknown:
            raise ValueError(f"Unknown semantic chunker options: {sorted(unknown)}")
        return cls(
            breakpoint_percentile=float(options.get("breakpoint_percentile", 95)),
            max_chunk_words=int(options.get("max_chunk_words", 220)),
            min_chunk_words=int(options.get("min_chunk_words", 40)),
        )

    def split_text(self, text: str) -> list[str]:
        raise RuntimeError("semantic chunking embeds the corpus; call chunk_corpus")

    def chunk_document(self, document: Document) -> list[Chunk]:
        return self.chunk_corpus([document], self.embedder)

    def chunk_corpus(self, documents: list[Document], embedder: Any | None = None) -> list[Chunk]:
        embedder = embedder or self.embedder
        if embedder is None:
            raise ValueError("semantic chunking needs an embedder")
        sentences_per_doc = [split_sentences(document.full_text) for document in documents]
        flat = [sentence for sentences in sentences_per_doc for sentence in sentences]
        if flat:
            logger.info("Semantic chunking is embedding %s sentences", f"{len(flat):,}")
            vectors = np.asarray(embedder.embed_documents(flat), dtype=np.float32)
        else:
            vectors = np.zeros((0, 1), dtype=np.float32)
        overflow = RecursiveChunker(chunk_size=self.max_chunk_words, overlap=0)
        chunks: list[Chunk] = []
        cursor = 0
        for document, sentences in zip(documents, sentences_per_doc, strict=True):
            count = len(sentences)
            doc_vectors = vectors[cursor : cursor + count]
            cursor += count
            parts: list[str] = []
            for group in self._groups(sentences, doc_vectors):
                if word_count(group) > self.max_chunk_words:
                    parts.extend(overflow.split_text(group))
                else:
                    parts.append(group)
            chunks.extend(pack_chunks(document, parts))
        return chunks

    def _groups(self, sentences: list[str], vectors: np.ndarray) -> list[str]:
        if not sentences:
            return []
        if len(sentences) == 1 or len(vectors) != len(sentences):
            return self._merge_small([" ".join(sentences)])
        distances = _adjacent_distances(vectors)
        # Fewer than three boundaries is not enough to estimate a percentile.
        if len(distances) < 3:
            return self._merge_small([" ".join(sentences)])
        threshold = float(np.percentile(distances, self.breakpoint_percentile))
        groups: list[list[str]] = [[sentences[0]]]
        for index, distance in enumerate(distances):
            if float(distance) > threshold:
                groups.append([sentences[index + 1]])
            else:
                groups[-1].append(sentences[index + 1])
        return self._merge_small([" ".join(group) for group in groups])

    def _merge_small(self, groups: list[str]) -> list[str]:
        if not groups:
            return []
        merged: list[str] = []
        buffer = groups[0]
        for group in groups[1:]:
            combined = word_count(buffer) + word_count(group)
            if word_count(buffer) < self.min_chunk_words and combined <= self.max_chunk_words:
                buffer = f"{buffer} {group}".strip()
            else:
                merged.append(buffer)
                buffer = group
        merged.append(buffer)
        if len(merged) >= 2 and word_count(merged[-1]) < self.min_chunk_words:
            combined = word_count(merged[-2]) + word_count(merged[-1])
            if combined <= self.max_chunk_words:
                merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
                merged.pop()
        return merged
