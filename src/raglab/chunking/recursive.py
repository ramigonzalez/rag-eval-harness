"""Split on paragraph and sentence boundaries, then pack up to a word budget.

This is the same idea as a recursive character splitter: try a coarse separator
first, and fall back to a finer one only when a piece is still too long.
"""

from __future__ import annotations

from raglab.chunking.base import Chunker
from raglab.chunking.registry import register_chunker
from raglab.textutil import word_count

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " "]


def _split_keep(text: str, separator: str) -> list[str]:
    if separator not in text:
        return [text]
    parts = text.split(separator)
    pieces: list[str] = []
    last = len(parts) - 1
    for index, part in enumerate(parts):
        piece = part + separator if index < last else part
        if piece.strip():
            pieces.append(piece)
    return pieces or [text]


def _join(atoms: list[str]) -> str:
    return "".join(atoms).strip()


def _overlap_tail(atoms: list[str], overlap: int) -> list[str]:
    if overlap <= 0 or not atoms:
        return []
    kept: list[str] = []
    total = 0
    for atom in reversed(atoms):
        words = word_count(atom)
        if words == 0:
            continue
        if total + words > overlap:
            break
        kept.append(atom)
        total += words
    kept.reverse()
    return kept


@register_chunker
class RecursiveChunker(Chunker):
    name = "recursive"

    def __init__(
        self,
        chunk_size: int = 100,
        overlap: int = 25,
        separators: list[str] | None = None,
    ) -> None:
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separators = list(separators) if separators is not None else list(DEFAULT_SEPARATORS)

    @classmethod
    def from_config(cls, options: dict) -> RecursiveChunker:
        unknown = set(options) - {"chunk_size", "overlap"}
        if unknown:
            raise ValueError(f"Unknown recursive chunker options: {sorted(unknown)}")
        return cls(
            chunk_size=int(options.get("chunk_size", 100)),
            overlap=int(options.get("overlap", 25)),
        )

    def split_text(self, text: str) -> list[str]:
        atoms = self._atoms(text.strip(), self.separators)
        if not atoms:
            return []
        chunks: list[str] = []
        current: list[str] = []
        for atom in atoms:
            if not current and word_count(atom) > self.chunk_size:
                chunks.append(atom.strip())
                continue
            if current and word_count(_join(current + [atom])) > self.chunk_size:
                chunks.append(_join(current))
                current = _overlap_tail(current, self.overlap)
                if current and word_count(_join(current + [atom])) > self.chunk_size:
                    current = []
            current.append(atom)
        if current:
            chunks.append(_join(current))
        return [chunk for chunk in chunks if chunk.strip()]

    def _atoms(self, text: str, separators: list[str]) -> list[str]:
        text = text.strip()
        if not text:
            return []
        if word_count(text) <= self.chunk_size:
            return [text]
        separator = next((item for item in separators if item and item in text), None)
        if separator is None:
            return self._hard_words(text)
        rest = separators[separators.index(separator) + 1 :]
        atoms: list[str] = []
        for piece in _split_keep(text, separator):
            if word_count(piece) <= self.chunk_size:
                atoms.append(piece)
            else:
                atoms.extend(self._atoms(piece.strip(), rest))
        return [atom for atom in atoms if atom.strip()]

    def _hard_words(self, text: str) -> list[str]:
        words = text.split()
        if not words:
            return []
        size = self.chunk_size
        return [" ".join(words[start : start + size]) for start in range(0, len(words), size)]
