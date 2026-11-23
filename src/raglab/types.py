"""Shared records passed between chunkers, retrievers, and metrics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str

    @property
    def full_text(self) -> str:
        title = self.title.strip()
        body = self.text.strip()
        if title and body:
            return f"{title}\n\n{body}"
        return title or body


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    index: int
    source_id: str = ""
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class GoldLabel:
    """One acceptable evidence location for a query.

    ``page`` is the 1-based page in the source. ``chunk_id`` matches a chunk
    directly when the corpus is already chunked. Both may be empty when the
    label is only a free-text id stored in ``text_id``.
    """

    source_id: str = ""
    page: int | None = None
    chunk_id: str = ""
    text_id: str = ""


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str
    gold: tuple[GoldLabel, ...] = ()
    answerable: bool = True


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float
    source_id: str = ""
    page_start: int | None = None
    page_end: int | None = None


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
