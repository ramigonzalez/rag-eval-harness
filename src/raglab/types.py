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


@dataclass(frozen=True)
class Query:
    query_id: str
    text: str


@dataclass(frozen=True)
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    queries: list[Query]
    qrels: dict[str, dict[str, int]]
