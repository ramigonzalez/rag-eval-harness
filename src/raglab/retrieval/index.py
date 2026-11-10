"""In-memory chunk index and the helpers that turn scores into hits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raglab.retrieval.bm25 import BM25Index
from raglab.textutil import tokenize
from raglab.types import Chunk, Hit


@dataclass
class Services:
    """Runtime dependencies a retriever may use. Unused fields stay None."""

    embedder: object | None = None
    reranker: object | None = None
    llm: object | None = None
    rrf_k: int = 60
    fusion_pool: int = 80
    rerank_pool: int = 30
    rag_fusion_queries: int = 3


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    embeddings: np.ndarray | None
    bm25: BM25Index | None

    def __post_init__(self) -> None:
        self.chunk_ids = [chunk.chunk_id for chunk in self.chunks]
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        if len(self.by_id) != len(self.chunks):
            raise RuntimeError("Chunk ids are not unique")


def build_search_index(
    chunks: list[Chunk],
    embedder: object | None,
    *,
    build_dense: bool,
    build_bm25: bool,
    bm25_k1: float,
    bm25_b: float,
) -> SearchIndex:
    embeddings = None
    if build_dense:
        if embedder is None:
            raise RuntimeError("dense retrieval needs an embedder")
        matrix = np.asarray(
            embedder.embed_documents([chunk.text for chunk in chunks]), dtype=np.float32
        )
        embeddings = l2_normalize(matrix)
    bm25 = None
    if build_bm25:
        bm25 = BM25Index([tokenize(chunk.text) for chunk in chunks], k1=bm25_k1, b=bm25_b)
    return SearchIndex(chunks=chunks, embeddings=embeddings, bm25=bm25)


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.maximum(norms, 1e-12)).astype(np.float32, copy=False)


def rank_indices(scores: np.ndarray, chunk_ids: list[str], k: int) -> list[int]:
    """Highest score first. Equal scores break on chunk id."""
    count = int(scores.shape[0])
    if count == 0 or k <= 0:
        return []
    take = min(k, count)
    ids = np.asarray(chunk_ids)
    order = np.lexsort((ids, -scores))
    return order[:take].tolist()


def hits_from_scores(index: SearchIndex, scores: np.ndarray, k: int) -> list[Hit]:
    hits: list[Hit] = []
    for position in rank_indices(scores, index.chunk_ids, k):
        chunk = index.chunks[position]
        hits.append(
            Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(scores[position]),
            )
        )
    return hits


def hits_from_id_scores(index: SearchIndex, ranked: list[tuple[str, float]], k: int) -> list[Hit]:
    hits: list[Hit] = []
    for chunk_id, score in ranked[:k]:
        chunk = index.by_id[chunk_id]
        hits.append(
            Hit(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                text=chunk.text,
                score=float(score),
            )
        )
    return hits
