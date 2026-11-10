"""Built-in retrievers. Each one reads a ``SearchIndex`` and returns ranked chunks.

HyDE and RAG-Fusion call an OpenAI-compatible chat model. The runner only
constructs them when an API key is set.
"""

from __future__ import annotations

import logging

import numpy as np

from raglab.llm import parse_json_object
from raglab.retrieval.fusion import reciprocal_rank_fusion
from raglab.retrieval.index import (
    SearchIndex,
    Services,
    hits_from_id_scores,
    hits_from_scores,
)
from raglab.retrieval.registry import register_retriever
from raglab.textutil import tokenize
from raglab.types import Hit

logger = logging.getLogger("raglab")

HYDE_PROMPT = """Task: hypothetical passage.

Write one short paragraph that would answer the question.
Do not mention that the paragraph is hypothetical. Stay under 120 words.

Question: {query}
"""

FUSION_PROMPT = """Task: search queries.

Generate {n} different web-search queries that would find documents relevant
to the question. Return JSON only, with this shape:
{{"queries": ["...", "..."]}}

Question: {query}
"""


class Retriever:
    name = "base"
    needs_embeddings = False
    needs_bm25 = False
    needs_llm = False
    needs_reranker = False
    score_type = "score"
    fallbacks = 0

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        raise NotImplementedError

    def retrieve_all(
        self, index: SearchIndex, queries: list[str], k: int, services: Services
    ) -> list[list[Hit]]:
        return [self.retrieve(index, query, k, services) for query in queries]


def _require_embeddings(index: SearchIndex) -> np.ndarray:
    if index.embeddings is None:
        raise RuntimeError("this retriever needs a dense index")
    return index.embeddings


def _require_bm25(index: SearchIndex):
    if index.bm25 is None:
        raise RuntimeError("this retriever needs a BM25 index")
    return index.bm25


@register_retriever
class BM25Retriever(Retriever):
    name = "bm25"
    needs_bm25 = True
    score_type = "bm25"

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        scores = _require_bm25(index).score_all(tokenize(query))
        return hits_from_scores(index, scores, k)

    def retrieve_all(
        self, index: SearchIndex, queries: list[str], k: int, services: Services
    ) -> list[list[Hit]]:
        bm25 = _require_bm25(index)
        return [hits_from_scores(index, bm25.score_all(tokenize(query)), k) for query in queries]


@register_retriever
class DenseRetriever(Retriever):
    name = "dense"
    needs_embeddings = True
    score_type = "cosine"

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        return self.retrieve_all(index, [query], k, services)[0]

    def retrieve_all(
        self, index: SearchIndex, queries: list[str], k: int, services: Services
    ) -> list[list[Hit]]:
        embeddings = _require_embeddings(index)
        if services.embedder is None:
            raise RuntimeError("dense retrieval needs an embedder")
        query_matrix = np.asarray(services.embedder.embed_queries(queries), dtype=np.float32)
        norms = np.linalg.norm(query_matrix, axis=1, keepdims=True)
        query_matrix = query_matrix / np.maximum(norms, 1e-12)
        scores = query_matrix @ embeddings.T
        return [hits_from_scores(index, scores[row], k) for row in range(scores.shape[0])]


@register_retriever
class HybridRetriever(Retriever):
    """Reciprocal rank fusion of BM25 and dense search."""

    name = "hybrid"
    needs_embeddings = True
    needs_bm25 = True
    score_type = "rrf"

    def __init__(self) -> None:
        self._dense = DenseRetriever()
        self._bm25 = BM25Retriever()

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        return self.retrieve_all(index, [query], k, services)[0]

    def retrieve_all(
        self, index: SearchIndex, queries: list[str], k: int, services: Services
    ) -> list[list[Hit]]:
        pool = max(k, services.fusion_pool)
        dense_hits = self._dense.retrieve_all(index, queries, pool, services)
        bm25_hits = self._bm25.retrieve_all(index, queries, pool, services)
        fused: list[list[Hit]] = []
        for dense, sparse in zip(dense_hits, bm25_hits, strict=True):
            ranked = reciprocal_rank_fusion(
                [[hit.chunk_id for hit in dense], [hit.chunk_id for hit in sparse]],
                k=services.rrf_k,
            )
            fused.append(hits_from_id_scores(index, ranked, k))
        return fused


def _order_head(head: list[Hit], scores: np.ndarray) -> list[Hit]:
    order = sorted(range(len(head)), key=lambda i: (-float(scores[i]), head[i].chunk_id))
    return [
        Hit(
            chunk_id=head[i].chunk_id,
            doc_id=head[i].doc_id,
            text=head[i].text,
            score=float(scores[i]),
        )
        for i in order
    ]


@register_retriever
class RerankRetriever(Retriever):
    """Cross-encoder over the head of a hybrid list. The tail keeps hybrid order."""

    name = "rerank"
    needs_embeddings = True
    needs_bm25 = True
    needs_reranker = True
    score_type = "cross_encoder_logit"

    def __init__(self) -> None:
        self._hybrid = HybridRetriever()

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        if services.reranker is None:
            raise RuntimeError("rerank needs a cross-encoder")
        pool = max(k, services.rerank_pool, services.fusion_pool)
        hits = self._hybrid.retrieve(index, query, pool, services)
        return self._apply(
            query, hits, k, services, services.reranker.score(query, _head_texts(hits, services))
        )

    def retrieve_all(
        self, index: SearchIndex, queries: list[str], k: int, services: Services
    ) -> list[list[Hit]]:
        if services.reranker is None:
            raise RuntimeError("rerank needs a cross-encoder")
        pool = max(k, services.rerank_pool, services.fusion_pool)
        base = self._hybrid.retrieve_all(index, queries, pool, services)
        pairs: list[tuple[str, str]] = []
        spans: list[tuple[int, int]] = []
        for query, hits in zip(queries, base, strict=True):
            start = len(pairs)
            for text in _head_texts(hits, services):
                pairs.append((query, text))
            spans.append((start, len(pairs)))
        logger.info("Reranking %s query-chunk pairs", f"{len(pairs):,}")
        scored = services.reranker.score_pairs(pairs)
        reranked: list[list[Hit]] = []
        for hits, (start, end) in zip(base, spans, strict=True):
            query = queries[len(reranked)]
            reranked.append(self._apply(query, hits, k, services, scored[start:end]))
        return reranked

    def _apply(
        self,
        query: str,
        hits: list[Hit],
        k: int,
        services: Services,
        scores: np.ndarray,
    ) -> list[Hit]:
        del query
        head_n = min(len(hits), services.rerank_pool)
        if len(scores) != head_n:
            raise RuntimeError(f"reranker returned {len(scores)} scores for {head_n} chunks")
        head = _order_head(hits[:head_n], scores)
        tail = hits[head_n:]
        return (head + tail)[:k]


def _head_texts(hits: list[Hit], services: Services) -> list[str]:
    return [hit.text for hit in hits[: services.rerank_pool]]


@register_retriever
class HyDERetriever(Retriever):
    """Embed a hypothetical answer, then dense-search with that vector."""

    name = "hyde"
    needs_embeddings = True
    needs_llm = True
    score_type = "cosine"

    def __init__(self) -> None:
        self._dense = DenseRetriever()
        self.fallbacks = 0

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        if services.llm is None:
            raise RuntimeError("HyDE needs an LLM client")
        try:
            passage = services.llm.complete(HYDE_PROMPT.format(query=query)).strip()
            if not passage:
                raise ValueError("empty hypothetical passage")
        except Exception as exc:
            self.fallbacks += 1
            logger.warning("HyDE fell back to the raw query (%s)", exc)
            passage = query
        return self._dense.retrieve(index, passage, k, services)


@register_retriever
class RAGFusionRetriever(Retriever):
    """Generate query variants, dense-search each one, fuse with RRF."""

    name = "rag_fusion"
    needs_embeddings = True
    needs_llm = True
    score_type = "rrf"

    def __init__(self) -> None:
        self._dense = DenseRetriever()
        self.fallbacks = 0

    def retrieve(self, index: SearchIndex, query: str, k: int, services: Services) -> list[Hit]:
        if services.llm is None:
            raise RuntimeError("RAG-Fusion needs an LLM client")
        variants = [query]
        try:
            raw = services.llm.complete(
                FUSION_PROMPT.format(n=services.rag_fusion_queries, query=query)
            )
            generated = _parse_queries(raw, services.rag_fusion_queries)
            if not generated:
                raise ValueError("model returned no queries")
            variants.extend(generated)
        except Exception as exc:
            self.fallbacks += 1
            logger.warning("RAG-Fusion used only the original query (%s)", exc)
        pool = max(k, services.fusion_pool)
        lists = self._dense.retrieve_all(index, variants, pool, services)
        ranked = reciprocal_rank_fusion(
            [[hit.chunk_id for hit in hits] for hits in lists],
            k=services.rrf_k,
        )
        return hits_from_id_scores(index, ranked, k)


def _parse_queries(text: str, limit: int) -> list[str]:
    payload = parse_json_object(text)
    queries = payload.get("queries", [])
    if not isinstance(queries, list):
        raise ValueError("queries was not a list")
    cleaned = []
    for query in queries:
        if isinstance(query, str) and query.strip():
            cleaned.append(query.strip())
    return cleaned[:limit]
