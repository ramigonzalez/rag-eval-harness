"""One retrieval pipeline made of switchable stages.

Stages, in order: chunking, query transform, retrieval, rerank.
``raglab run`` does not use this module. The ablation command does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from raglab.chunking import build_chunker
from raglab.llm import llm_configured
from raglab.retrieval.fusion import reciprocal_rank_fusion, union_fusion
from raglab.retrieval.index import (
    SearchIndex,
    Services,
    build_search_index,
    hits_from_id_scores,
    l2_normalize,
)
from raglab.retrieval.strategies import (
    FUSION_PROMPT,
    HYDE_PROMPT,
    BM25Retriever,
    DenseRetriever,
    _parse_queries,
)
from raglab.types import Document, Hit

STEP_BACK_PROMPT = """Task: step back.

Write one broader question that would help retrieve background
for the user question. Return only that question.

Question: {query}
"""

LLM_TRANSFORMS = {"hyde", "rag_fusion", "step_back"}
RETRIEVALS = {"bm25", "dense", "rrf", "union"}
RERANKERS = {"none", "cross_encoder", "mmr"}
TRANSFORMS = {"none", *LLM_TRANSFORMS}


@dataclass(frozen=True)
class PipelineSpec:
    chunking: str
    query_transform: str
    retrieval: str
    rerank: str


@dataclass(frozen=True)
class PipelineParams:
    chunk_size: int = 100
    overlap: int = 25
    top_k: int = 10
    hit_k: int = 60
    fusion_pool: int = 80
    rrf_k: int = 60
    bm25_weight: float = 1.0
    dense_weight: float = 1.0
    rerank_pool: int = 20
    mmr_lambda: float = 0.7
    semantic_percentile: float = 95.0
    semantic_max_words: int = 220
    semantic_min_words: int = 40
    sentence_window: int = 3
    sentence_overlap: int = 1
    rag_fusion_queries: int = 3
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


def chunker_options(name: str, params: PipelineParams) -> dict:
    if name in {"fixed", "recursive", "parent_document"}:
        return {"chunk_size": params.chunk_size, "overlap": params.overlap}
    if name == "semantic":
        return {
            "breakpoint_percentile": params.semantic_percentile,
            "max_chunk_words": params.semantic_max_words,
            "min_chunk_words": params.semantic_min_words,
        }
    if name == "sentence_window":
        return {
            "window_sentences": params.sentence_window,
            "overlap_sentences": params.sentence_overlap,
        }
    if name in {"whole", "markdown"}:
        return {}
    raise KeyError(name)


def build_pipeline_index(documents: list[Document], name: str, params: PipelineParams, embedder):
    chunker = build_chunker(name, chunker_options(name, params))
    chunks = chunker.chunk_corpus(documents, embedder)
    if not chunks:
        raise RuntimeError(f"Chunker {name} produced no chunks")
    index = build_search_index(
        chunks,
        embedder,
        build_dense=True,
        build_bm25=True,
        bm25_k1=params.bm25_k1,
        bm25_b=params.bm25_b,
    )
    return index


def transform_available(name: str) -> bool:
    if name not in TRANSFORMS:
        raise KeyError(f"Unknown query transform {name!r}")
    if name in LLM_TRANSFORMS:
        return llm_configured()
    return True


def apply_transform(query: str, name: str, services: Services, params: PipelineParams) -> list[str]:
    if name == "none":
        return [query]
    if services.llm is None:
        raise RuntimeError(f"{name} needs an LLM client")
    if name == "hyde":
        passage = services.llm.complete(HYDE_PROMPT.format(query=query)).strip()
        return [passage or query]
    if name == "step_back":
        broader = services.llm.complete(STEP_BACK_PROMPT.format(query=query)).strip()
        return [broader or query]
    if name == "rag_fusion":
        raw = services.llm.complete(FUSION_PROMPT.format(n=params.rag_fusion_queries, query=query))
        try:
            generated = _parse_queries(raw, params.rag_fusion_queries)
        except (ValueError, TypeError):
            generated = []
        return [query, *generated] if generated else [query]
    raise KeyError(name)


def retrieve_batch(
    index: SearchIndex,
    queries: list[str],
    kind: str,
    pool: int,
    services: Services,
    params: PipelineParams,
) -> list[list[Hit]]:
    if kind == "bm25":
        return BM25Retriever().retrieve_all(index, queries, pool, services)
    if kind == "dense":
        return DenseRetriever().retrieve_all(index, queries, pool, services)
    if kind not in {"rrf", "union"}:
        raise KeyError(f"Unknown retrieval {kind!r}")
    dense_hits = DenseRetriever().retrieve_all(index, queries, pool, services)
    bm25_hits = BM25Retriever().retrieve_all(index, queries, pool, services)
    fuse = union_fusion if kind == "union" else reciprocal_rank_fusion
    weights = [params.bm25_weight, params.dense_weight]
    fused: list[list[Hit]] = []
    for dense, sparse in zip(dense_hits, bm25_hits, strict=True):
        ranked = fuse(
            [[hit.chunk_id for hit in sparse], [hit.chunk_id for hit in dense]],
            k=params.rrf_k,
            weights=weights,
        )
        fused.append(hits_from_id_scores(index, ranked, pool))
    return fused


def run_pipeline(
    index: SearchIndex,
    queries: list[str],
    spec: PipelineSpec,
    services: Services,
    params: PipelineParams,
) -> list[list[Hit]]:
    """Run every query. Transforms are per query; retrieval is batched."""
    pool = max(params.top_k, params.hit_k, params.fusion_pool, params.rerank_pool)
    variants = [apply_transform(query, spec.query_transform, services, params) for query in queries]
    width = max(len(item) for item in variants)
    # Pad shorter variant lists by repeating the original query so the batches align.
    padded = [item + [queries[index]] * (width - len(item)) for index, item in enumerate(variants)]
    per_variant: list[list[list[Hit]]] = []
    for column in range(width):
        column_queries = [padded[row][column] for row in range(len(queries))]
        per_variant.append(
            retrieve_batch(index, column_queries, spec.retrieval, pool, services, params)
        )
    retrieved: list[list[Hit]] = []
    for row in range(len(queries)):
        lists = [per_variant[column][row] for column in range(width)]
        # Drop the padding copies: only fuse the real variants.
        lists = lists[: len(variants[row])]
        if len(lists) == 1:
            merged = lists[0]
        else:
            ranked = reciprocal_rank_fusion(
                [[hit.chunk_id for hit in hits] for hits in lists],
                k=params.rrf_k,
            )
            merged = hits_from_id_scores(index, ranked, pool)
        retrieved.append(_rerank(queries[row], merged, index, spec, services, params))
    return retrieved


def _rerank(query, hits, index, spec: PipelineSpec, services: Services, params: PipelineParams):
    if spec.rerank == "none":
        return hits[: params.top_k]
    if spec.rerank == "cross_encoder":
        return _cross_encode(query, hits, services, params)
    if spec.rerank == "mmr":
        return _mmr(query, hits, index, services, params)
    raise KeyError(spec.rerank)


def _cross_encode(query, hits, services: Services, params: PipelineParams) -> list[Hit]:
    if services.reranker is None:
        raise RuntimeError("cross_encoder rerank needs a reranker")
    head_n = min(len(hits), params.rerank_pool)
    head = hits[:head_n]
    if not head:
        return []
    scores = services.reranker.score(query, [hit.text for hit in head])
    order = sorted(range(head_n), key=lambda index: (-float(scores[index]), head[index].chunk_id))
    ranked = [
        Hit(head[index].chunk_id, head[index].doc_id, head[index].text, float(scores[index]))
        for index in order
    ]
    return (ranked + list(hits[head_n:]))[: params.top_k]


def _mmr(query, hits, index: SearchIndex, services: Services, params: PipelineParams) -> list[Hit]:
    if index.embeddings is None or services.embedder is None:
        raise RuntimeError("MMR needs chunk embeddings")
    head_n = min(len(hits), params.rerank_pool)
    head = hits[:head_n]
    if not head:
        return []
    position_of = {chunk_id: position for position, chunk_id in enumerate(index.chunk_ids)}
    positions = [position_of[hit.chunk_id] for hit in head]
    matrix = np.asarray(index.embeddings[positions], dtype=np.float32)
    query_vector = np.asarray(services.embedder.embed_query(query), dtype=np.float32)
    query_vector = l2_normalize(query_vector.reshape(1, -1))[0]
    relevance = matrix @ query_vector
    selected: list[int] = []
    remaining = set(range(head_n))
    lam = params.mmr_lambda
    while remaining and len(selected) < params.top_k:
        best_index = None
        best_score = None
        for candidate in remaining:
            diversity = 0.0
            if selected:
                diversity = max(float(matrix[candidate] @ matrix[chosen]) for chosen in selected)
            score = lam * float(relevance[candidate]) - (1.0 - lam) * diversity
            if (
                best_score is None
                or score > best_score
                or (score == best_score and head[candidate].chunk_id < head[best_index].chunk_id)
            ):
                best_score = score
                best_index = candidate
        selected.append(best_index)
        remaining.remove(best_index)
    ranked = [
        Hit(head[index].chunk_id, head[index].doc_id, head[index].text, float(relevance[index]))
        for index in selected
    ]
    return ranked
