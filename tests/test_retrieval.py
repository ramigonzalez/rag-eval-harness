import math

import numpy as np
import pytest

from raglab.retrieval.bm25 import BM25Index
from raglab.retrieval.fusion import reciprocal_rank_fusion
from raglab.retrieval.index import SearchIndex, rank_indices
from raglab.retrieval.strategies import HyDERetriever, RAGFusionRetriever
from raglab.types import Chunk


def test_bm25_matches_the_okapi_definition():
    corpus = [["red", "red", "blue"], ["blue", "green"], ["red"]]
    k1, b = 1.5, 0.75
    index = BM25Index(corpus, k1=k1, b=b)
    scores = index.score_all(["red"])
    n_docs = 3
    avgdl = 2.0
    df = 2
    idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def expected(tf: int, dl: int) -> float:
        if tf == 0:
            return 0.0
        return idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * dl / avgdl))

    assert scores == pytest.approx([expected(2, 3), expected(0, 2), expected(1, 1)])


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_id():
    fused = reciprocal_rank_fusion([["d1", "d2"], ["d1", "d3"]], k=60)
    assert [chunk_id for chunk_id, _score in fused] == ["d1", "d2", "d3"]
    assert fused[0][1] == pytest.approx(2 / 61)
    assert fused[1][1] == pytest.approx(1 / 62)
    assert fused[2][1] == pytest.approx(1 / 62)


def test_rank_ties_break_on_chunk_id():
    scores = np.array([0.2, 0.2, 0.9])
    order = rank_indices(scores, ["b", "a", "c"], k=3)
    assert order == [2, 1, 0]


def _index() -> SearchIndex:
    chunks = [
        Chunk("a::c0", "a", "alpha document", 0),
        Chunk("b::c0", "b", "beta document", 0),
    ]
    return SearchIndex(chunks=chunks, embeddings=np.ones((2, 4), dtype=np.float32), bm25=None)


class _CaptureEmbedder:
    model_name = "capture"
    dim = 4

    def __init__(self):
        self.queries: list[str] = []

    def embed_queries(self, texts):
        self.queries.extend(texts)
        return np.ones((len(texts), self.dim), dtype=np.float32)

    def embed_query(self, text):
        return self.embed_queries([text])[0]

    def embed_documents(self, texts):
        return np.ones((len(texts), self.dim), dtype=np.float32)


class _FakeLLM:
    def __init__(self, response: str):
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        self.prompts.append(prompt)
        return self.response


def test_hyde_embeds_the_hypothetical_passage():
    from raglab.retrieval.index import Services

    embedder = _CaptureEmbedder()
    llm = _FakeLLM("A glacier core records annual snowfall.")
    services = Services(embedder=embedder, llm=llm)
    hits = HyDERetriever().retrieve(_index(), "How old is the ice?", k=1, services=services)
    assert embedder.queries == ["A glacier core records annual snowfall."]
    assert len(hits) == 1


def test_rag_fusion_searches_the_original_query_and_the_variants():
    from raglab.retrieval.index import Services

    embedder = _CaptureEmbedder()
    llm = _FakeLLM('{"queries": ["glacier ice age", "ice core dating", "firn layers"]}')
    services = Services(embedder=embedder, llm=llm, rag_fusion_queries=3, rrf_k=60, fusion_pool=2)
    hits = RAGFusionRetriever().retrieve(_index(), "How old is the ice?", k=2, services=services)
    assert embedder.queries[0] == "How old is the ice?"
    assert embedder.queries[1:] == ["glacier ice age", "ice core dating", "firn layers"]
    assert len(hits) == 2


def test_rag_fusion_falls_back_when_the_model_returns_junk():
    from raglab.retrieval.index import Services

    embedder = _CaptureEmbedder()
    llm = _FakeLLM("not json")
    retriever = RAGFusionRetriever()
    services = Services(embedder=embedder, llm=llm, rag_fusion_queries=3)
    hits = retriever.retrieve(_index(), "How old is the ice?", k=1, services=services)
    assert retriever.fallbacks == 1
    assert embedder.queries == ["How old is the ice?"]
    assert hits
