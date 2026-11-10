"""Retrieval strategies. Importing this package registers the built-ins."""

from raglab.retrieval.registry import RETRIEVERS, build_retriever, register_retriever
from raglab.retrieval.strategies import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HyDERetriever,
    RAGFusionRetriever,
    RerankRetriever,
)

__all__ = [
    "BM25Retriever",
    "DenseRetriever",
    "HybridRetriever",
    "HyDERetriever",
    "RAGFusionRetriever",
    "RETRIEVERS",
    "RerankRetriever",
    "build_retriever",
    "register_retriever",
]
