"""Local embedders. The benchmark default is MiniLM through ONNX, not a hosted API."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path

import numpy as np

from raglab.textutil import tokenize

logger = logging.getLogger("raglab")


class HashEmbedder:
    """Deterministic bag-of-tokens vectors for tests. Not a semantic model."""

    def __init__(self, dim: int = 256) -> None:
        if dim < 8:
            raise ValueError("hash embedder dim must be >= 8")
        self.model_name = "hash"
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in tokenize(text):
                digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                matrix[row, bucket] += sign
        return matrix

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class FastEmbedEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from fastembed import TextEmbedding

        logger.info("Loading embedding model %s", model_name)
        self.model = TextEmbedding(model_name=model_name)
        self.model_name = model_name
        self.batch_size = batch_size
        self.dim = int(self.model.embedding_size)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.embed(list(texts), batch_size=self.batch_size)
        return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.query_embed(list(texts))
        return np.vstack([np.asarray(vector, dtype=np.float32) for vector in vectors])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


class CachingEmbedder:
    """Disk cache for document embeddings. Query embeddings are not cached."""

    def __init__(self, inner: object, cache_dir: Path | None) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.dim = inner.dim
        self.cache_dir = cache_dir

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts or self.cache_dir is None or self.model_name == "hash":
            return self.inner.embed_documents(texts)
        digest = _digest(self.model_name, texts)
        path = self.cache_dir / f"emb-{digest}.npy"
        if path.exists():
            cached = np.load(path)
            if cached.shape[0] == len(texts):
                logger.info("Using %s cached embeddings", f"{len(texts):,}")
                return cached
        logger.info("Embedding %s texts with %s", f"{len(texts):,}", self.model_name)
        matrix = np.asarray(self.inner.embed_documents(texts), dtype=np.float32)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.inner.embed_queries(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self.inner.embed_query(text)


def load_embedder(kind: str, model: str, batch_size: int, dim: int, cache_dir: Path | None):
    if kind == "hash":
        return HashEmbedder(dim=dim)
    if kind == "fastembed":
        if not model:
            raise ValueError("fastembed embedder needs a model name")
        return CachingEmbedder(FastEmbedEmbedder(model, batch_size=batch_size), cache_dir)
    raise ValueError(f"Unknown embedder kind '{kind}'. Use 'fastembed' or 'hash'.")


def _digest(model_name: str, texts: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(model_name.encode())
    hasher.update(str(len(texts)).encode())
    for text in texts:
        hasher.update(b"\x1f")
        hasher.update(text.encode())
    return hasher.hexdigest()[:24]
