"""Cross-encoder and lexical rerankers. Both return one score per input, in order."""

from __future__ import annotations

import numpy as np

from raglab.textutil import tokenize


class LexicalReranker:
    """Overlap reranker used by tests and offline smoke runs. Not a model."""

    name = "lexical"

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        return self.score_pairs([(query, text) for text in texts])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        scores = np.zeros(len(pairs), dtype=np.float32)
        for index, (query, text) in enumerate(pairs):
            query_terms = set(tokenize(query))
            tokens = tokenize(text)
            if not query_terms or not tokens:
                continue
            scores[index] = sum(1 for token in tokens if token in query_terms) / len(tokens)
        return scores


class FastEmbedReranker:
    """Local ONNX cross-encoder. Scores are logits; higher means more relevant."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self.name = model_name
        self.batch_size = batch_size
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model = TextCrossEncoder(model_name=model_name)

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = list(self.model.rerank(query, texts, batch_size=self.batch_size))
        return np.asarray(scores, dtype=np.float32)

    def score_pairs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if not pairs:
            return np.zeros(0, dtype=np.float32)
        scores = list(self.model.rerank_pairs(pairs, batch_size=self.batch_size))
        return np.asarray(scores, dtype=np.float32)
