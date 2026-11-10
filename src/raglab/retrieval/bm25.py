"""Okapi BM25 over an inverted index.

Each query term contributes once. The idf variant matches the common
``log(1 + (N - df + 0.5) / (df + 0.5))`` form, which stays positive when a
term is in every document.
"""

from __future__ import annotations

import math

import numpy as np


class BM25Index:
    def __init__(
        self,
        tokenized_docs: list[list[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_len = np.asarray([len(doc) for doc in tokenized_docs], dtype=np.float64)
        self.n_docs = len(tokenized_docs)
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        postings: dict[str, list[tuple[int, int]]] = {}
        doc_freq: dict[str, int] = {}
        for doc_index, doc in enumerate(tokenized_docs):
            counts: dict[str, int] = {}
            for token in doc:
                counts[token] = counts.get(token, 0) + 1
            for token, tf in counts.items():
                doc_freq[token] = doc_freq.get(token, 0) + 1
                postings.setdefault(token, []).append((doc_index, tf))
        self.postings = postings
        self.idf = {
            token: math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))
            for token, freq in doc_freq.items()
        }

    def score_all(self, query_tokens: list[str]) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0 or self.avgdl <= 0:
            return scores
        k1 = self.k1
        b = self.b
        avgdl = self.avgdl
        for token in dict.fromkeys(query_tokens):
            posting = self.postings.get(token)
            if not posting:
                continue
            idf = self.idf[token]
            for doc_index, tf in posting:
                dl = self.doc_len[doc_index]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_index] += idf * (tf * (k1 + 1.0)) / denom
        return scores
