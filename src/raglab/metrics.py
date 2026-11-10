"""Retrieval metrics.

Recall, MRR, and nDCG are document-level: chunks are collapsed to their source
document, keeping the best rank. Context precision stays chunk-level, because
that is the window a generator would see. A chunk counts as relevant when its
source document is a labeled relevant document.

Context precision follows the RAGAS definition. For verdicts v_k (1 if chunk k
is relevant), precision@k is the fraction of the first k chunks that are
relevant, and the score is sum(precision@k * v_k) / sum(v_k). Irrelevant chunks
placed below the last relevant chunk do not change it. Irrelevant chunks placed
above a relevant one do.
"""

from __future__ import annotations

import math

from raglab.types import Hit


def dcg(relevances: list[float], k: int) -> float:
    total = 0.0
    for index, relevance in enumerate(relevances[:k]):
        total += (2.0**relevance - 1.0) / math.log2(index + 2)
    return total


def dedupe_documents(hits: list[Hit]) -> list[str]:
    seen: set[str] = set()
    documents: list[str] = []
    for hit in hits:
        if hit.doc_id in seen:
            continue
        seen.add(hit.doc_id)
        documents.append(hit.doc_id)
    return documents


def recall_at_k(documents: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    found = sum(1 for doc_id in documents[:k] if doc_id in relevant)
    return found / len(relevant)


def mrr_at_k(documents: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(documents[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(documents: list[str], grades: dict[str, int], k: int) -> float:
    relevances = [float(grades.get(doc_id, 0)) for doc_id in documents[:k]]
    ideal = sorted((float(grade) for grade in grades.values() if grade > 0), reverse=True)
    ideal_dcg = dcg(ideal, k)
    if ideal_dcg == 0:
        raise ValueError("nDCG needs at least one relevant document")
    return dcg(relevances, k) / ideal_dcg


def context_precision_at_k(hits: list[Hit], relevant: set[str], k: int) -> float:
    verdicts = [1.0 if hit.doc_id in relevant else 0.0 for hit in hits[:k]]
    denominator = sum(verdicts)
    if denominator == 0:
        return 0.0
    numerator = 0.0
    seen = 0.0
    for index, verdict in enumerate(verdicts, start=1):
        seen += verdict
        if verdict:
            numerator += seen / index
    return numerator / denominator


def score_rankings(
    rankings: dict[str, list[Hit]],
    qrels: dict[str, dict[str, int]],
    k_values: list[int],
) -> dict:
    """Macro-average each metric over queries that have a relevant document."""
    names = ("recall", "mrr", "ndcg", "context_precision")
    buckets = {f"{name}@{k}": [] for k in k_values for name in names}
    unique_counts: list[int] = []
    used = 0
    skipped = 0
    for query_id, hits in rankings.items():
        grades = {doc_id: grade for doc_id, grade in qrels.get(query_id, {}).items() if grade > 0}
        if not grades:
            skipped += 1
            continue
        used += 1
        documents = dedupe_documents(hits)
        unique_counts.append(len(documents))
        relevant = set(grades)
        for k in k_values:
            buckets[f"recall@{k}"].append(recall_at_k(documents, relevant, k))
            buckets[f"mrr@{k}"].append(mrr_at_k(documents, relevant, k))
            buckets[f"ndcg@{k}"].append(ndcg_at_k(documents, grades, k))
            buckets[f"context_precision@{k}"].append(context_precision_at_k(hits, relevant, k))
    metrics = {
        key: (sum(values) / len(values) if values else None) for key, values in buckets.items()
    }
    return {
        "metrics": metrics,
        "num_queries": used,
        "skipped_queries": skipped,
        "min_unique_docs": min(unique_counts) if unique_counts else 0,
        "mean_unique_docs": (sum(unique_counts) / len(unique_counts) if unique_counts else 0.0),
    }
