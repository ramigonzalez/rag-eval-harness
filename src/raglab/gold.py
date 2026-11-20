"""Match retrieved chunks to gold labels.

A label can name a source and a 1-based page, or a chunk id. A page label
matches a chunk when any page in the chunk's span is within ``tolerance`` of
the labeled page. ``any_source`` ignores the source id.
"""

from __future__ import annotations

from raglab.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from raglab.types import GoldLabel, Hit, Query


def label_key(label: GoldLabel, *, any_source: bool, match: str) -> str | None:
    if match == "chunk_id":
        if label.chunk_id:
            return f"chunk:{label.chunk_id}"
        if label.text_id:
            return f"chunk:{label.text_id}"
        return None
    if label.page is None:
        return None
    if any_source or not label.source_id:
        return f"page:{label.page}"
    return f"{label.source_id}:{label.page}"


def relevant_keys(query: Query, *, any_source: bool, match: str) -> set[str]:
    keys = set()
    for label in query.gold:
        key = label_key(label, any_source=any_source, match=match)
        if key:
            keys.add(key)
    return keys


def hit_matches(
    hit: Hit, label: GoldLabel, *, tolerance: int, any_source: bool, match: str
) -> bool:
    if match == "chunk_id":
        wanted = {label.chunk_id, label.text_id} - {""}
        return bool(wanted) and hit.chunk_id in wanted
    if label.page is None or hit.page_start is None:
        return False
    end = hit.page_end if hit.page_end is not None else hit.page_start
    if not any(abs(page - label.page) <= tolerance for page in range(hit.page_start, end + 1)):
        return False
    if any_source or not label.source_id:
        return True
    return (hit.source_id or hit.doc_id) == label.source_id


def ranked_keys(
    hits: list[Hit],
    query: Query,
    *,
    tolerance: int,
    any_source: bool,
    match: str,
) -> list[str]:
    """Gold keys in the order a retrieved chunk first satisfies them."""
    seen: set[str] = set()
    ordered: list[str] = []
    for hit in hits:
        for label in query.gold:
            key = label_key(label, any_source=any_source, match=match)
            if key is None or key in seen:
                continue
            if hit_matches(hit, label, tolerance=tolerance, any_source=any_source, match=match):
                seen.add(key)
                ordered.append(key)
    return ordered


def score_query(
    hits: list[Hit],
    query: Query,
    *,
    tolerance: int,
    any_source: bool,
    match: str,
    k_values: list[int],
) -> dict[str, float] | None:
    if not query.answerable:
        return None
    relevant = relevant_keys(query, any_source=any_source, match=match)
    if not relevant:
        return None
    ranked = ranked_keys(hits, query, tolerance=tolerance, any_source=any_source, match=match)
    grades = {key: 1 for key in relevant}
    scores: dict[str, float] = {}
    for k in k_values:
        scores[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        scores[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
        scores[f"ndcg@{k}"] = ndcg_at_k(ranked, grades, k)
    return scores
