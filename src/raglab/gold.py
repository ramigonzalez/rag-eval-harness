"""Match retrieved chunks to gold labels.

A label can name a source and a 1-based page, or a chunk id. A page label
matches a chunk when any page in the chunk's span is within ``tolerance`` of
the labeled page. ``any_source`` ignores the source id.
"""

from __future__ import annotations

from raglab.metrics import dcg
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


def first_ranks(
    hits: list[Hit],
    query: Query,
    *,
    tolerance: int,
    any_source: bool,
    match: str,
) -> tuple[dict[str, int], list[float]]:
    """Hit index of each gold key, and a gain of 1 for the hit that first covers one.

    Each retrieved hit occupies one rank. A repeated page still counts, and a
    window that spans several pages counts once. Labels that share a key count once.
    """
    found: dict[str, int] = {}
    gains: list[float] = []
    for index, hit in enumerate(hits):
        new = False
        for label in query.gold:
            key = label_key(label, any_source=any_source, match=match)
            if key is None or key in found:
                continue
            if hit_matches(hit, label, tolerance=tolerance, any_source=any_source, match=match):
                found[key] = index
                new = True
        gains.append(1.0 if new else 0.0)
    return found, gains


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
    found, gains = first_ranks(hits, query, tolerance=tolerance, any_source=any_source, match=match)
    scores: dict[str, float] = {}
    for k in k_values:
        inside = [rank for rank in found.values() if rank < k]
        scores[f"recall@{k}"] = len(inside) / len(relevant)
        scores[f"mrr@{k}"] = 0.0 if not inside else 1.0 / (min(inside) + 1)
        scores[f"ndcg@{k}"] = dcg(gains, k) / dcg([1.0] * len(relevant), k)
    return scores
