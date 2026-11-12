"""Reciprocal rank fusion.

``score(d) = sum over lists of 1 / (k + rank(d))`` with ranks starting at 1.
Ties break on the chunk id so the same inputs always produce the same order.
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_ids: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Sum of ``weight / (k + rank)`` across lists. Default weight is 1."""
    return _fuse(ranked_ids, k, weights, combine=sum)


def union_fusion(
    ranked_ids: list[list[str]],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[str, float]]:
    """Union of the lists. Score is the best single-list ``weight / (k + rank)``."""
    return _fuse(ranked_ids, k, weights, combine=max)


def _fuse(ranked_ids, k, weights, combine) -> list[tuple[str, float]]:
    if k < 0:
        raise ValueError("rrf k must be >= 0")
    if weights is None:
        weights = [1.0] * len(ranked_ids)
    if len(weights) != len(ranked_ids):
        raise ValueError("fusion weights must match the number of lists")
    scores: dict[str, float] = {}
    for weight, ids in zip(weights, ranked_ids, strict=True):
        for rank, chunk_id in enumerate(ids, start=1):
            value = float(weight) / (k + rank)
            previous = scores.get(chunk_id)
            scores[chunk_id] = value if previous is None else combine((previous, value))
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
