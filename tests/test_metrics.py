import math

import pytest

from raglab.metrics import (
    context_precision_at_k,
    dcg,
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    score_rankings,
)
from raglab.types import Hit


def _hit(doc_id: str, rank: int) -> Hit:
    return Hit(chunk_id=f"{doc_id}::c0", doc_id=doc_id, text=doc_id, score=1.0 / rank)


def test_recall_mrr_and_ndcg_on_a_known_ranking():
    documents = ["A", "C", "B"]
    grades = {"A": 1, "B": 1}
    assert recall_at_k(documents, set(grades), k=2) == pytest.approx(0.5)
    assert mrr_at_k(documents, set(grades), k=10) == pytest.approx(1.0)
    ideal = 1.0 + 1.0 / math.log2(3)
    actual = 1.0 + 1.0 / math.log2(4)
    assert ndcg_at_k(documents, grades, k=3) == pytest.approx(actual / ideal)
    assert dcg([1, 0, 1], 3) == pytest.approx(actual)


def test_graded_ndcg_uses_exponential_gain():
    # gains: rel 2 -> 3, rel 1 -> 1. Ranking puts the weaker document first.
    grades = {"A": 2, "B": 1}
    ranking = ["B", "A"]
    actual = 1.0 / math.log2(2) + 3.0 / math.log2(3)
    ideal = 3.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranking, grades, k=2) == pytest.approx(actual / ideal)


def test_context_precision_penalizes_an_irrelevant_chunk_above_a_relevant_one():
    hits = [_hit("A", 1), _hit("C", 2), _hit("B", 3)]
    score = context_precision_at_k(hits, {"A", "B"}, k=3)
    assert score == pytest.approx(5 / 6)


def test_trailing_irrelevant_chunks_do_not_lower_context_precision():
    hits = [_hit("A", 1), _hit("B", 2), _hit("C", 3), _hit("D", 4)]
    assert context_precision_at_k(hits, {"A", "B"}, k=4) == pytest.approx(1.0)


def test_duplicate_chunks_collapse_to_the_first_document_rank():
    hits = [
        Hit("A::c0", "A", "a0", 0.9),
        Hit("A::c1", "A", "a1", 0.8),
        Hit("B::c0", "B", "b", 0.1),
    ]
    summary = score_rankings({"q": hits}, {"q": {"A": 1}}, k_values=[1, 2])
    assert summary["metrics"]["recall@1"] == pytest.approx(1.0)
    assert summary["metrics"]["mrr@1"] == pytest.approx(1.0)
    assert summary["num_queries"] == 1


def test_queries_without_labels_are_skipped():
    hits = [_hit("A", 1)]
    summary = score_rankings({"q": hits}, {"q": {"A": 0}}, k_values=[1])
    assert summary["num_queries"] == 0
    assert summary["skipped_queries"] == 1
    assert summary["metrics"]["recall@1"] is None
