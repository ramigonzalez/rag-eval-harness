"""Parity with rank_bm25 and pytrec_eval.

Tolerances are absolute unless noted. The BM25 comparison is not an identity
check: rank_bm25.BM25Okapi uses log((N-df+0.5)/(df+0.5)) with an epsilon floor
and scores repeated query terms once per occurrence. This repo uses
log(1+(N-df+0.5)/(df+0.5)) and scores each query term once.
"""

import math

import pytest
from rank_bm25 import BM25Okapi

from raglab.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from raglab.retrieval.bm25 import BM25Index

BM25_RTOL = 1e-9
TREC_ATOL = 1e-8


def test_single_term_scores_match_rank_bm25_up_to_the_idf_variant():
    corpus = [["red", "red", "blue"], ["blue", "green"], ["alpha"], ["beta", "gamma"]]
    ours = BM25Index(corpus, k1=1.5, b=0.75)
    theirs = BM25Okapi(corpus, k1=1.5, b=0.75)
    query = ["red"]
    our_scores = ours.score_all(query)
    their_scores = theirs.get_scores(query)
    our_idf = ours.idf["red"]
    their_idf = theirs.idf["red"]
    assert their_idf > 0
    for our_score, their_score in zip(our_scores, their_scores, strict=True):
        if their_score == 0:
            assert our_score == pytest.approx(0.0, abs=BM25_RTOL)
            continue
        assert our_score / their_score == pytest.approx(our_idf / their_idf, rel=BM25_RTOL)
    # The formulas are not the same number.
    assert our_idf != pytest.approx(their_idf)


def test_rank_bm25_counts_a_repeated_query_term_and_we_do_not():
    corpus = [["cats", "cats", "dogs"]]
    ours = BM25Index(corpus, k1=1.5, b=0.75)
    theirs = BM25Okapi(corpus, k1=1.5, b=0.75)
    once = ours.score_all(["cats"])[0]
    repeated_ours = ours.score_all(["cats", "cats"])[0]
    repeated_theirs = theirs.get_scores(["cats", "cats"])[0]
    single_theirs = theirs.get_scores(["cats"])[0]
    assert repeated_ours == pytest.approx(once, rel=BM25_RTOL)
    assert repeated_theirs == pytest.approx(2 * single_theirs, rel=BM25_RTOL)


def test_metrics_match_pytrec_eval():
    import pytrec_eval

    qrels = {
        "q1": {"A": 1, "B": 1},
        "q2": {"C": 1},
    }
    # Ranked document lists. Scores only encode order for pytrec_eval.
    runs = {
        "q1": ["A", "C", "B"],
        "q2": ["Z", "C"],
    }
    our_ndcg = [
        ndcg_at_k(runs["q1"], qrels["q1"], 10),
        ndcg_at_k(runs["q2"], qrels["q2"], 10),
    ]
    our_recall = [
        recall_at_k(runs["q1"], set(qrels["q1"]), 10),
        recall_at_k(runs["q2"], {"C"}, 10),
    ]
    our_mrr = [
        mrr_at_k(runs["q1"], set(qrels["q1"]), 10),
        mrr_at_k(runs["q2"], {"C"}, 10),
    ]
    trec_run = {
        query_id: {doc_id: float(len(docs) - rank) for rank, doc_id in enumerate(docs[:10])}
        for query_id, docs in runs.items()
    }
    # recip_rank on a run already cut at 10 is MRR@10.
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {"ndcg_cut.10", "recall.10", "recip_rank"})
    per_query = evaluator.evaluate(trec_run)
    assert _mean(our_ndcg) == pytest.approx(
        _mean([per_query[query]["ndcg_cut_10"] for query in runs]), abs=TREC_ATOL
    )
    assert _mean(our_recall) == pytest.approx(
        _mean([per_query[query]["recall_10"] for query in runs]), abs=TREC_ATOL
    )
    assert _mean(our_mrr) == pytest.approx(
        _mean([per_query[query]["recip_rank"] for query in runs]), abs=TREC_ATOL
    )
    # A relevant document past rank 10 does not count for MRR@10.
    long_docs = ["Z"] * 10 + ["C"]
    assert mrr_at_k(long_docs, {"C"}, 10) == 0.0
    long_run = {"q2": {f"d{i}": float(20 - i) for i in range(10)}}
    long_run["q2"]["C"] = 0.1
    # C is rank 11 if we keep only the ten highest scores, so drop it before pytrec.
    top10 = dict(sorted(long_run["q2"].items(), key=lambda item: -item[1])[:10])
    assert "C" not in top10
    cut = pytrec_eval.RelevanceEvaluator({"q2": {"C": 1}}, {"recip_rank"}).evaluate({"q2": top10})
    assert cut["q2"]["recip_rank"] == pytest.approx(0.0, abs=TREC_ATOL)
    assert math.isclose(our_ndcg[0], per_query["q1"]["ndcg_cut_10"], abs_tol=TREC_ATOL)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)
