import json
from pathlib import Path

import yaml

from raglab.config import load_config
from raglab.runner import run_benchmark

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path: Path, **overrides) -> Path:
    payload = {
        "dataset": "fixture",
        "fixture_path": str(ROOT / "data" / "fixtures" / "smoke.json"),
        "output_dir": str(tmp_path / "out"),
        "data_dir": str(tmp_path / "data"),
        "cache_dir": str(tmp_path / "cache"),
        "seed": 7,
        "k_values": [5, 10],
        "hit_k": 20,
        "fusion_pool": 20,
        "rerank_pool": 10,
        "rrf_k": 60,
        "latency_queries": 2,
        "embedder": {"kind": "hash", "dim": 256},
        "reranker": {"kind": "lexical"},
        "chunkers": [
            {"name": "whole"},
            {"name": "fixed", "chunk_size": 40, "overlap": 10},
            {"name": "recursive", "chunk_size": 40, "overlap": 10},
            {
                "name": "semantic",
                "breakpoint_percentile": 90,
                "max_chunk_words": 80,
                "min_chunk_words": 15,
            },
        ],
        "retrievers": ["bm25", "dense", "hybrid", "rerank"],
        "llm_retrievers": ["hyde", "rag_fusion"],
        "generation": {"mode": "auto", "max_queries": 2, "k": 3, "pairs": [["fixed", "hybrid"]]},
    }
    payload.update(overrides)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_fixture_grid_ranks_the_labeled_document_first(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAGBENCH_API_KEY", raising=False)
    result = run_benchmark(load_config(_config(tmp_path)))
    assert result["generation"]["status"] == "skipped"
    assert "run with API key" in result["generation"]["reason"]
    assert result["skipped_retrievers"] == ["hyde", "rag_fusion"]
    by_pair = {(row["chunker"], row["retriever"]): row for row in result["rows"]}
    for chunker in ("whole", "fixed", "recursive", "semantic"):
        for retriever in ("bm25", "dense", "hybrid", "rerank"):
            row = by_pair[(chunker, retriever)]
            assert row["status"] == "ok", row
            assert row["metrics"]["recall@10"] == 1.0
        # BM25 is deterministic. The nonce token in each labeled doc keeps it at rank 1.
        assert by_pair[(chunker, "bm25")]["metrics"]["mrr@10"] == 1.0
        assert by_pair[(chunker, "hybrid")]["metrics"]["mrr@10"] == 1.0
    output = tmp_path / "out"
    assert (output / "metrics.csv").exists()
    assert (output / "metrics.json").exists()
    assert (output / "results.md").exists()
    assert (output / "ndcg_at_10.png").stat().st_size > 0
    saved = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert saved["generation"]["status"] == "skipped"


def test_threshold_file_fails_a_score_above_one(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAGBENCH_API_KEY", raising=False)
    check = tmp_path / "thresholds.json"
    check.write_text(
        json.dumps(
            {"rules": [{"chunker": "fixed", "retriever": "bm25", "metric": "mrr@10", "min": 1.1}]}
        ),
        encoding="utf-8",
    )
    config = load_config(_config(tmp_path))
    config = type(config)(**{**config.__dict__, "check_path": check})
    result = run_benchmark(config)
    assert result["threshold_failures"]


def test_shipped_configs_parse():
    scifact = load_config(ROOT / "configs" / "scifact.yaml")
    smoke = load_config(ROOT / "configs" / "smoke.yaml")
    assert [spec.name for spec in scifact.chunkers] == ["whole", "fixed", "recursive", "semantic"]
    assert scifact.retrievers == ["bm25", "dense", "hybrid", "rerank"]
    assert scifact.llm_retrievers == ["hyde", "rag_fusion"]
    assert scifact.embedder.model == "sentence-transformers/all-MiniLM-L6-v2"
    assert smoke.dataset == "fixture"
    assert smoke.generation.mode == "off"
