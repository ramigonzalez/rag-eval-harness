"""Stage-wise ablation.

Each stage is chosen on a seeded dev split. The chosen pipeline is then scored
on a held-out test split. The table is the test nDCG@10 delta of each choice,
plus latency and dollar cost. Local stages cost $0. LLM query transforms are
skipped when no API key is set, and this module does not invent their scores.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from raglab.ablation_report import write_ablation
from raglab.config import ConfigError
from raglab.datasets import load_dataset, load_scifact
from raglab.embeddings import load_embedder
from raglab.llm import LLMClient, llm_configured
from raglab.metrics import score_rankings
from raglab.pipeline import (
    LLM_TRANSFORMS,
    PipelineParams,
    PipelineSpec,
    build_pipeline_index,
    run_pipeline,
    transform_available,
)
from raglab.retrieval.index import Services
from raglab.retrieval.rerank import FastEmbedReranker, LexicalReranker
from raglab.runner import select_queries
from raglab.types import Dataset, Query

logger = logging.getLogger("raglab")

STAGE_ORDER = ("chunking", "query_transform", "retrieval", "rerank")


@dataclass
class AblationConfig:
    dataset: str
    fixture_path: Path | None
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    seed: int
    dev_split: str | None
    test_split: str | None
    dev_max_queries: int | None
    test_max_queries: int | None
    dev_count: int | None
    baseline: PipelineSpec
    stages: dict[str, list[str]]
    params: PipelineParams
    embedder_kind: str
    embedder_model: str
    embedder_batch_size: int
    embedder_dim: int
    reranker_kind: str
    reranker_model: str
    reranker_batch_size: int
    input_usd_per_million: float
    output_usd_per_million: float


def load_ablation_config(path: str | Path) -> AblationConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ConfigError("Ablation config must be a mapping")
    root = _find_root(config_path)
    baseline = payload.get("baseline") or {}
    stages = payload.get("stages") or {}
    for stage in STAGE_ORDER:
        options = stages.get(stage)
        if not isinstance(options, list) or not options:
            raise ConfigError(f"stages.{stage} must be a non-empty list")
    params = payload.get("params") or {}
    embedder = payload.get("embedder") or {}
    reranker = payload.get("reranker") or {}
    llm = payload.get("llm") or {}
    return AblationConfig(
        dataset=str(payload.get("dataset", "scifact")),
        fixture_path=_optional_path(root, payload.get("fixture_path")),
        data_dir=_path(root, payload.get("data_dir", "data")),
        cache_dir=_path(root, payload.get("cache_dir", "data/cache")),
        output_dir=_path(root, payload.get("output_dir", "results/ablation")),
        seed=int(payload.get("seed", 7)),
        dev_split=payload.get("dev_split"),
        test_split=payload.get("test_split"),
        dev_max_queries=_optional_int(payload.get("dev_max_queries")),
        test_max_queries=_optional_int(payload.get("test_max_queries")),
        dev_count=_optional_int(payload.get("dev_count")),
        baseline=PipelineSpec(
            chunking=str(baseline.get("chunking", "whole")),
            query_transform=str(baseline.get("query_transform", "none")),
            retrieval=str(baseline.get("retrieval", "bm25")),
            rerank=str(baseline.get("rerank", "none")),
        ),
        stages={stage: [str(item) for item in stages[stage]] for stage in STAGE_ORDER},
        params=PipelineParams(
            chunk_size=int(params.get("chunk_size", 100)),
            overlap=int(params.get("overlap", 25)),
            top_k=int(params.get("top_k", 10)),
            hit_k=int(params.get("hit_k", 60)),
            fusion_pool=int(params.get("fusion_pool", 80)),
            rrf_k=int(params.get("rrf_k", 60)),
            bm25_weight=float(params.get("bm25_weight", 1.0)),
            dense_weight=float(params.get("dense_weight", 1.0)),
            rerank_pool=int(params.get("rerank_pool", 20)),
            mmr_lambda=float(params.get("mmr_lambda", 0.7)),
            semantic_percentile=float(params.get("semantic_percentile", 95)),
            semantic_max_words=int(params.get("semantic_max_words", 220)),
            semantic_min_words=int(params.get("semantic_min_words", 40)),
            sentence_window=int(params.get("sentence_window", 3)),
            sentence_overlap=int(params.get("sentence_overlap", 1)),
            rag_fusion_queries=int(params.get("rag_fusion_queries", 3)),
        ),
        embedder_kind=str(embedder.get("kind", "fastembed")),
        embedder_model=str(embedder.get("model", "sentence-transformers/all-MiniLM-L6-v2")),
        embedder_batch_size=int(embedder.get("batch_size", 128)),
        embedder_dim=int(embedder.get("dim", 256)),
        reranker_kind=str(reranker.get("kind", "fastembed")),
        reranker_model=str(reranker.get("model", "Xenova/ms-marco-MiniLM-L-6-v2")),
        reranker_batch_size=int(reranker.get("batch_size", 32)),
        input_usd_per_million=float(llm.get("input_usd_per_million", 0.15)),
        output_usd_per_million=float(llm.get("output_usd_per_million", 0.60)),
    )


def run_ablation(config: AblationConfig) -> dict:
    dev_queries, test_queries, dev_qrels, test_qrels, documents, dev_meta, test_meta = _load_splits(
        config
    )
    embedder = load_embedder(
        config.embedder_kind,
        config.embedder_model,
        config.embedder_batch_size,
        config.embedder_dim,
        config.cache_dir,
    )
    reranker = _load_reranker(config)
    llm = LLMClient() if llm_configured() else None
    services = Services(
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        rrf_k=config.params.rrf_k,
        fusion_pool=config.params.fusion_pool,
        rerank_pool=config.params.rerank_pool,
        rag_fusion_queries=config.params.rag_fusion_queries,
    )
    indexes: dict = {}
    current = config.baseline
    rows = []
    skipped: list[dict] = []
    logger.info("Scoring the baseline on the dev and test splits")
    baseline_dev = _score(current, dev_queries, dev_qrels, documents, indexes, services, config)
    baseline_test = _score(current, test_queries, test_qrels, documents, indexes, services, config)
    previous_test = baseline_test["ndcg@10"]
    rows.append(
        _row(
            stage="baseline",
            chosen=_spec_label(current),
            dev=baseline_dev,
            test=baseline_test,
            delta=None,
            candidates=[],
        )
    )
    for stage in STAGE_ORDER:
        logger.info("Selecting %s on the dev split", stage)
        candidates = []
        winner_name = None
        winner_dev = None
        for option in config.stages[stage]:
            if stage == "query_transform" and not transform_available(option):
                skipped.append({"stage": stage, "option": option, "reason": "no API key"})
                logger.info("Skipping %s/%s because no API key is set", stage, option)
                continue
            spec = _with_stage(current, stage, option)
            measured = _score(spec, dev_queries, dev_qrels, documents, indexes, services, config)
            candidates.append({"option": option, **measured})
            if winner_dev is None or measured["ndcg@10"] > winner_dev:
                winner_dev = measured["ndcg@10"]
                winner_name = option
        if winner_name is None:
            raise ConfigError(f"No available option for stage {stage}")
        current = _with_stage(current, stage, winner_name)
        logger.info(
            "Chosen %s=%s on dev nDCG@10 %.3f. Scoring the test split.",
            stage,
            winner_name,
            winner_dev,
        )
        test_measured = _score(
            current, test_queries, test_qrels, documents, indexes, services, config
        )
        delta = test_measured["ndcg@10"] - previous_test
        previous_test = test_measured["ndcg@10"]
        chosen_dev = next(item for item in candidates if item["option"] == winner_name)
        rows.append(
            _row(
                stage=stage,
                chosen=winner_name,
                dev=chosen_dev,
                test=test_measured,
                delta=delta,
                candidates=candidates,
            )
        )
    result = {
        "dataset": config.dataset,
        "seed": config.seed,
        "dev": dev_meta,
        "test": test_meta,
        "baseline": {
            "chunking": config.baseline.chunking,
            "query_transform": config.baseline.query_transform,
            "retrieval": config.baseline.retrieval,
            "rerank": config.baseline.rerank,
        },
        "final": {
            "chunking": current.chunking,
            "query_transform": current.query_transform,
            "retrieval": current.retrieval,
            "rerank": current.rerank,
        },
        "rows": rows,
        "skipped": skipped,
        "llm_transforms_available": sorted(name for name in LLM_TRANSFORMS if llm_configured()),
        "params": {
            "chunk_size": config.params.chunk_size,
            "overlap": config.params.overlap,
            "top_k": config.params.top_k,
            "hit_k": config.params.hit_k,
            "fusion_pool": config.params.fusion_pool,
            "rrf_k": config.params.rrf_k,
            "bm25_weight": config.params.bm25_weight,
            "dense_weight": config.params.dense_weight,
            "rerank_pool": config.params.rerank_pool,
            "mmr_lambda": config.params.mmr_lambda,
        },
    }
    write_ablation(result, config.output_dir)
    logger.info("Wrote ablation results to %s", config.output_dir)
    return result


def _score(spec, queries, qrels, documents, indexes, services, config: AblationConfig) -> dict:
    index = _index_for(spec.chunking, documents, indexes, services.embedder, config.params)
    started = time.perf_counter()
    hits = run_pipeline(index, [query.text for query in queries], spec, services, config.params)
    elapsed = time.perf_counter() - started
    rankings = {query.query_id: found for query, found in zip(queries, hits, strict=True)}
    summary = score_rankings(rankings, qrels, [config.params.top_k])
    metric = f"ndcg@{config.params.top_k}"
    return {
        "ndcg@10": summary["metrics"][metric],
        "latency_ms": elapsed / max(len(queries), 1) * 1000.0,
        "estimated_usd": 0.0,
        "num_queries": summary["num_queries"],
    }


def _index_for(chunking, documents, indexes, embedder, params):
    if chunking not in indexes:
        logger.info("Building the %s index", chunking)
        indexes[chunking] = build_pipeline_index(documents, chunking, params, embedder)
    return indexes[chunking]


def _with_stage(spec: PipelineSpec, stage: str, option: str) -> PipelineSpec:
    if stage == "chunking":
        return replace(spec, chunking=option)
    if stage == "query_transform":
        return replace(spec, query_transform=option)
    if stage == "retrieval":
        return replace(spec, retrieval=option)
    if stage == "rerank":
        return replace(spec, rerank=option)
    raise KeyError(stage)


def _row(stage, chosen, dev, test, delta, candidates) -> dict:
    return {
        "stage": stage,
        "chosen": chosen,
        "dev_ndcg@10": dev["ndcg@10"],
        "test_ndcg@10": test["ndcg@10"],
        "delta_test_ndcg@10": delta,
        "test_latency_ms": test["latency_ms"],
        "estimated_usd": test["estimated_usd"],
        "candidates": candidates,
    }


def _spec_label(spec: PipelineSpec) -> str:
    return f"{spec.chunking}/{spec.query_transform}/{spec.retrieval}/{spec.rerank}"


def _load_splits(config: AblationConfig):
    if config.dataset == "scifact":
        dev_data = load_scifact(config.data_dir, split=config.dev_split or "train")
        test_data = load_scifact(config.data_dir, split=config.test_split or "test")
        documents = [doc for doc in test_data.documents if doc.full_text.strip()]
        dev_queries = select_queries(
            dev_data.queries, dev_data.qrels, config.dev_max_queries, config.seed
        )
        test_queries = select_queries(
            test_data.queries, test_data.qrels, config.test_max_queries, config.seed + 1
        )
        dev_meta = _meta(config.dev_split or "train", dev_queries, _labeled_count(dev_data))
        test_meta = _meta(config.test_split or "test", test_queries, _labeled_count(test_data))
        return (
            dev_queries,
            test_queries,
            dev_data.qrels,
            test_data.qrels,
            documents,
            dev_meta,
            test_meta,
        )
    dataset = load_dataset(config.dataset, config.data_dir, config.fixture_path)
    documents = [doc for doc in dataset.documents if doc.full_text.strip()]
    labeled = select_queries(dataset.queries, dataset.qrels, None, config.seed)
    import random

    order = list(labeled)
    random.Random(config.seed).shuffle(order)
    dev_count = config.dev_count if config.dev_count is not None else max(1, len(labeled) // 2)
    if dev_count >= len(labeled):
        raise ConfigError("dev_count must leave at least one held-out test query")
    dev_queries = sorted(order[:dev_count], key=lambda query: query.query_id)
    test_queries = sorted(order[dev_count:], key=lambda query: query.query_id)
    dev_meta = _meta("seeded subset of the labeled queries", dev_queries, len(labeled))
    test_meta = _meta("held-out remainder", test_queries, len(labeled))
    return dev_queries, test_queries, dataset.qrels, dataset.qrels, documents, dev_meta, test_meta


def _labeled_count(dataset: Dataset) -> int:
    return len(select_queries(dataset.queries, dataset.qrels, None, 0))


def _meta(split: str, queries: list[Query], available: int) -> dict:
    return {
        "split": split,
        "num_queries": len(queries),
        "labeled_queries_available": available,
        "query_ids": [query.query_id for query in queries],
    }


def _load_reranker(config: AblationConfig):
    if config.reranker_kind == "lexical":
        return LexicalReranker()
    if config.reranker_kind == "fastembed":
        logger.info("Loading reranker %s", config.reranker_model)
        return FastEmbedReranker(config.reranker_model, batch_size=config.reranker_batch_size)
    raise ConfigError(f"Unknown reranker kind {config.reranker_kind!r}")


def _find_root(config_path: Path) -> Path:
    for start in (Path.cwd().resolve(), config_path.resolve().parent):
        for candidate in [start, *start.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
                return candidate
    return Path.cwd().resolve()


def _path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _optional_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    return _path(root, value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)
