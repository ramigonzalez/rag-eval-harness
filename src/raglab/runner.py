"""Run the chunker by retriever grid and write the result files."""

from __future__ import annotations

import importlib
import logging
import os
import platform
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from raglab import __version__
from raglab.chunking import build_chunker
from raglab.config import ConfigError, RunConfig
from raglab.datasets import load_dataset
from raglab.embeddings import load_embedder
from raglab.generation import run_generation
from raglab.llm import SKIP_REASON, LLMClient, llm_configured
from raglab.metrics import score_rankings
from raglab.report import check_thresholds, write_outputs
from raglab.retrieval import RETRIEVERS, build_retriever
from raglab.retrieval.index import Services, build_search_index
from raglab.retrieval.rerank import FastEmbedReranker, LexicalReranker
from raglab.textutil import word_count
from raglab.types import Document, Query

logger = logging.getLogger("raglab")


def run_benchmark(config: RunConfig) -> dict:
    _import_plugins(config.plugins)
    dataset = load_dataset(config.dataset, config.data_dir, config.fixture_path)
    documents = [doc for doc in dataset.documents if doc.full_text.strip()]
    queries = select_queries(dataset.queries, dataset.qrels, config.max_queries, config.seed)
    if not documents or not queries:
        raise ConfigError("The dataset has no documents or no labeled queries to score")
    retrievers, skipped_llm = _resolve_retrievers(config)
    generation_mode = _generation_mode(config)
    needs_embeddings = any(item.needs_embeddings for item in retrievers) or any(
        spec.name == "semantic" for spec in config.chunkers
    )
    needs_bm25 = any(item.needs_bm25 for item in retrievers)
    needs_reranker = any(item.needs_reranker for item in retrievers)
    if generation_mode == "on":
        for _chunker_name, retriever_name in config.generation.pairs:
            generated = _retriever_class(retriever_name)
            needs_embeddings = needs_embeddings or generated.needs_embeddings
            needs_bm25 = needs_bm25 or generated.needs_bm25
            needs_reranker = needs_reranker or generated.needs_reranker
    needs_llm = any(item.needs_llm for item in retrievers) or generation_mode == "on"
    embedder = None
    if needs_embeddings or generation_mode == "on":
        embedder = load_embedder(
            config.embedder.kind,
            config.embedder.model,
            config.embedder.batch_size,
            config.embedder.dim,
            config.cache_dir,
        )
    reranker = _load_reranker(config) if needs_reranker else None
    llm = LLMClient() if needs_llm else None
    services = Services(
        embedder=embedder,
        reranker=reranker,
        llm=llm,
        rrf_k=config.rrf_k,
        fusion_pool=config.fusion_pool,
        rerank_pool=config.rerank_pool,
        rag_fusion_queries=config.rag_fusion_queries,
    )
    rows: list[dict] = []
    chunk_reports: list[dict] = []
    indexes = {}
    examples: list[dict] = []
    example_queries = _evenly(queries, 2)
    latency_queries = _evenly(queries, config.latency_queries)
    logger.info(
        "Scoring %s labeled queries over %s documents",
        f"{len(queries):,}",
        f"{len(documents):,}",
    )
    if skipped_llm:
        logger.info("Skipping %s because no API key is set", ", ".join(skipped_llm))

    for spec in config.chunkers:
        try:
            chunker = build_chunker(spec.name, spec.options)
        except (KeyError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc
        logger.info("Chunking with %s", spec.name)
        started = time.perf_counter()
        chunks = chunker.chunk_corpus(documents, embedder)
        chunk_seconds = time.perf_counter() - started
        if not chunks:
            raise ConfigError(f"Chunker '{spec.name}' produced no chunks")
        logger.info(
            "%s produced %s chunks in %.1fs",
            spec.name,
            f"{len(chunks):,}",
            chunk_seconds,
        )
        started = time.perf_counter()
        index = build_search_index(
            chunks,
            embedder,
            build_dense=needs_embeddings,
            build_bm25=needs_bm25,
            bm25_k1=config.bm25_k1,
            bm25_b=config.bm25_b,
        )
        index_seconds = time.perf_counter() - started
        indexes[spec.name] = index
        chunk_reports.append(
            _chunk_report(spec.name, chunks, len(documents), chunk_seconds, index_seconds)
        )
        for retriever in retrievers:
            rows.append(
                _evaluate_strategy(
                    spec.name,
                    retriever,
                    index,
                    queries,
                    dataset.qrels,
                    config,
                    services,
                    latency_queries,
                )
            )
            row = rows[-1]
            if row["status"] != "ok":
                continue
            for query in example_queries:
                hits = row["rankings"][query.query_id][:3]
                examples.append(
                    {
                        "chunker": spec.name,
                        "retriever": retriever.name,
                        "query_id": query.query_id,
                        "query": query.text,
                        "hits": [
                            {
                                "doc_id": hit.doc_id,
                                "chunk_id": hit.chunk_id,
                                "score": round(hit.score, 4),
                                "text": _snippet(hit.text),
                            }
                            for hit in hits
                        ],
                    }
                )

    for row in rows:
        row.pop("rankings", None)

    if generation_mode == "on":
        logger.info("Running generation metrics")
        generation = run_generation(
            indexes,
            queries,
            config.generation.pairs,
            services,
            k=config.generation.k,
            max_queries=config.generation.max_queries,
            seed=config.seed,
        )
    else:
        reason = SKIP_REASON if config.generation.mode == "auto" else "Generation was turned off."
        generation = {"status": "skipped", "reason": reason, "pairs": []}

    result = {
        "raglab_version": __version__,
        "dataset": _dataset_summary(dataset.name, documents, queries, dataset.qrels),
        "config": config.to_public_dict(),
        "environment": _environment(config.root),
        "chunkers": chunk_reports,
        "rows": rows,
        "skipped_retrievers": skipped_llm,
        "generation": generation,
        "examples": examples,
        "latency_queries": min(config.latency_queries, len(queries)),
        "threshold_failures": [],
    }
    if config.check_path is not None:
        result["threshold_failures"] = check_thresholds(rows, config.check_path)
    write_outputs(result, config.output_dir)
    logger.info("Wrote results to %s", config.output_dir)
    return result


def select_queries(
    queries: list[Query],
    qrels: dict[str, dict[str, int]],
    max_queries: int | None,
    seed: int,
) -> list[Query]:
    labeled = []
    for query in queries:
        grades = qrels.get(query.query_id, {})
        if query.text.strip() and any(grade > 0 for grade in grades.values()):
            labeled.append(query)
    if max_queries is not None and max_queries < len(labeled):
        import random

        rng = random.Random(seed)
        labeled = rng.sample(labeled, max_queries)
    labeled.sort(key=lambda query: query.query_id)
    return labeled


def _evaluate_strategy(
    chunker, retriever, index, queries, qrels, config, services, latency_queries
):
    logger.info("Evaluating %s / %s", chunker, retriever.name)
    retriever.fallbacks = 0
    started = time.perf_counter()
    try:
        hit_lists = retriever.retrieve_all(
            index, [query.text for query in queries], config.hit_k, services
        )
    except Exception as exc:
        logger.exception("%s / %s failed", chunker, retriever.name)
        return {
            "chunker": chunker,
            "retriever": retriever.name,
            "status": "error",
            "note": str(exc),
            "score_type": retriever.score_type,
            "num_queries": 0,
            "metrics": {},
            "eval_seconds": time.perf_counter() - started,
            "online_latency_ms": None,
            "fallbacks": retriever.fallbacks,
            "min_unique_docs": 0,
            "mean_unique_docs": 0.0,
        }
    eval_seconds = time.perf_counter() - started
    if len(hit_lists) != len(queries):
        raise RuntimeError(
            f"{retriever.name} returned {len(hit_lists)} lists for {len(queries)} queries"
        )
    rankings = {query.query_id: hits for query, hits in zip(queries, hit_lists, strict=True)}
    summary = score_rankings(rankings, qrels, config.k_values)
    fallbacks = retriever.fallbacks
    online_ms = _online_latency(retriever, index, latency_queries, config.hit_k, services)
    note = ""
    if summary["min_unique_docs"] < max(config.k_values):
        note = (
            f"some queries had fewer than {max(config.k_values)} unique documents "
            "in the retrieved chunks"
        )
        logger.warning("%s / %s: %s", chunker, retriever.name, note)
    metrics = summary["metrics"]
    logger.info(
        "%s / %s  recall@10=%s  mrr@10=%s  ndcg@10=%s  context_precision@10=%s  online=%.1fms",
        chunker,
        retriever.name,
        _num(metrics.get("recall@10")),
        _num(metrics.get("mrr@10")),
        _num(metrics.get("ndcg@10")),
        _num(metrics.get("context_precision@10")),
        online_ms or 0.0,
    )
    return {
        "chunker": chunker,
        "retriever": retriever.name,
        "status": "ok",
        "note": note,
        "score_type": retriever.score_type,
        "num_queries": summary["num_queries"],
        "skipped_queries": summary["skipped_queries"],
        "metrics": metrics,
        "eval_seconds": eval_seconds,
        "online_latency_ms": online_ms,
        "fallbacks": fallbacks,
        "min_unique_docs": summary["min_unique_docs"],
        "mean_unique_docs": summary["mean_unique_docs"],
        "rankings": rankings,
    }


def _online_latency(retriever, index, queries, k, services) -> float | None:
    if not queries:
        return None
    started = time.perf_counter()
    for query in queries:
        retriever.retrieve(index, query.text, k, services)
    return (time.perf_counter() - started) / len(queries) * 1000.0


def _resolve_retrievers(config: RunConfig):
    # Importing the package registers the built-ins. Plugins were imported first.
    import raglab.retrieval  # noqa: F401

    if config.force_retrievers:
        names = list(config.retrievers)
        for name in names:
            cls = _retriever_class(name)
            if cls.needs_llm and not llm_configured():
                from raglab.llm import LLMNotConfigured

                raise LLMNotConfigured(SKIP_REASON)
        skipped: list[str] = []
    else:
        names = list(config.retrievers)
        if llm_configured():
            for name in config.llm_retrievers:
                if name not in names:
                    names.append(name)
            skipped = []
        else:
            skipped = list(config.llm_retrievers)
    return [_retriever_instance(name) for name in names], skipped


def _retriever_class(name: str):
    if name not in RETRIEVERS:
        known = ", ".join(sorted(RETRIEVERS)) or "(none)"
        raise ConfigError(f"Unknown retriever '{name}'. Known: {known}.")
    return RETRIEVERS[name]


def _retriever_instance(name: str):
    try:
        return build_retriever(name)
    except KeyError as exc:
        raise ConfigError(str(exc)) from exc


def _generation_mode(config: RunConfig) -> str:
    mode = config.generation.mode
    if mode == "off":
        return "off"
    if mode == "on":
        if not llm_configured():
            from raglab.llm import LLMNotConfigured

            raise LLMNotConfigured(SKIP_REASON)
        return "on"
    if mode == "auto":
        return "on" if llm_configured() else "off"
    raise ConfigError(f"Unknown generation mode '{mode}'")


def _load_reranker(config: RunConfig):
    kind = config.reranker.kind
    if kind == "lexical":
        return LexicalReranker()
    if kind == "fastembed":
        if not config.reranker.model:
            raise ConfigError("fastembed reranker needs a model name")
        logger.info("Loading reranker %s", config.reranker.model)
        return FastEmbedReranker(config.reranker.model, batch_size=config.reranker.batch_size)
    raise ConfigError(f"Unknown reranker kind '{kind}'. Use 'fastembed' or 'lexical'.")


def _import_plugins(modules: list[str]) -> None:
    import raglab.chunking  # noqa: F401
    import raglab.retrieval  # noqa: F401

    for module in modules:
        logger.info("Loading plugin %s", module)
        importlib.import_module(module)


def _chunk_report(name, chunks, num_documents, chunk_seconds, index_seconds) -> dict:
    words = [word_count(chunk.text) for chunk in chunks]
    return {
        "name": name,
        "num_chunks": len(chunks),
        "avg_chunk_words": sum(words) / len(words),
        "avg_chunks_per_doc": len(chunks) / num_documents,
        "chunk_seconds": chunk_seconds,
        "index_seconds": index_seconds,
    }


def _dataset_summary(name, documents: list[Document], queries: list[Query], qrels) -> dict:
    doc_words = [word_count(document.full_text) for document in documents]
    relevant_counts = []
    for query in queries:
        relevant_counts.append(
            sum(1 for grade in qrels.get(query.query_id, {}).values() if grade > 0)
        )
    return {
        "name": name,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "mean_doc_words": sum(doc_words) / len(doc_words),
        "mean_relevant_docs": sum(relevant_counts) / len(relevant_counts),
        "query_ids": [query.query_id for query in queries],
    }


def _environment(root: Path) -> dict:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "raglab": __version__,
        "numpy": _package_version("numpy"),
        "fastembed": _package_version("fastembed"),
        "git_sha": _git_sha(root),
    }


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def _git_sha(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _evenly(items: list, count: int) -> list:
    if count <= 0 or not items:
        return []
    if len(items) <= count:
        return list(items)
    return [items[int(index * len(items) / count)] for index in range(count)]


def _snippet(text: str, limit: int = 240) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 3] + "..."


def _num(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"
