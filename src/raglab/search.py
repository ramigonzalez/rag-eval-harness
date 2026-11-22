"""Stage-wise search over chunking and retrieval for any labeled corpus.

Stage 1 scores every chunk size and overlap with one retriever. Stage 2 runs
the other retrievers on the best config of each chunker family. Stage 3 adds a
per-source merge. ``mode: full`` crosses the whole grid instead.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

from raglab.chunking.fixed import FixedChunker
from raglab.chunking.markdown import MarkdownChunker
from raglab.chunking.recursive import RecursiveChunker
from raglab.chunking.semantic import SemanticChunker
from raglab.embeddings import HashEmbedder, load_embedder
from raglab.gold import score_query
from raglab.ingest import (
    Block,
    IngestError,
    blocks_from_path,
    fetch_bytes,
    load_manifest,
    select_sources,
)
from raglab.lang_pt import tokenize_for_language
from raglab.llm import LLMClient, llm_configured
from raglab.retrieval.fusion import reciprocal_rank_fusion
from raglab.retrieval.index import SearchIndex, Services, build_search_index, hits_from_id_scores
from raglab.retrieval.strategies import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    HyDERetriever,
    RAGFusionRetriever,
)
from raglab.search_report import write_search_outputs
from raglab.textutil import word_count
from raglab.types import Chunk, Document, GoldLabel, Hit, Query

logger = logging.getLogger("raglab")

RETRIEVERS = ("bm25", "dense", "hybrid", "hyde", "rag_fusion")
LLM_RETRIEVERS = {"hyde", "rag_fusion"}
DENSE_RETRIEVERS = {"dense", "hybrid", "hyde", "rag_fusion"}


@dataclass(frozen=True)
class ChunkSpec:
    chunker: str
    chunk_size: int
    overlap: float
    cross_page: bool = False

    @property
    def key(self) -> str:
        base = f"{self.chunker}:{self.chunk_size}:overlap={self.overlap:.2f}"
        if self.cross_page:
            return f"{base}:cross-page"
        return base


@dataclass(frozen=True)
class RunSpec:
    chunk: ChunkSpec
    retriever: str
    per_source: bool
    stage: str

    @property
    def config_id(self) -> str:
        scope = "per-source" if self.per_source else "corpus"
        return f"{self.chunk.key}|{self.retriever}|{scope}"


@dataclass
class SearchConfig:
    name: str
    manifest_url: str
    manifest_path: Path | None
    roles: list[str]
    eval_url: str
    eval_path: Path | None
    language: str
    page_tolerance: int
    gold_match: str
    any_source: bool
    embedder_kind: str
    embedder_model: str
    query_prefix: str
    passage_prefix: str
    cache_dir: Path
    llm_cache_dir: Path
    output_dir: Path
    mode: str
    stage1_retriever: str
    top_k_chunking: int
    chunkers: list[str]
    chunk_sizes: list[int]
    overlaps: list[float]
    retrievers: list[str]
    bootstrap_samples: int
    seed: int
    k_values: list[int]
    hit_k: int
    cross_page: bool
    word_budget: int
    top_k_per_family: int


def load_search_config(path: Path) -> SearchConfig:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise IngestError(f"{path} must be a mapping")
    root = path.resolve().parent
    corpus = payload.get("corpus") or {}
    evaluation = payload.get("eval") or {}
    embedder = payload.get("embedder") or {}
    search = payload.get("search") or {}
    gold = search.get("gold") or {}

    def beside_config(value: str | None) -> Path | None:
        if not value:
            return None
        candidate = Path(value)
        return candidate if candidate.is_absolute() else root / candidate

    def from_cwd(value: str) -> Path:
        candidate = Path(value)
        return candidate if candidate.is_absolute() else Path.cwd() / candidate

    return SearchConfig(
        name=str(payload.get("name") or "search"),
        manifest_url=str(corpus.get("manifest_url") or ""),
        manifest_path=beside_config(corpus.get("manifest")),
        roles=list(corpus.get("roles") or ["core"]),
        eval_url=str(evaluation.get("url") or ""),
        eval_path=beside_config(evaluation.get("path")),
        language=str(payload.get("language") or "en"),
        page_tolerance=int(gold.get("page_tolerance", 0)),
        gold_match=str(gold.get("match") or "source_page"),
        any_source=bool(gold.get("any_source", False)),
        embedder_kind=str(embedder.get("kind") or "fastembed"),
        embedder_model=str(embedder.get("model") or "intfloat/multilingual-e5-large"),
        query_prefix=str(embedder.get("query_prefix") or ""),
        passage_prefix=str(embedder.get("passage_prefix") or ""),
        cache_dir=from_cwd(str(payload.get("cache_dir") or "data/cache")),
        llm_cache_dir=from_cwd(
            str((payload.get("llm") or {}).get("cache_dir") or "data/cache/llm")
        ),
        output_dir=from_cwd(str(payload.get("output_dir") or "results/search")),
        mode=str(search.get("mode") or "staged"),
        stage1_retriever=str(search.get("stage1_retriever") or "bm25"),
        top_k_chunking=int(search.get("top_k_chunking") or 3),
        chunkers=list(search.get("chunkers") or ["markdown", "recursive", "fixed"]),
        chunk_sizes=[int(value) for value in search.get("chunk_sizes") or [256, 512, 1024]],
        overlaps=[float(value) for value in search.get("overlaps") or [0.0, 0.1, 0.2]],
        retrievers=list(search.get("retrievers") or list(RETRIEVERS)),
        bootstrap_samples=int(search.get("bootstrap_samples") or 1000),
        seed=int(search.get("seed") or 7),
        k_values=[int(value) for value in search.get("k_values") or [5, 10]],
        hit_k=int(search.get("hit_k") or 60),
        cross_page=bool(search.get("cross_page", False)),
        word_budget=int(2000 if search.get("word_budget") is None else search.get("word_budget")),
        top_k_per_family=int(search.get("top_k_per_family") or 1),
    )


def chunk_specs(config: SearchConfig) -> list[ChunkSpec]:
    specs = []
    for name in config.chunkers:
        # Semantic splits do not use a word overlap, so the grid keeps one config per size.
        overlaps = [0.0] if name == "semantic" else config.overlaps
        for size in config.chunk_sizes:
            for overlap in overlaps:
                specs.append(ChunkSpec(name, size, overlap, config.cross_page and name == "fixed"))
    return specs


def _stage2_chunkings(config: SearchConfig) -> int:
    specs = chunk_specs(config)
    per_family = max(1, config.top_k_per_family)
    available = {}
    for spec in specs:
        available[spec.chunker] = available.get(spec.chunker, 0) + 1
    return sum(min(per_family, count) for count in available.values())


def planned_count(config: SearchConfig) -> int:
    chunks = len(chunk_specs(config))
    if config.mode == "full":
        return chunks * len(config.retrievers)
    others = [name for name in config.retrievers if name != config.stage1_retriever]
    return chunks + _stage2_chunkings(config) * len(others) + 2


def describe_plan(config: SearchConfig) -> str:
    chunks = len(chunk_specs(config))
    if config.mode == "full":
        total = planned_count(config)
        return (
            f"full grid: {chunks} chunking configs x {len(config.retrievers)} retrievers "
            f"= {total} configs"
        )
    others = [name for name in config.retrievers if name != config.stage1_retriever]
    families = len({spec.chunker for spec in chunk_specs(config)})
    kept = _stage2_chunkings(config)
    stage2 = kept * len(others)
    return (
        f"staged: {chunks} chunking configs with {config.stage1_retriever}, "
        f"then {config.top_k_per_family} best from each of {families} families "
        f"({kept} chunkings x {len(others)} retrievers = {stage2}), "
        f"then 2 per-source combines = {planned_count(config)} configs"
    )


def stage1_runs(config: SearchConfig) -> list[RunSpec]:
    if config.mode == "full":
        return [
            RunSpec(chunk, retriever, False, "full")
            for chunk in chunk_specs(config)
            for retriever in config.retrievers
        ]
    return [
        RunSpec(chunk, config.stage1_retriever, False, "stage1") for chunk in chunk_specs(config)
    ]


def stage2_runs(config: SearchConfig, stage1_rows: list[dict]) -> list[RunSpec]:
    if config.mode == "full":
        return []
    winners = _top_chunks(config, stage1_rows)
    others = [name for name in config.retrievers if name != config.stage1_retriever]
    return [RunSpec(chunk, retriever, False, "stage2") for chunk in winners for retriever in others]


def stage3_runs(config: SearchConfig, rows: list[dict]) -> list[RunSpec]:
    if config.mode == "full":
        return []
    scored = [row for row in rows if row.get("status") == "ok" and not row["per_source"]]
    if not scored:
        return []
    best = min(scored, key=_quality_key)
    return _stage3(_spec_from_row(best), rows)


def _spec_from_row(row: dict) -> ChunkSpec:
    return ChunkSpec(
        row["chunker"],
        int(row["chunk_size"]),
        float(row["overlap"]),
        bool(row.get("cross_page")),
    )


def _top_chunks(config: SearchConfig, rows: list[dict]) -> list[ChunkSpec]:
    """Best configs of each chunker family, so stage 2 is not only the global top few."""
    grouped: dict[str, list[dict]] = {}
    for row in sorted(rows, key=_quality_key):
        if row.get("status") != "ok" or row.get("per_source"):
            continue
        grouped.setdefault(row["chunker"], []).append(row)
    winners: list[ChunkSpec] = []
    seen: set[str] = set()
    for family in config.chunkers:
        kept = 0
        for row in grouped.get(family, []):
            if row["chunk_key"] in seen:
                continue
            seen.add(row["chunk_key"])
            winners.append(_spec_from_row(row))
            kept += 1
            if kept >= config.top_k_per_family:
                break
    return winners


def _stage3(chunk: ChunkSpec, rows: list[dict]) -> list[RunSpec]:
    scored = [
        row
        for row in rows
        if row.get("status") == "ok" and row["chunk_key"] == chunk.key and not row["per_source"]
    ]
    winner = min(scored, key=_quality_key)["retriever"] if scored else "bm25"
    names = [winner]
    for extra in ("hybrid", "bm25"):
        if extra not in names:
            names.append(extra)
        if len(names) == 2:
            break
    # HyDE and RAG-Fusion stay corpus-level. The combine step is per-source BM25/dense/hybrid.
    concrete = []
    for name in names:
        if name in LLM_RETRIEVERS:
            continue
        concrete.append(name)
    if "hybrid" not in concrete:
        concrete.append("hybrid")
    if "bm25" not in concrete:
        concrete.append("bm25")
    return [RunSpec(chunk, name, True, "stage3") for name in concrete[:2]]


def _quality_key(row: dict):
    ndcg = row.get("ndcg@10")
    ndcg_value = -1.0 if ndcg is None else -float(ndcg)
    latency = row.get("latency_ms")
    latency_value = 1e18 if latency is None else float(latency)
    return (ndcg_value, latency_value, row["config_id"])


class PrefixedEmbedder:
    """Add E5 query and passage prefixes. SciFact does not use this wrapper."""

    def __init__(self, inner, query_prefix: str, passage_prefix: str) -> None:
        self.inner = inner
        self.model_name = inner.model_name
        self.dim = inner.dim
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix

    def embed_documents(self, texts: list[str]):
        return self.inner.embed_documents([self.passage_prefix + text for text in texts])

    def embed_queries(self, texts: list[str]):
        return self.inner.embed_queries([self.query_prefix + text for text in texts])

    def embed_query(self, text: str):
        return self.embed_queries([text])[0]


def make_embedder(config: SearchConfig):
    inner = load_embedder(
        config.embedder_kind,
        config.embedder_model,
        32,
        256,
        config.cache_dir,
    )
    if config.query_prefix or config.passage_prefix:
        return PrefixedEmbedder(inner, config.query_prefix, config.passage_prefix)
    return inner


def chunk_blocks(blocks: list[Block], spec: ChunkSpec, embedder=None) -> list[Chunk]:
    overlap = int(round(spec.chunk_size * spec.overlap))
    if overlap >= spec.chunk_size:
        overlap = max(0, spec.chunk_size - 1)
    chunks: list[Chunk] = []
    grouped: dict[str, list[Block]] = {}
    for block in blocks:
        grouped.setdefault(block.source_id, []).append(block)
    semantic = embedder or HashEmbedder(dim=32)
    for source_id, source_blocks in grouped.items():
        if spec.chunker == "fixed" and spec.cross_page:
            chunks.extend(
                _fixed_across_pages(source_id, source_blocks, spec.chunk_size, overlap, len(chunks))
            )
            continue
        for block in source_blocks:
            parts = _split_page(spec, block.text, overlap, semantic)
            for part in parts:
                chunks.append(_make_chunk(source_id, part, block.page, block.page, len(chunks)))
    if not chunks:
        raise IngestError(f"{spec.key} produced no chunks")
    return chunks


def _split_page(spec: ChunkSpec, text: str, overlap: int, embedder) -> list[str]:
    if spec.chunker == "markdown":
        return _pack_sections(MarkdownChunker().split_text(text), spec.chunk_size, overlap)
    if spec.chunker == "recursive":
        return RecursiveChunker(spec.chunk_size, overlap).split_text(text)
    if spec.chunker == "fixed":
        return FixedChunker(spec.chunk_size, overlap).split_text(text)
    if spec.chunker == "semantic":
        return _semantic_parts(text, spec.chunk_size, embedder)
    raise KeyError(spec.chunker)


def _pack_sections(sections: list[str], size: int, overlap: int) -> list[str]:
    """Merge short heading sections up to the target size, then window a long one."""
    packed: list[str] = []
    buffer = ""
    window = FixedChunker(size, overlap)

    def flush() -> None:
        nonlocal buffer
        if buffer.strip():
            packed.append(buffer.strip())
        buffer = ""

    for section in sections:
        words = word_count(section)
        if words > size:
            flush()
            packed.extend(window.split_text(section))
            continue
        if buffer and word_count(buffer) + words > size:
            flush()
        buffer = f"{buffer}\n\n{section}".strip() if buffer else section
    flush()
    return packed


def _semantic_parts(text: str, size: int, embedder) -> list[str]:
    """Split one page with the existing semantic chunker and a local hash embedder."""
    if not text.strip():
        return []
    minimum = min(40, size)
    chunker = SemanticChunker(
        breakpoint_percentile=95,
        max_chunk_words=size,
        min_chunk_words=minimum,
        embedder=embedder,
    )
    document = Document(doc_id="page", title="", text=text)
    return [
        chunk.text for chunk in chunker.chunk_corpus([document], embedder) if chunk.text.strip()
    ]


def _fixed_across_pages(
    source_id: str, blocks: list[Block], size: int, overlap: int, index_start: int
) -> list[Chunk]:
    stream: list[tuple[str, int]] = []
    for block in blocks:
        for token in block.text.split():
            stream.append((token, block.page))
    if not stream:
        return []
    step = max(1, size - overlap)
    chunks = []
    start = 0
    while start < len(stream):
        end = min(start + size, len(stream))
        window = stream[start:end]
        text = " ".join(token for token, _page in window)
        pages = [page for _token, page in window]
        chunks.append(
            _make_chunk(source_id, text, min(pages), max(pages), index_start + len(chunks))
        )
        if end == len(stream):
            break
        start += step
    return chunks


def _make_chunk(source_id: str, text: str, page_start: int, page_end: int, index: int) -> Chunk:
    return Chunk(
        chunk_id=f"{source_id}::p{page_start}-{page_end}::c{index}",
        doc_id=source_id,
        text=text,
        index=index,
        source_id=source_id,
        page_start=page_start,
        page_end=page_end,
    )


def load_blocks(config: SearchConfig) -> list[Block]:
    if config.manifest_path is not None:
        manifest = config.manifest_path
    elif config.manifest_url:
        manifest = config.cache_dir / "manifest.yaml"
        fetch_bytes(config.manifest_url, manifest)
    else:
        raise IngestError("The search config needs corpus.manifest or corpus.manifest_url")
    specs = select_sources(load_manifest(manifest), config.roles)
    if not specs:
        raise IngestError(f"No downloadable sources in {manifest} for roles {config.roles}")
    blocks: list[Block] = []
    for spec in specs:
        path = _source_path(config, manifest, spec)
        blocks.extend(blocks_from_path(path, spec.source_id, spec.title))
    if not blocks:
        raise IngestError("Ingest produced no text blocks")
    return blocks


def _source_path(config: SearchConfig, manifest: Path, spec) -> Path:
    filename = Path(spec.filename).name
    local = manifest.parent / filename
    if local.exists():
        return local
    if spec.url.startswith(("http://", "https://")):
        dest = config.cache_dir / "sources" / filename
        fetch_bytes(spec.url, dest, spec.sha256)
        return dest
    if spec.url:
        return Path(spec.url)
    raise IngestError(f"No file for source {spec.source_id} ({filename})")


def load_queries(config: SearchConfig) -> list[Query]:
    if config.eval_path is not None:
        payload = yaml.safe_load(config.eval_path.read_text(encoding="utf-8")) or {}
    elif config.eval_url:
        dest = config.cache_dir / "eval.yaml"
        fetch_bytes(config.eval_url, dest)
        payload = yaml.safe_load(dest.read_text(encoding="utf-8")) or {}
    else:
        raise IngestError("The search config needs eval.path or eval.url")
    queries = []
    for row in payload.get("questions") or []:
        labels = []
        for gold in row.get("gold") or []:
            page = gold.get("page", gold.get("pdf_page"))
            labels.append(
                GoldLabel(
                    source_id=str(gold.get("source_id") or ""),
                    page=int(page) if page is not None else None,
                    chunk_id=str(gold.get("chunk_id") or ""),
                    text_id=str(gold.get("text_id") or ""),
                )
            )
        queries.append(
            Query(
                query_id=str(row.get("id") or row.get("query_id")),
                text=str(row.get("question") or row.get("text") or ""),
                gold=tuple(labels),
                answerable=bool(row.get("answerable", True)),
            )
        )
    if not queries:
        raise IngestError("The eval file has no questions")
    return queries


def _tokenizer(config: SearchConfig):
    def tokenize(text: str) -> list[str]:
        return tokenize_for_language(text, config.language)

    return tokenize


def _index_for(
    spec: RunSpec,
    chunks: list[Chunk],
    services: Services,
    config: SearchConfig,
    cache: dict[tuple[str, bool], SearchIndex],
) -> SearchIndex:
    need_dense = spec.retriever in DENSE_RETRIEVERS
    key = (spec.chunk.key, need_dense)
    cached = cache.get(key)
    if cached is not None:
        return cached
    if need_dense and services.embedder is None:
        logger.info("Loading embedder %s", config.embedder_model or config.embedder_kind)
        services.embedder = make_embedder(config)
    index = build_search_index(
        chunks,
        services.embedder if need_dense else None,
        build_dense=need_dense,
        build_bm25=True,
        bm25_k1=1.5,
        bm25_b=0.75,
        tokenizer=_tokenizer(config),
    )
    cache[key] = index
    return index


def retrieve_query(
    index: SearchIndex,
    text: str,
    spec: RunSpec,
    services: Services,
    config: SearchConfig,
) -> list[Hit]:
    tokenizer = _tokenizer(config)
    k = max(config.hit_k, max(config.k_values))
    if spec.per_source:
        return _per_source(index, text, spec.retriever, tokenizer, k, services)
    if spec.retriever == "bm25":
        return BM25Retriever(tokenizer).retrieve(index, text, k, services)
    if spec.retriever == "dense":
        return DenseRetriever().retrieve(index, text, k, services)
    if spec.retriever == "hybrid":
        return HybridRetriever(tokenizer).retrieve(index, text, k, services)
    if spec.retriever == "hyde":
        return HyDERetriever().retrieve(index, text, k, services)
    if spec.retriever == "rag_fusion":
        return RAGFusionRetriever().retrieve(index, text, k, services)
    raise KeyError(spec.retriever)


def _per_source(index, text, kind, tokenizer, k, services: Services) -> list[Hit]:
    if kind not in {"bm25", "dense", "hybrid"}:
        raise KeyError(f"per-source merge is not defined for {kind}")
    groups: dict[str, list[int]] = {}
    for position, chunk in enumerate(index.chunks):
        groups.setdefault(chunk.source_id or chunk.doc_id, []).append(position)
    bm25_scores = index.bm25.score_all(tokenizer(text)) if kind in {"bm25", "hybrid"} else None
    dense_scores = _dense_scores(index, text, services) if kind in {"dense", "hybrid"} else None
    lists: list[list[str]] = []
    for positions in groups.values():
        if kind == "hybrid":
            sparse = _ids_at(index, bm25_scores, positions, k)
            dense = _ids_at(index, dense_scores, positions, k)
            fused = reciprocal_rank_fusion([sparse, dense], k=services.rrf_k)
            lists.append([chunk_id for chunk_id, _score in fused])
        else:
            scores = bm25_scores if kind == "bm25" else dense_scores
            lists.append(_ids_at(index, scores, positions, k))
    fused = reciprocal_rank_fusion([item for item in lists if item], k=services.rrf_k)
    return hits_from_id_scores(index, fused, k)


def _dense_scores(index: SearchIndex, text: str, services: Services):
    import numpy as np

    if index.embeddings is None or services.embedder is None:
        raise RuntimeError("dense retrieval needs embeddings")
    matrix = np.asarray(services.embedder.embed_queries([text]), dtype=np.float32)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
    return (matrix @ index.embeddings.T)[0]


def _ids_at(index: SearchIndex, scores, positions: list[int], k: int) -> list[str]:
    ordered = sorted(positions, key=lambda pos: (-float(scores[pos]), index.chunk_ids[pos]))
    return [index.chunk_ids[pos] for pos in ordered[:k]]


def score_run(
    index: SearchIndex,
    queries: list[Query],
    spec: RunSpec,
    services: Services,
    config: SearchConfig,
) -> dict:
    per_ndcg: list[float] = []
    query_ids: list[str] = []
    buckets: dict[str, list[float]] = {}
    started = time.perf_counter()
    for query in queries:
        hits = retrieve_query(index, query.text, spec, services, config)
        primary = score_query(
            hits,
            query,
            tolerance=config.page_tolerance,
            any_source=config.any_source,
            match=config.gold_match,
            k_values=config.k_values,
        )
        strict = score_query(
            hits,
            query,
            tolerance=config.page_tolerance,
            any_source=False,
            match=config.gold_match,
            k_values=config.k_values,
        )
        broad = score_query(
            hits,
            query,
            tolerance=config.page_tolerance,
            any_source=True,
            match=config.gold_match,
            k_values=config.k_values,
        )
        if primary is None:
            continue
        query_ids.append(query.query_id)
        per_ndcg.append(primary.get("ndcg@10", primary[f"ndcg@{max(config.k_values)}"]))
        _collect(buckets, primary, "")
        if strict is not None:
            _collect(buckets, strict, "source_page_")
        if broad is not None:
            _collect(buckets, broad, "any_source_")
        top = hits[:10]
        buckets.setdefault("mean_words@10", []).append(
            float(sum(word_count(hit.text) for hit in top))
        )
        if config.word_budget > 0:
            trimmed = _hits_within_budget(hits, config.word_budget)
            budget_scores = score_query(
                trimmed,
                query,
                tolerance=config.page_tolerance,
                any_source=config.any_source,
                match=config.gold_match,
                k_values=config.k_values,
            )
            if budget_scores is not None:
                for k in config.k_values:
                    buckets.setdefault(f"recall_budget@{k}", []).append(
                        budget_scores[f"recall@{k}"]
                    )
    elapsed_ms = (time.perf_counter() - started) / max(len(queries), 1) * 1000.0
    row = _base_row(spec, "ok", "")
    row["latency_ms"] = elapsed_ms
    row["n_queries"] = len(per_ndcg)
    row["query_ids"] = query_ids
    row["per_query_ndcg@10"] = per_ndcg
    row["mean_chunk_words"] = sum(word_count(chunk.text) for chunk in index.chunks) / len(
        index.chunks
    )
    for key, values in buckets.items():
        row[key] = sum(values) / len(values) if values else None
    return row


def _hits_within_budget(hits: list[Hit], budget: int) -> list[Hit]:
    """Keep a prefix of hits whose combined words stay inside the budget.

    A hit that would cross the budget is left out, including a first hit that is
    already longer than the budget. That keeps a page-spanning window from
    winning a recall number by carrying more text than the other configs.
    """
    kept: list[Hit] = []
    words = 0
    for hit in hits:
        count = word_count(hit.text)
        if words + count > budget:
            break
        kept.append(hit)
        words += count
    return kept


def _collect(buckets: dict[str, list[float]], scores: dict[str, float], prefix: str) -> None:
    for key, value in scores.items():
        if prefix:
            metric, cutoff = key.split("@", 1)
            name = f"{metric}_{prefix.rstrip('_')}@{cutoff}"
        else:
            name = key
        buckets.setdefault(name, []).append(value)


def _base_row(spec: RunSpec, status: str, reason: str) -> dict:
    return {
        "config_id": spec.config_id,
        "stage": spec.stage,
        "chunker": spec.chunk.chunker,
        "chunk_size": spec.chunk.chunk_size,
        "overlap": spec.chunk.overlap,
        "chunk_key": spec.chunk.key,
        "retriever": spec.retriever,
        "per_source": spec.per_source,
        "cross_page": spec.chunk.cross_page,
        "status": status,
        "reason": reason,
        "latency_ms": None,
        "n_queries": 0,
        "ndcg_ci_low": None,
        "ndcg_ci_high": None,
        "vs_best": "",
        "query_ids": [],
        "per_query_ndcg@10": [],
    }


def bootstrap_ci(
    candidate: list[float], baseline: list[float], samples: int, seed: int
) -> tuple[float | None, float | None]:
    if not candidate or len(candidate) != len(baseline):
        return None, None
    import numpy as np

    rng = np.random.default_rng(seed)
    left = np.asarray(candidate, dtype=np.float64)
    right = np.asarray(baseline, dtype=np.float64)
    draws = rng.integers(0, len(left), size=(samples, len(left)))
    diffs = left[draws].mean(axis=1) - right[draws].mean(axis=1)
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def attach_intervals(rows: list[dict], samples: int, seed: int) -> None:
    scored = [row for row in rows if row.get("status") == "ok" and row.get("per_query_ndcg@10")]
    if not scored:
        return
    best = min(scored, key=_quality_key)
    for row in scored:
        low, high = bootstrap_ci(row["per_query_ndcg@10"], best["per_query_ndcg@10"], samples, seed)
        row["ndcg_ci_low"] = low
        row["ndcg_ci_high"] = high
        row["vs_best"] = best["config_id"]


def run_search(
    config: SearchConfig, *, services: Services | None = None, dry_run: bool = False
) -> dict:
    plan = describe_plan(config)
    if dry_run:
        return {"plan": plan, "count": planned_count(config), "rows": []}
    create_llm = services is None
    if services is None:
        services = Services(rrf_k=60, fusion_pool=max(config.hit_k, 60), rag_fusion_queries=3)
    else:
        services.fusion_pool = max(services.fusion_pool, config.hit_k)
    blocks = load_blocks(config)
    queries = load_queries(config)
    logger.info("%s", plan)
    logger.info("%s blocks, %s questions", len(blocks), len(queries))
    chunk_cache: dict[str, list[Chunk]] = {}
    index_cache: dict[tuple[str, bool], SearchIndex] = {}
    rows: list[dict] = []
    first = stage1_runs(config)
    rows.extend(
        _execute(first, blocks, queries, services, config, chunk_cache, index_cache, create_llm)
    )
    rows.extend(
        _execute(
            stage2_runs(config, rows),
            blocks,
            queries,
            services,
            config,
            chunk_cache,
            index_cache,
            create_llm,
        )
    )
    rows.extend(
        _execute(
            stage3_runs(config, rows),
            blocks,
            queries,
            services,
            config,
            chunk_cache,
            index_cache,
            create_llm,
        )
    )
    attach_intervals(rows, config.bootstrap_samples, config.seed)
    write_search_outputs(config.output_dir, rows, plan)
    return {"plan": plan, "count": len(rows), "rows": rows, "output_dir": str(config.output_dir)}


def _execute(
    runs, blocks, queries, services, config, chunk_cache, index_cache, create_llm: bool
) -> list[dict]:
    rows = []
    seen = set()
    semantic_embedder = HashEmbedder(dim=32)
    for spec in runs:
        if spec.config_id in seen:
            continue
        seen.add(spec.config_id)
        logger.info("Scoring %s", spec.config_id)
        if spec.retriever in LLM_RETRIEVERS and services.llm is None:
            if create_llm and llm_configured():
                services.llm = LLMClient()
                services.llm.set_disk_cache(config.llm_cache_dir)
            else:
                rows.append(_base_row(spec, "skipped", "needs OPENROUTER_API_KEY"))
                continue
        chunks = chunk_cache.get(spec.chunk.key)
        if chunks is None:
            chunks = chunk_blocks(blocks, spec.chunk, semantic_embedder)
            chunk_cache[spec.chunk.key] = chunks
        index = _index_for(spec, chunks, services, config, index_cache)
        rows.append(score_run(index, queries, spec, services, config))
    return rows
