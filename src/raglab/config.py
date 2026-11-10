"""YAML run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    pass


@dataclass
class ChunkerSpec:
    name: str
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelSpec:
    kind: str
    model: str = ""
    batch_size: int = 64
    dim: int = 256


@dataclass
class GenerationSpec:
    mode: str
    max_queries: int
    k: int
    pairs: list[tuple[str, str]]


@dataclass
class RunConfig:
    dataset: str
    fixture_path: Path | None
    seed: int
    max_queries: int | None
    k_values: list[int]
    hit_k: int
    fusion_pool: int
    rerank_pool: int
    rrf_k: int
    bm25_k1: float
    bm25_b: float
    embedder: ModelSpec
    reranker: ModelSpec
    chunkers: list[ChunkerSpec]
    retrievers: list[str]
    llm_retrievers: list[str]
    force_retrievers: bool
    rag_fusion_queries: int
    generation: GenerationSpec
    output_dir: Path
    data_dir: Path
    cache_dir: Path
    plugins: list[str]
    latency_queries: int
    check_path: Path | None
    root: Path

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "fixture_path": _display(self.fixture_path),
            "seed": self.seed,
            "max_queries": self.max_queries,
            "k_values": self.k_values,
            "hit_k": self.hit_k,
            "fusion_pool": self.fusion_pool,
            "rerank_pool": self.rerank_pool,
            "rrf_k": self.rrf_k,
            "bm25": {"k1": self.bm25_k1, "b": self.bm25_b},
            "embedder": {
                "kind": self.embedder.kind,
                "model": self.embedder.model,
                "batch_size": self.embedder.batch_size,
                "dim": self.embedder.dim,
            },
            "reranker": {
                "kind": self.reranker.kind,
                "model": self.reranker.model,
                "batch_size": self.reranker.batch_size,
            },
            "chunkers": [{"name": spec.name, **spec.options} for spec in self.chunkers],
            "retrievers": list(self.retrievers),
            "llm_retrievers": list(self.llm_retrievers),
            "force_retrievers": self.force_retrievers,
            "rag_fusion_queries": self.rag_fusion_queries,
            "generation": {
                "mode": self.generation.mode,
                "max_queries": self.generation.max_queries,
                "k": self.generation.k,
                "pairs": [list(pair) for pair in self.generation.pairs],
            },
            "latency_queries": self.latency_queries,
            "plugins": list(self.plugins),
        }


def load_config(path: str | Path) -> RunConfig:
    config_path = Path(path)
    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path
    if not config_path.exists():
        raise ConfigError(f"Config not found: {config_path}")
    root = find_root(config_path)
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Could not parse {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConfigError("Config root must be a mapping")
    return _build(payload, root)


def find_root(config_path: Path) -> Path:
    starts = [Path.cwd().resolve(), config_path.resolve().parent]
    seen: set[Path] = set()
    for start in starts:
        for candidate in [start, *start.parents]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "pyproject.toml").exists() and (candidate / "configs").exists():
                return candidate
    return Path.cwd().resolve()


def _build(payload: dict[str, Any], root: Path) -> RunConfig:
    dataset = str(payload.get("dataset", "scifact"))
    k_values = [int(value) for value in payload.get("k_values", [5, 10])]
    if not k_values or any(value < 1 for value in k_values):
        raise ConfigError("k_values must be positive integers")
    k_values = sorted(set(k_values))
    hit_k = int(payload.get("hit_k", max(k_values)))
    if hit_k < max(k_values):
        raise ConfigError("hit_k must be at least the largest k")
    chunkers = _chunkers(payload.get("chunkers"))
    names = [spec.name for spec in chunkers]
    if len(names) != len(set(names)):
        raise ConfigError("chunker names must be unique")
    retrievers = _name_list(payload.get("retrievers"), "retrievers")
    llm_retrievers = _name_list(payload.get("llm_retrievers", []), "llm_retrievers")
    generation = _generation(payload.get("generation") or {})
    for chunker_name, _retriever_name in generation.pairs:
        if chunker_name not in names:
            raise ConfigError(f"generation pair uses unknown chunker '{chunker_name}'")
    bm25 = payload.get("bm25") or {}
    embedder = _model(payload.get("embedder") or {}, "embedder", default_kind="fastembed")
    reranker = _model(payload.get("reranker") or {}, "reranker", default_kind="fastembed")
    fixture = payload.get("fixture_path")
    return RunConfig(
        dataset=dataset,
        fixture_path=_optional_path(root, fixture),
        seed=int(payload.get("seed", 7)),
        max_queries=_optional_int(payload.get("max_queries")),
        k_values=k_values,
        hit_k=hit_k,
        fusion_pool=int(payload.get("fusion_pool", hit_k)),
        rerank_pool=int(payload.get("rerank_pool", 30)),
        rrf_k=int(payload.get("rrf_k", 60)),
        bm25_k1=float(bm25.get("k1", 1.5)),
        bm25_b=float(bm25.get("b", 0.75)),
        embedder=embedder,
        reranker=reranker,
        chunkers=chunkers,
        retrievers=retrievers,
        llm_retrievers=llm_retrievers,
        force_retrievers=False,
        rag_fusion_queries=int(payload.get("rag_fusion_queries", 3)),
        generation=generation,
        output_dir=_path(root, payload.get("output_dir", "results/run")),
        data_dir=_path(root, payload.get("data_dir", "data")),
        cache_dir=_path(root, payload.get("cache_dir", "data/cache")),
        plugins=[str(item) for item in payload.get("plugins") or []],
        latency_queries=int(payload.get("latency_queries", 20)),
        check_path=None,
        root=root,
    )


def _chunkers(value: Any) -> list[ChunkerSpec]:
    if not isinstance(value, list) or not value:
        raise ConfigError("chunkers must be a non-empty list")
    specs = []
    for item in value:
        if isinstance(item, str):
            specs.append(ChunkerSpec(name=item))
            continue
        if not isinstance(item, dict) or "name" not in item:
            raise ConfigError("each chunker needs a name")
        options = {key: val for key, val in item.items() if key != "name"}
        specs.append(ChunkerSpec(name=str(item["name"]), options=options))
    return specs


def _generation(value: dict[str, Any]) -> GenerationSpec:
    # YAML 1.1 reads bare on/off as booleans. Accept those and the words.
    raw_mode = value.get("mode", "auto")
    if raw_mode is True:
        mode = "on"
    elif raw_mode is False:
        mode = "off"
    else:
        mode = str(raw_mode)
    if mode not in {"auto", "on", "off"}:
        raise ConfigError("generation.mode must be auto, on, or off")
    pairs = []
    for item in value.get("pairs") or [["recursive", "hybrid"]]:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ConfigError("generation pairs must be [chunker, retriever]")
        pairs.append((str(item[0]), str(item[1])))
    if not pairs:
        raise ConfigError("generation.pairs must not be empty")
    return GenerationSpec(
        mode=mode,
        max_queries=int(value.get("max_queries", 20)),
        k=int(value.get("k", 5)),
        pairs=pairs,
    )


def _model(value: dict[str, Any], label: str, default_kind: str) -> ModelSpec:
    if not isinstance(value, dict):
        raise ConfigError(f"{label} must be a mapping")
    return ModelSpec(
        kind=str(value.get("kind", default_kind)),
        model=str(value.get("model", "")),
        batch_size=int(value.get("batch_size", 64)),
        dim=int(value.get("dim", 256)),
    )


def _name_list(value: Any, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ConfigError(f"{label} must be a list of names")
    return list(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    number = int(value)
    if number < 1:
        raise ConfigError("max_queries must be >= 1")
    return number


def _path(root: Path, value: Any) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else root / path


def _optional_path(root: Path, value: Any) -> Path | None:
    if value is None:
        return None
    return _path(root, value)


def _display(path: Path | None) -> str | None:
    return None if path is None else str(path)
