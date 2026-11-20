"""Search, ingest, Portuguese tokens, and page-level gold. No downloads and no API."""

import csv
import hashlib
import json
from pathlib import Path

import pytest

from raglab.embeddings import HashEmbedder
from raglab.gold import score_query
from raglab.ingest import (
    Block,
    IngestError,
    blocks_from_path,
    fetch_bytes,
    load_manifest,
    select_sources,
)
from raglab.lang_pt import stem_pt, tokenize_pt
from raglab.llm import LLMClient
from raglab.retrieval.index import Services
from raglab.search import (
    ChunkSpec,
    PrefixedEmbedder,
    chunk_blocks,
    describe_plan,
    load_search_config,
    planned_count,
    run_search,
)
from raglab.types import GoldLabel, Hit, Query

ROOT = Path(__file__).resolve().parents[1]


def test_portuguese_stopwords_are_removed_before_stemming(monkeypatch):
    seen = []

    def wrapped(word):
        seen.append(word)
        return stem_pt(word)

    monkeypatch.setattr("raglab.lang_pt.stem_pt", wrapped)
    tokens = tokenize_pt("que alimentações não crianças")
    assert "que" not in seen
    assert "não" not in seen
    assert "alimentações" in seen
    assert "crianças" in seen
    assert "alimentação" in tokens
    assert "criança" in tokens
    assert "que" not in tokens


def test_page_gold_tolerance_any_source_and_chunk_id():
    query = Query(
        "q",
        "leite",
        gold=(GoldLabel(source_id="guia", page=5),),
        answerable=True,
    )
    off_page = [Hit("c", "guia", "texto", 1.0, source_id="guia", page_start=6, page_end=6)]
    missed = score_query(
        off_page, query, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
    )
    near = score_query(
        off_page, query, tolerance=1, any_source=False, match="source_page", k_values=[5, 10]
    )
    assert missed["recall@5"] == 0
    assert near["recall@5"] == 1

    other = [Hit("c2", "outro", "texto", 1.0, source_id="outro", page_start=5, page_end=5)]
    strict = score_query(
        other, query, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
    )
    broad = score_query(
        other, query, tolerance=0, any_source=True, match="source_page", k_values=[5, 10]
    )
    assert strict["recall@5"] == 0
    assert broad["recall@5"] == 1

    labeled = Query("q2", "leite", gold=(GoldLabel(chunk_id="chunk-9"),), answerable=True)
    hits = [Hit("chunk-9", "guia", "texto", 1.0, source_id="guia")]
    by_id = score_query(
        hits, labeled, tolerance=0, any_source=False, match="chunk_id", k_values=[5, 10]
    )
    assert by_id["recall@5"] == 1
    assert (
        score_query(
            hits, labeled, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
        )
        is None
    )
    text_label = Query("q3", "leite", gold=(GoldLabel(text_id="chunk-9"),))
    by_text = score_query(
        hits, text_label, tolerance=0, any_source=False, match="chunk_id", k_values=[5, 10]
    )
    assert by_text["mrr@10"] == 1
    buried = [
        Hit(f"c{page}", "outro", "x", 1.0, source_id="outro", page_start=page, page_end=page)
        for page in range(1, 7)
    ]
    buried.append(Hit("gold", "guia", "y", 1.0, source_id="guia", page_start=5, page_end=5))
    late = score_query(
        buried, query, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
    )
    assert late["recall@5"] == 0
    assert late["recall@10"] == 1
    assert late["mrr@5"] == 0
    assert late["mrr@10"] == pytest.approx(1 / 7)
    repeated = [
        Hit(f"same{i}", "outro", "x", 1.0, source_id="outro", page_start=1, page_end=1)
        for i in range(6)
    ]
    repeated.append(Hit("gold", "guia", "y", 1.0, source_id="guia", page_start=5, page_end=5))
    held = score_query(
        repeated, query, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
    )
    assert held["recall@5"] == 0
    assert held["mrr@10"] == pytest.approx(1 / 7)
    wide = [Hit("wide", "outro", "x", 1.0, source_id="outro", page_start=1, page_end=8)]
    wide.append(Hit("gold", "guia", "y", 1.0, source_id="guia", page_start=5, page_end=5))
    spanned = score_query(
        wide, query, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
    )
    assert spanned["mrr@5"] == pytest.approx(0.5)
    duplicated = Query(
        "dup",
        "leite",
        gold=(GoldLabel(source_id="guia", page=3), GoldLabel(source_id="outro", page=3)),
    )
    once = [Hit("c", "guia", "texto", 1.0, source_id="guia", page_start=3, page_end=3)]
    collapsed = score_query(
        once, duplicated, tolerance=0, any_source=True, match="source_page", k_values=[5, 10]
    )
    assert collapsed["recall@5"] == 1
    assert collapsed["ndcg@10"] == 1
    both_pages = Query(
        "span",
        "leite",
        gold=(GoldLabel(source_id="guia", page=1), GoldLabel(source_id="guia", page=2)),
    )
    covered = score_query(
        [Hit("wide", "guia", "texto", 1.0, source_id="guia", page_start=1, page_end=2)],
        both_pages,
        tolerance=0,
        any_source=False,
        match="source_page",
        k_values=[5, 10],
    )
    split = score_query(
        [
            Hit("a", "guia", "texto", 1.0, source_id="guia", page_start=1, page_end=1),
            Hit("b", "guia", "texto", 1.0, source_id="guia", page_start=2, page_end=2),
        ],
        both_pages,
        tolerance=0,
        any_source=False,
        match="source_page",
        k_values=[5, 10],
    )
    assert covered["recall@5"] == 1
    assert covered["ndcg@10"] == 1
    assert split["recall@5"] == 1
    assert split["ndcg@10"] < 1
    skipped = Query("q4", "leite", gold=(GoldLabel(source_id="guia", page=1),), answerable=False)
    assert (
        score_query(
            hits, skipped, tolerance=0, any_source=False, match="source_page", k_values=[5, 10]
        )
        is None
    )


def test_ingest_markdown_html_and_text_keep_source_and_page(tmp_path):
    markdown = tmp_path / "guia.md"
    markdown.write_text("# Leite materno\n\nO leite materno exclusivo.\n", encoding="utf-8")
    html = tmp_path / "nota.html"
    html.write_text("<h2>Alergia</h2><p>Evite mel no primeiro ano.</p>", encoding="utf-8")
    plain = tmp_path / "nota.txt"
    plain.write_text("INTRODUCAO\n\nOfereca alimentos amassados.\n", encoding="utf-8")
    md_blocks = blocks_from_path(markdown, "guia", "Guia")
    html_blocks = blocks_from_path(html, "nota", "Nota")
    txt_blocks = blocks_from_path(plain, "txt", "Txt")
    assert md_blocks[0].source_id == "guia" and md_blocks[0].page == 1
    assert md_blocks[0].text.startswith("# Leite materno")
    assert html_blocks[0].text.startswith("## Alergia")
    assert html_blocks[0].source_id == "nota"
    assert txt_blocks[0].text.startswith("## INTRODUCAO")
    chunks = chunk_blocks(
        [
            Block("guia", 1, "um dois tres"),
            Block("guia", 2, "quatro cinco seis"),
        ],
        ChunkSpec("fixed", 4, 0.0),
    )
    assert chunks[0].source_id == "guia"
    assert chunks[0].page_start == 1 and chunks[0].page_end == 2
    assert "quatro" in chunks[0].text
    headed = chunk_blocks(
        [Block("guia", 3, "# Leite\n\nO leite materno exclusivo.")],
        ChunkSpec("markdown", 32, 0.0),
    )
    assert headed[0].page_start == headed[0].page_end == 3
    assert headed[0].text.startswith("# Leite")


def test_cached_download_checks_sha256_without_fetching_again(tmp_path, monkeypatch):
    dest = tmp_path / "already.bin"
    calls = {"n": 0}

    class _Response:
        def read(self):
            return b"not-the-file"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(request, timeout=120):
        calls["n"] += 1
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    digest = hashlib.sha256(b"abc").hexdigest()
    dest.write_bytes(b"abc")
    assert fetch_bytes("https://example.com/already.bin", dest, digest) == b"abc"
    assert calls["n"] == 0

    dest.write_bytes(b"poison")
    with pytest.raises(IngestError, match="Checksum mismatch"):
        fetch_bytes("https://example.com/already.bin", dest, digest)
    assert calls["n"] == 1
    assert not dest.exists()

    fresh = tmp_path / "fresh.bin"
    with pytest.raises(IngestError, match="Checksum mismatch"):
        fetch_bytes("https://example.com/fresh.bin", fresh, digest)
    assert not fresh.exists()
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
sources:
  - id: keep
    filename: keep.md
    role: core
    download: true
  - id: skip-role
    filename: skip.md
    role: optional
    download: true
  - id: skip-flag
    filename: flag.md
    role: core
    download: false
""",
        encoding="utf-8",
    )
    chosen = select_sources(load_manifest(manifest), ["core"])
    assert [spec.source_id for spec in chosen] == ["keep"]


def test_disk_cache_does_not_store_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-secret-value")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAGBENCH_API_KEY", raising=False)
    client = LLMClient()
    client.set_disk_cache(tmp_path)
    client._write_disk_cache(("model", 0.0, "prompt"), "leite materno", 4, 2)
    blob = next(tmp_path.glob("*.json")).read_text(encoding="utf-8")
    assert "leite materno" in blob
    assert "sk-test-secret-value" not in blob


def test_baby_food_dry_run_counts_the_staged_grid():
    config = load_search_config(ROOT / "examples" / "baby-food-rag.yaml")
    assert config.embedder_model == "intfloat/multilingual-e5-large"
    assert config.language == "pt"
    assert planned_count(config) == 41
    assert "41" in describe_plan(config)
    assert "manifest.yaml" in config.manifest_url
    assert config.eval_url.endswith("eval/set.yaml")


def test_staged_search_on_local_files(tmp_path):
    (tmp_path / "guia.md").write_text(
        "# Leite\n\nO leite materno exclusivo e recomendado ate os seis meses de idade.\n",
        encoding="utf-8",
    )
    (tmp_path / "outro.md").write_text(
        "# Amendoim\n\nA introducao de amendoim pode ser discutida com o pediatra.\n",
        encoding="utf-8",
    )
    (tmp_path / "manifest.yaml").write_text(
        """
sources:
  - {id: guia, title: Guia, filename: guia.md, role: core, download: true}
  - {id: outro, title: Outro, filename: outro.md, role: core, download: true}
""",
        encoding="utf-8",
    )
    (tmp_path / "eval.yaml").write_text(
        """
questions:
  - id: q1
    question: Quando o leite materno exclusivo e recomendado?
    answerable: true
    gold:
      - {source_id: guia, pdf_page: 1}
  - id: q2
    question: Pergunta sem resposta no material.
    answerable: false
    gold:
      - {source_id: guia, pdf_page: 1}
""",
        encoding="utf-8",
    )
    config_path = tmp_path / "search.yaml"
    config_path.write_text(
        f"""
name: tiny
language: pt
corpus:
  manifest: {tmp_path / "manifest.yaml"}
eval:
  path: {tmp_path / "eval.yaml"}
embedder:
  kind: hash
  query_prefix: "query: "
  passage_prefix: "passage: "
cache_dir: {tmp_path / "cache"}
llm:
  cache_dir: {tmp_path / "llm"}
output_dir: {tmp_path / "out"}
search:
  mode: staged
  stage1_retriever: bm25
  top_k_chunking: 1
  chunkers: [markdown]
  chunk_sizes: [32]
  overlaps: [0.0]
  retrievers: [bm25, dense, hyde]
  gold:
    match: source_page
    page_tolerance: 0
    any_source: false
  bootstrap_samples: 200
  seed: 1
  k_values: [5, 10]
  hit_k: 10
""",
        encoding="utf-8",
    )
    config = load_search_config(config_path)
    assert planned_count(config) == 5

    class FakeLLM:
        def complete(self, prompt, temperature=0.0, use_cache=True):
            if "search queries" in prompt:
                return '{"queries": ["leite materno exclusivo"]}'
            return "O leite materno exclusivo e recomendado ate os seis meses."

    services = Services(
        embedder=PrefixedEmbedder(HashEmbedder(dim=32), "query: ", "passage: "),
        llm=FakeLLM(),
        rrf_k=60,
        fusion_pool=10,
        rag_fusion_queries=3,
    )
    result = run_search(config, services=services)
    assert result["count"] == 5
    rows = {row["config_id"]: row for row in result["rows"]}
    bm25 = next(
        row for row in result["rows"] if row["retriever"] == "bm25" and not row["per_source"]
    )
    assert bm25["status"] == "ok"
    assert bm25["n_queries"] == 1
    assert bm25["recall@5"] == 1
    assert bm25["ndcg@10"] == 1
    assert any(row["per_source"] and row["status"] == "ok" for row in result["rows"])
    assert any(row["retriever"] == "hyde" and row["status"] == "ok" for row in result["rows"])
    csv_path = tmp_path / "out" / "leaderboard.csv"
    with csv_path.open(encoding="utf-8") as handle:
        recorded = list(csv.DictReader(handle))
    assert len(recorded) == 5
    assert (tmp_path / "out" / "leaderboard.png").read_bytes().startswith(b"\x89PNG")
    saved = json.loads((tmp_path / "out" / "per_query.json").read_text(encoding="utf-8"))
    assert saved[0]["query_ids"] == ["q1"]
    assert "sk-" not in csv_path.read_text(encoding="utf-8")
    assert rows[bm25["config_id"]]["recall_source_page@5"] == 1

    bare = Services(embedder=services.embedder, llm=None, rrf_k=60, fusion_pool=10)
    skipped = run_search(config, services=bare)
    hyde = next(row for row in skipped["rows"] if row["retriever"] == "hyde")
    assert hyde["status"] == "skipped"
    assert hyde["reason"] == "needs OPENROUTER_API_KEY"
