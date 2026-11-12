import importlib.util
import json
from pathlib import Path

import yaml

from raglab.ablation import load_ablation_config, run_ablation
from raglab.chunking import MarkdownChunker, ParentDocumentChunker, SentenceWindowChunker
from raglab.retrieval.fusion import reciprocal_rank_fusion, union_fusion

ROOT = Path(__file__).resolve().parents[1]


def _ragas_module():
    path = ROOT / "scripts" / "ragas_agreement.py"
    spec = importlib.util.spec_from_file_location("ragas_agreement", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_markdown_splits_on_headings_and_keeps_plain_text_whole():
    sections = MarkdownChunker().split_text("# Title\n\nIntro line.\n\n## Next\n\nBody line.")
    assert sections[0].startswith("# Title")
    assert sections[1].startswith("## Next")
    assert MarkdownChunker().split_text("No headings here.") == ["No headings here."]


def test_sentence_window_and_parent_document_chunk():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    windows = SentenceWindowChunker(window_sentences=2, overlap_sentences=1).split_text(text)
    assert len(windows) >= 2
    assert windows[0].startswith("One two three.")
    children = ParentDocumentChunker(chunk_size=4, overlap=1).split_text(
        " ".join(f"w{i}" for i in range(10))
    )
    assert children[0] == "w0 w1 w2 w3"
    assert "w3" in children[1]


def test_union_keeps_the_best_list_while_rrf_adds_them():
    lists = [["d1", "d2"], ["d2", "d3"]]
    rrf = dict(reciprocal_rank_fusion(lists, k=60))
    union = dict(union_fusion(lists, k=60))
    assert rrf["d2"] == pytest_approx(1 / 62 + 1 / 61)
    assert union["d2"] == pytest_approx(1 / 61)
    assert union["d2"] < rrf["d2"]


def pytest_approx(value):
    import pytest

    return pytest.approx(value)


def test_weighted_rrf_can_change_the_winner():
    lists = [["d1", "d2"], ["d2", "d1"]]
    even = [item[0] for item in reciprocal_rank_fusion(lists, k=60, weights=[1.0, 1.0])]
    tilted = [item[0] for item in reciprocal_rank_fusion(lists, k=60, weights=[5.0, 1.0])]
    assert even[0] == "d1" or even[0] == "d2"
    assert tilted[0] == "d1"


def test_ablation_selects_on_dev_and_reports_test_delta(tmp_path):
    path = tmp_path / "ablation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "dataset": "fixture",
                "fixture_path": str(ROOT / "data" / "fixtures" / "smoke.json"),
                "output_dir": str(tmp_path / "out"),
                "data_dir": str(tmp_path / "data"),
                "cache_dir": str(tmp_path / "cache"),
                "seed": 7,
                "dev_count": 3,
                "baseline": {
                    "chunking": "whole",
                    "query_transform": "none",
                    "retrieval": "bm25",
                    "rerank": "none",
                },
                "stages": {
                    "chunking": ["whole", "fixed"],
                    "query_transform": ["none", "hyde"],
                    "retrieval": ["bm25", "dense"],
                    "rerank": ["none", "mmr"],
                },
                "params": {"chunk_size": 40, "overlap": 10, "top_k": 10, "rerank_pool": 8},
                "embedder": {"kind": "hash", "dim": 256},
                "reranker": {"kind": "lexical"},
            }
        ),
        encoding="utf-8",
    )
    result = run_ablation(load_ablation_config(path))
    stages = [row["stage"] for row in result["rows"]]
    assert stages == ["baseline", "chunking", "query_transform", "retrieval", "rerank"]
    assert result["rows"][0]["delta_test_ndcg@10"] is None
    assert result["rows"][2]["chosen"] == "none"
    assert {item["option"] for item in result["skipped"]} == {"hyde"}
    for row in result["rows"][1:]:
        previous = result["rows"][result["rows"].index(row) - 1]["test_ndcg@10"]
        assert row["delta_test_ndcg@10"] == pytest_approx(row["test_ndcg@10"] - previous)
        assert row["estimated_usd"] == 0.0
    saved = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert saved["final"]["query_transform"] == "none"
    assert (tmp_path / "out" / "results.md").exists()


def test_ragas_agreement_math_and_missing_inputs(tmp_path, capsys):
    module = _ragas_module()
    assert module.agreement([1.0, 0.5], [0.5, 0.5])["mean_abs_diff"] == pytest_approx(0.25)
    missing = tmp_path / "missing.jsonl"
    assert module.main(["--answers", str(missing)]) == 0
    assert "No numbers were computed" in capsys.readouterr().err
