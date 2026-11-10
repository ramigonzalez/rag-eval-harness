import json
from pathlib import Path

import pytest
import yaml

from raglab.cli import main
from raglab.retrieval import RETRIEVERS

ROOT = Path(__file__).resolve().parents[1]


def test_help_exits_cleanly():
    with pytest.raises(SystemExit) as caught:
        main(["--help"])
    assert caught.value.code == 0


def test_missing_config_is_a_usage_error():
    assert main(["run", "--config", "configs/does-not-exist.yaml"]) == 2


def test_cli_check_fails_when_the_gate_is_impossible(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAGBENCH_API_KEY", raising=False)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": "fixture",
                "fixture_path": str(ROOT / "data" / "fixtures" / "smoke.json"),
                "output_dir": str(tmp_path / "out"),
                "data_dir": str(tmp_path / "data"),
                "cache_dir": str(tmp_path / "cache"),
                "latency_queries": 1,
                "hit_k": 10,
                "embedder": {"kind": "hash", "dim": 128},
                "reranker": {"kind": "lexical"},
                "chunkers": [{"name": "fixed", "chunk_size": 40, "overlap": 10}],
                "retrievers": ["bm25"],
                "generation": {"mode": "off", "pairs": [["fixed", "bm25"]]},
            }
        ),
        encoding="utf-8",
    )
    check = tmp_path / "gate.json"
    check.write_text(
        json.dumps(
            {"rules": [{"chunker": "fixed", "retriever": "bm25", "metric": "ndcg@10", "min": 2}]}
        ),
        encoding="utf-8",
    )
    code = main(["run", "--config", str(config_path), "--check", str(check)])
    assert code == 1


def test_plugin_module_registers_a_retriever(tmp_path, monkeypatch):
    plugin = tmp_path / "raglab_plugin_demo.py"
    plugin.write_text(
        "\n".join(
            [
                "from raglab.retrieval import register_retriever",
                "from raglab.retrieval.strategies import BM25Retriever",
                "",
                "@register_retriever",
                "class DemoRetriever(BM25Retriever):",
                "    name = 'demo'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import importlib

    module = importlib.import_module("raglab_plugin_demo")
    assert RETRIEVERS["demo"] is module.DemoRetriever
    del RETRIEVERS["demo"]
