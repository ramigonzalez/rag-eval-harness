"""Charts are redrawn from the result files, including the 1200x627 export."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "make_charts.py"
    spec = importlib.util.spec_from_file_location("make_charts", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_charts_come_from_result_files_and_linkedin_size_is_exact():
    module = _load_script()
    module.main()
    linkedin = ROOT / "results" / "scifact" / "ndcg_vs_latency_linkedin.png"
    quality = ROOT / "results" / "scifact" / "ndcg_vs_latency.png"
    judge = ROOT / "results" / "generation" / "faithfulness_hallucination.png"
    assert quality.stat().st_size > 0
    assert judge.stat().st_size > 0
    assert module.png_size(linkedin) == (1200, 627)
