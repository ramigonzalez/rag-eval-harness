"""The README may only quote numbers this run actually wrote down."""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _table(text: str, header: str) -> list[list[str]]:
    lines = text.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith(header))
    rows = []
    for line in lines[start + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def test_readme_retrieval_table_matches_the_csv():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    with (ROOT / "results" / "scifact" / "metrics.csv").open(encoding="utf-8") as handle:
        recorded = {(row["chunker"], row["retriever"]): row for row in csv.DictReader(handle)}
    shown = _table(readme, "| Chunker | Retriever | Recall@10 |")
    assert len(shown) == len(recorded)
    for cells in shown:
        source = recorded[(cells[0], cells[1])]
        assert cells[2] == f"{float(source['recall@10']):.3f}"
        assert cells[3] == f"{float(source['mrr@10']):.3f}"
        assert cells[4] == f"{float(source['ndcg@10']):.3f}"
        assert cells[5] == f"{float(source['context_precision@10']):.3f}"
        assert cells[6] == f"{float(source['online_latency_ms']):.1f}"


def test_readme_marks_generation_as_needing_a_key():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    recorded = json.loads((ROOT / "results" / "scifact" / "metrics.json").read_text())
    assert recorded["generation"]["status"] == "skipped"
    assert recorded["skipped_retrievers"] == ["hyde", "rag_fusion"]
    assert "run with API key" in readme
    assert "5,183" in readme
    assert "15,596" in readme
    assert recorded["dataset"]["num_queries"] == 300
    assert recorded["environment"]["cpu_count"] == 4
