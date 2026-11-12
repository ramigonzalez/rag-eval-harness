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
    scifact = json.loads((ROOT / "results" / "scifact" / "metrics.json").read_text())
    assert scifact["dataset"]["num_queries"] == 300
    assert scifact["environment"]["cpu_count"] == 4
    assert scifact["generation"]["status"] == "skipped"
    assert "5,183" in readme


def test_readme_generation_table_matches_the_judge_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = (ROOT / "results" / "generation-100" / "results.md").read_text(encoding="utf-8")
    recorded = json.loads((ROOT / "results" / "generation-100" / "metrics.json").read_text())
    shown = _table(readme, "| Chunker | Retriever | Refusal rate |")
    in_file = _table(results, "| Chunker | Retriever | Refusal rate |")
    pairs = {(pair["chunker"], pair["retriever"]): pair for pair in recorded["generation"]["pairs"]}
    assert shown == in_file
    assert len(shown) == len(pairs) == 5
    faith_values = []
    for cells in shown:
        source = pairs[(cells[0], cells[1])]
        assert cells[2] == f"{float(source['refusal_rate']):.3f}"
        assert cells[3] == str(source["num_scored"])
        assert cells[4] == f"{float(source['faithfulness']):.3f}"
        assert cells[5] == f"{float(source['hallucination_rate']):.3f}"
        assert cells[6] == f"{float(source['answer_relevancy']):.3f}"
        assert cells[7] == str(source["num_errors"])
        assert source["all_rows"]["faithfulness"] < source["faithfulness"]
        faith_values.append(float(source["faithfulness"]))
    low = f"{min(faith_values):.3f}"
    high = f"{max(faith_values):.3f}"
    assert low in readme and high in readme
    assert "from **0.596**" not in readme
    assert "to **0.644**" not in readme
    fixed = pairs[("fixed", "rerank")]
    assert (
        f"{fixed['num_answered']} answered queries and {fixed['num_errors']} error, "
        f"so n is {fixed['num_scored']}"
    ) in readme
    gold = recorded["generation"]["refusal_gold"]
    stacks = ", ".join(
        f"{stack['chunker']}/{stack['retriever']} "
        f"{stack['num_with_gold_doc']}/{stack['num_refusals']}"
        for stack in gold["stacks"]
    )
    gold_sentence = (
        f"{gold['num_with_gold_doc']}/{gold['num_refusals']} refusals had a "
        f"gold-relevant document in the 5 contexts ({stacks})."
    )
    assert gold_sentence in readme
    assert gold_sentence in results
    assert recorded["generation"]["num_queries"] == 100
    assert recorded["generation"]["model"] == "openai/gpt-4o-mini"
    assert recorded["generation"]["context_k"] == 5
    assert recorded["llm"]["provider"] == "openrouter"
    assert recorded["llm"]["usage"]["calls"] == 1122
    assert "results/generation-100/results.md" in readme
    assert "not yet validated against human labels" in readme
    assert "RAGAS agreement is pending" in readme
    assert "did not run" not in readme
    assert "rag-eval-harness" in readme
    assert "raglab run --config configs/generation.yaml" in readme
    assert "raglab run --config configs/generation_100.yaml" in readme
    assert recorded["dataset"]["num_queries"] == 100
    assert recorded["environment"]["cpu_count"] == 8


def test_readme_ablation_table_matches_the_csv():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    with (ROOT / "results" / "ablation" / "metrics.csv").open(encoding="utf-8") as handle:
        recorded = list(csv.DictReader(handle))
    shown = _table(readme, "| Stage | Chosen | Dev nDCG@10 |")
    assert len(shown) == len(recorded)
    for cells, source in zip(shown, recorded, strict=True):
        assert cells[0] == source["stage"]
        assert cells[1] == source["chosen"]
        assert cells[2] == f"{float(source['dev_ndcg@10']):.3f}"
        assert cells[3] == f"{float(source['test_ndcg@10']):.3f}"
        if source["delta_test_ndcg@10"] == "":
            assert cells[4] == ""
        else:
            assert cells[4] == f"{float(source['delta_test_ndcg@10']):.3f}"
        assert cells[5] == f"{float(source['test_latency_ms']):.1f}"
        assert cells[6] == f"{float(source['estimated_usd']):.2f}"
