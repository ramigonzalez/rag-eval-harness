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


def _json_number(value: object) -> str:
    return json.dumps(value)


def test_readme_generation_table_matches_the_judge_file():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    results = (ROOT / "results" / "gen-100-judge-v2" / "results.md").read_text(encoding="utf-8")
    recorded = json.loads((ROOT / "results" / "gen-100-judge-v2" / "metrics.json").read_text())
    v1 = json.loads((ROOT / "results" / "generation-100" / "metrics.json").read_text())
    shown = _table(readme, "| Chunker | Retriever | Refusal rate | Answered |")
    in_file = _table(results, "| Chunker | Retriever | Refusal rate | Answered |")
    assert shown == in_file
    pairs = {(pair["chunker"], pair["retriever"]): pair for pair in recorded["pairs"]}
    assert len(shown) == len(pairs) == 5
    faith_values = []
    for cells in shown:
        source = pairs[(cells[0], cells[1])]
        assert cells[2] == f"{float(source['refusal_rate']):.2f}"
        assert cells[3] == str(source["num_answered"])
        assert cells[4] == f"{float(source['faithfulness']):.3f}"
        assert cells[5] == f"{float(source['hallucination_rate']):.3f}"
        assert cells[6] == f"{float(source['answer_relevancy']):.3f}"
        assert cells[7] == f"{float(source['faithfulness_v1_answered_only']):.3f}"
        faith_values.append(float(source["faithfulness"]))
    low = f"{min(faith_values):.3f}"
    high = f"{max(faith_values):.3f}"
    assert low in readme and high in readme
    assert "from **0.596**" not in readme
    assert "to **0.644**" not in readme
    fixed = pairs[("fixed", "rerank")]
    assert fixed["num_faithfulness_scored"] == 50
    assert "n=50" in readme
    v1_fixed = next(
        pair
        for pair in v1["generation"]["pairs"]
        if pair["chunker"] == "fixed" and pair["retriever"] == "rerank"
    )
    assert v1_fixed["num_scored"] == 49
    assert "n=49" in readme
    gold = v1["generation"]["refusal_gold"]
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
    analysis = json.loads(
        (ROOT / "results" / "judge-validation" / "analysis_opus_v2.json").read_text()
    )
    validation = _table(readme, "| Judge | n | MAE |")
    blocks = {
        "v1": analysis["faithfulness"]["old_judge_v1"],
        "v2": analysis["faithfulness"]["new_judge_v2"],
        "RAGAS": analysis["faithfulness"]["ragas"],
    }
    assert [row[0] for row in validation] == ["v1", "v2", "RAGAS"]
    for row in validation:
        block = blocks[row[0]]
        assert row[1:] == [
            _json_number(block["n"]),
            _json_number(block["mae"]),
            _json_number(block["agreement_within_0.25"]),
            _json_number(block["spearman"]),
            _json_number(block["accuracy_at_0.5"]),
        ]
    for item in analysis["contradiction_items"]:
        old = _json_number(item["old"])
        new = _json_number(item["new"])
        assert f"item {item['item']}, {old} to {new}" in readme
    assert "not human annotation" in readme
    assert "not against human labels" in readme or "not human" in readme
    assert recorded["judge_model"] == "openai/gpt-4o-mini"
    assert recorded["context_k"] == 5
    assert "results/generation-100/results.md" in readme
    assert "did not run" not in readme
    assert "rag-eval-harness" in readme
    assert "raglab run --config configs/generation.yaml" in readme
    assert "raglab run --config configs/generation_100.yaml" in readme
    assert v1["dataset"]["num_queries"] == 100
    assert v1["environment"]["cpu_count"] == 8


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
