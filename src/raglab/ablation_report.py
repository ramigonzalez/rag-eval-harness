"""Write the ablation table. Kept separate so ``raglab run`` reports stay unchanged."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_ablation(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    _write_csv(result["rows"], output_dir / "metrics.csv")
    (output_dir / "results.md").write_text(_markdown(result), encoding="utf-8")


def _write_csv(rows: list[dict], path: Path) -> None:
    fields = [
        "stage",
        "chosen",
        "dev_ndcg@10",
        "test_ndcg@10",
        "delta_test_ndcg@10",
        "test_latency_ms",
        "estimated_usd",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "stage": row["stage"],
                    "chosen": row["chosen"],
                    "dev_ndcg@10": _num(row["dev_ndcg@10"]),
                    "test_ndcg@10": _num(row["test_ndcg@10"]),
                    "delta_test_ndcg@10": (
                        "" if row["delta_test_ndcg@10"] is None else _num(row["delta_test_ndcg@10"])
                    ),
                    "test_latency_ms": _num(row["test_latency_ms"]),
                    "estimated_usd": _num(row["estimated_usd"]),
                }
            )


def _markdown(result: dict) -> str:
    dev = result["dev"]
    test = result["test"]
    lines = [
        "# Stage-wise ablation",
        "",
        (
            f"Dev: {dev['split']}, {dev['num_queries']} queries "
            f"(of {dev['labeled_queries_available']} available), seed {result['seed']}."
        ),
        (
            f"Test: {test['split']}, {test['num_queries']} queries "
            f"(of {test['labeled_queries_available']} available)."
        ),
        "",
        "Each stage is chosen by dev nDCG@10. Delta is the change in test nDCG@10 "
        "from the previous row. Local stages have an estimated cost of $0.",
        "",
        (
            "| Stage | Chosen | Dev nDCG@10 | Test nDCG@10 | "
            "Delta test nDCG@10 | Test latency (ms) | Est. USD |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["rows"]:
        lines.append(
            "| {stage} | {chosen} | {dev} | {test} | {delta} | {latency} | {usd} |".format(
                stage=row["stage"],
                chosen=row["chosen"],
                dev=f"{row['dev_ndcg@10']:.3f}",
                test=f"{row['test_ndcg@10']:.3f}",
                delta=(
                    "" if row["delta_test_ndcg@10"] is None else f"{row['delta_test_ndcg@10']:.3f}"
                ),
                latency=f"{row['test_latency_ms']:.1f}",
                usd=f"{row['estimated_usd']:.2f}",
            )
        )
    if result["skipped"]:
        lines.extend(["", "Skipped because no API key was set:"])
        for item in result["skipped"]:
            lines.append(f"- {item['stage']} / {item['option']}")
    final = result["final"]
    lines.extend(
        [
            "",
            (
                "Final pipeline chosen on dev: "
                f"chunking={final['chunking']}, query_transform={final['query_transform']}, "
                f"retrieval={final['retrieval']}, rerank={final['rerank']}."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _num(value: float) -> str:
    return f"{float(value):.6f}"
