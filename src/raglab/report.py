"""CSV, JSON, markdown, and the nDCG chart."""

from __future__ import annotations

import csv
import json
from pathlib import Path

METRIC_FIELDS = [
    "recall@5",
    "recall@10",
    "mrr@5",
    "mrr@10",
    "ndcg@5",
    "ndcg@10",
    "context_precision@5",
    "context_precision@10",
]

CHUNKER_LABELS = {
    "whole": "Whole document",
    "fixed": "Fixed",
    "recursive": "Recursive",
    "semantic": "Semantic",
}
RETRIEVER_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "hybrid": "Hybrid",
    "rerank": "Rerank",
    "hyde": "HyDE",
    "rag_fusion": "RAG-Fusion",
}
CHUNKER_COLORS = {
    "whole": "#4C4C4C",
    "fixed": "#0072B2",
    "recursive": "#009E73",
    "semantic": "#E69F00",
}
FALLBACK_COLORS = ["#56B4E9", "#CC79A7", "#D55E00", "#F0E442"]


def write_outputs(result: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    examples = result.get("examples", [])
    payload = {key: value for key, value in result.items() if key != "examples"}
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "examples.json").write_text(
        json.dumps(examples, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_csv(result["rows"], output_dir / "metrics.csv")
    (output_dir / "results.md").write_text(render_markdown(result), encoding="utf-8")
    write_chart(result, output_dir / "ndcg_at_10.png")


def render_markdown(result: dict) -> str:
    dataset = result["dataset"]
    lines = [
        f"# {dataset['name']} retrieval results",
        "",
        (
            f"{dataset['num_documents']:,} documents, "
            f"{dataset['num_queries']:,} labeled queries, "
            f"mean {dataset['mean_relevant_docs']:.2f} relevant documents per query, "
            f"mean document length {dataset['mean_doc_words']:.1f} words."
        ),
        "",
        "## Chunking",
        "",
        "| Chunker | Chunks | Avg words | Chunks per doc | Chunk time (s) | Index time (s) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for chunker in result["chunkers"]:
        lines.append(
            "| {name} | {num_chunks:,} | {avg_chunk_words:.1f} | {avg_chunks_per_doc:.2f} | "
            "{chunk_seconds:.1f} | {index_seconds:.1f} |".format(**chunker)
        )
    lines.extend(
        [
            "",
            "Index time is shared by every retriever on that chunker. "
            "It includes embedding when a dense index is built.",
            "",
            "## Retrieval",
            "",
            _markdown_row(
                [
                    "Chunker",
                    "Retriever",
                    "Recall@10",
                    "MRR@10",
                    "nDCG@10",
                    "Context precision@10",
                    "Online latency (ms)",
                ]
            ),
            _markdown_row(["---", "---", "---:", "---:", "---:", "---:", "---:"]),
        ]
    )
    for row in result["rows"]:
        if row["status"] != "ok":
            continue
        metrics = row["metrics"]
        lines.append(
            _markdown_row(
                [
                    row["chunker"],
                    row["retriever"],
                    _fmt(metrics.get("recall@10")),
                    _fmt(metrics.get("mrr@10")),
                    _fmt(metrics.get("ndcg@10")),
                    _fmt(metrics.get("context_precision@10")),
                    _fmt(row.get("online_latency_ms"), digits=1),
                ]
            )
        )
    lines.extend(
        [
            "",
            (
                f"Online latency is the mean of up to {result['latency_queries']} single-query "
                "calls on this machine, including query embedding. "
                "It is not the batched throughput of the full eval."
            ),
            "",
            "Metrics are macro-averaged over labeled queries. Recall, MRR, and nDCG are "
            "computed after mapping chunks back to documents. Context precision scores the "
            "chunks in the order they were retrieved.",
            "",
        ]
    )
    errors = [row for row in result["rows"] if row["status"] != "ok"]
    if errors:
        lines.append("Rows that did not finish:")
        lines.append("")
        for row in errors:
            lines.append(
                f"- {row['chunker']} / {row['retriever']}: {row['status']} ({row.get('note', '')})"
            )
        lines.append("")
    lines.extend(
        [
            "## Generation",
            "",
            result["generation"]["reason"]
            if result["generation"]["status"] != "ok"
            else _generation_markdown(result["generation"]),
            "",
            "![nDCG@10 by chunker and retriever](ndcg_at_10.png)",
            "",
        ]
    )
    return "\n".join(lines)


def write_chart(result: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in result["rows"]
        if row["status"] == "ok" and row["metrics"].get("ndcg@10") is not None
    ]
    chunkers = list(dict.fromkeys(row["chunker"] for row in rows))
    retrievers = list(dict.fromkeys(row["retriever"] for row in rows))
    if not chunkers or not retrievers:
        return
    lookup = {(row["chunker"], row["retriever"]): row["metrics"]["ndcg@10"] for row in rows}
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    positions = list(range(len(retrievers)))
    width = min(0.8 / max(len(chunkers), 1), 0.18)
    for index, chunker in enumerate(chunkers):
        offset = (index - (len(chunkers) - 1) / 2) * width
        values = [lookup.get((chunker, retriever), float("nan")) for retriever in retrievers]
        color = CHUNKER_COLORS.get(chunker) or FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
        ax.bar(
            [pos + offset for pos in positions],
            values,
            width=width * 0.92,
            label=CHUNKER_LABELS.get(chunker, chunker),
            color=color,
        )
    ax.set_ylim(0, 1)
    ax.set_xticks(positions)
    ax.set_xticklabels([RETRIEVER_LABELS.get(name, name) for name in retrievers])
    ax.set_ylabel("nDCG@10")
    dataset = result["dataset"]
    title_name = {"scifact": "SciFact", "smoke": "Smoke"}.get(dataset["name"], dataset["name"])
    ax.set_title(
        f"{title_name} nDCG@10 — {dataset['num_queries']:,} queries",
        loc="left",
        fontsize=13,
        pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(
        title="Chunker",
        frameon=False,
        ncol=min(4, len(chunkers)),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
    )
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def check_thresholds(rows: list[dict], path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rules = payload.get("rules", [])
    available = {(row["chunker"], row["retriever"]): row for row in rows if row["status"] == "ok"}
    failures = []
    for rule in rules:
        key = (rule["chunker"], rule["retriever"])
        row = available.get(key)
        metric = rule["metric"]
        minimum = float(rule["min"])
        if row is None:
            failures.append(f"{key[0]} / {key[1]} is missing, so {metric} cannot clear {minimum}")
            continue
        value = row["metrics"].get(metric)
        if value is None or float(value) < minimum:
            failures.append(
                f"{key[0]} / {key[1]} {metric} = {value} is below the minimum {minimum}"
            )
    return failures


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "chunker",
        "retriever",
        "status",
        "num_queries",
        *METRIC_FIELDS,
        "eval_seconds",
        "online_latency_ms",
        "min_unique_docs",
        "mean_unique_docs",
        "fallbacks",
        "note",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = {
                "chunker": row["chunker"],
                "retriever": row["retriever"],
                "status": row["status"],
                "num_queries": row.get("num_queries", ""),
                "eval_seconds": _csv_float(row.get("eval_seconds")),
                "online_latency_ms": _csv_float(row.get("online_latency_ms")),
                "min_unique_docs": row.get("min_unique_docs", ""),
                "mean_unique_docs": _csv_float(row.get("mean_unique_docs")),
                "fallbacks": row.get("fallbacks", ""),
                "note": row.get("note", ""),
            }
            metrics = row.get("metrics") or {}
            for field in METRIC_FIELDS:
                flat[field] = _csv_float(metrics.get(field))
            writer.writerow(flat)


def _generation_markdown(generation: dict) -> str:
    lines = [
        f"Model `{generation['model']}` judged {generation['num_queries']} queries.",
        "",
        "| Chunker | Retriever | Faithfulness | Answer relevancy | Hallucination rate | Errors |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for pair in generation["pairs"]:
        lines.append(
            "| {chunker} | {retriever} | {faithfulness} | {answer_relevancy} | "
            "{hallucination_rate} | {num_errors} |".format(
                chunker=pair["chunker"],
                retriever=pair["retriever"],
                faithfulness=_fmt(pair["faithfulness"]),
                answer_relevancy=_fmt(pair["answer_relevancy"]),
                hallucination_rate=_fmt(pair["hallucination_rate"]),
                num_errors=pair["num_errors"],
            )
        )
    return "\n".join(lines)


def _markdown_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _fmt(value: object, digits: int = 3) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _csv_float(value: object) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.6f}"
