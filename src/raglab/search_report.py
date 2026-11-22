"""Leaderboard files for a search run. The chart is drawn from the CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path

COLUMNS = [
    "config_id",
    "stage",
    "chunker",
    "chunk_size",
    "overlap",
    "retriever",
    "per_source",
    "status",
    "n_queries",
    "recall@5",
    "recall@10",
    "mrr@5",
    "mrr@10",
    "ndcg@5",
    "ndcg@10",
    "latency_ms",
    "ndcg_ci_low",
    "ndcg_ci_high",
    "vs_best",
    "recall_any_source@5",
    "recall_any_source@10",
    "recall_source_page@5",
    "recall_source_page@10",
    "mean_chunk_words",
    "mean_words@10",
    "recall_budget@5",
    "recall_budget@10",
    "reason",
]

RETRIEVER_COLORS = {
    "bm25": "#0072B2",
    "dense": "#009E73",
    "hybrid": "#E69F00",
    "hyde": "#CC79A7",
    "rag_fusion": "#D55E00",
}


def write_search_outputs(output_dir: Path, rows: list[dict], plan: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "leaderboard.csv"
    _write_csv(rows, csv_path)
    (output_dir / "leaderboard.md").write_text(_markdown(rows, plan), encoding="utf-8")
    _write_queries(rows, output_dir / "per_query.json")
    write_search_chart(csv_path, output_dir / "leaderboard.png")


def write_search_chart(csv_path: Path, png_path: Path) -> None:
    """Scatter nDCG@10 against latency. Every point is a row of ``csv_path``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") != "ok" or not row.get("ndcg@10") or not row.get("latency_ms"):
                continue
            points.append(row)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    if not points:
        axis.set_title("No scored configs")
    else:
        by_retriever: dict[str, list[dict]] = {}
        for row in points:
            by_retriever.setdefault(row["retriever"], []).append(row)
        for retriever, group in by_retriever.items():
            axis.scatter(
                [float(row["latency_ms"]) for row in group],
                [float(row["ndcg@10"]) for row in group],
                label=retriever,
                color=RETRIEVER_COLORS.get(retriever, "#4C4C4C"),
            )
        best = max(points, key=lambda row: float(row["ndcg@10"]))
        axis.annotate(
            best["config_id"],
            (float(best["latency_ms"]), float(best["ndcg@10"])),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
        )
        axis.legend()
    axis.set_xlabel("Latency per query (ms)")
    axis.set_ylabel("nDCG@10")
    axis.set_title("Search leaderboard")
    figure.tight_layout()
    figure.savefig(png_path, dpi=120)
    plt.close(figure)


def _write_csv(rows: list[dict], path: Path) -> None:
    ordered = sorted(rows, key=lambda row: (row.get("status") != "ok", _sort_key(row)))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in ordered:
            flat = {column: row.get(column, "") for column in COLUMNS}
            for column in COLUMNS:
                value = flat[column]
                if isinstance(value, float):
                    flat[column] = f"{value:.6f}"
                elif isinstance(value, bool):
                    flat[column] = "true" if value else "false"
            writer.writerow(flat)


def _write_queries(rows: list[dict], path: Path) -> None:
    payload = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        payload.append(
            {
                "config_id": row["config_id"],
                "query_ids": row.get("query_ids") or [],
                "ndcg@10": row.get("per_query_ndcg@10") or [],
            }
        )
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _markdown(rows: list[dict], plan: str) -> str:
    ordered = sorted(rows, key=lambda row: (row.get("status") != "ok", _sort_key(row)))
    lines = [
        "# Search leaderboard",
        "",
        plan,
        "",
        "nDCG CI is the paired bootstrap 95% interval of this config minus the current best, "
        "on per-query nDCG@10. Recall columns are the primary gold match. "
        "`recall_any_source` ignores source id. "
        "`words/chunk` is the mean chunk length. "
        "`words@10` is the mean words in the first 10 hits. "
        "`R@budget` is recall@10 after the hit list is cut to the configured word budget. "
        "The chart is drawn from `leaderboard.csv`.",
        "",
        "| Config | R@10 | nDCG@10 | Words/chunk | Words@10 | R@budget | Latency ms | CI vs best |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in ordered:
        lines.append(
            "| {config} | {r10} | {ndcg} | {chunk_words} | {top_words} "
            "| {budget} | {latency} | {ci} |".format(
                config=row["config_id"],
                r10=_num(row.get("recall@10")),
                ndcg=_num(row.get("ndcg@10")),
                chunk_words=_num(row.get("mean_chunk_words"), 0),
                top_words=_num(row.get("mean_words@10"), 0),
                budget=_num(row.get("recall_budget@10")),
                latency=_num(row.get("latency_ms"), 1),
                ci=_ci(row),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _sort_key(row: dict):
    ndcg = row.get("ndcg@10")
    return (-1.0 if ndcg is None else -float(ndcg), row["config_id"])


def _num(value, digits: int = 3) -> str:
    if value is None or value == "":
        return ""
    return f"{float(value):.{digits}f}"


def _ci(row: dict) -> str:
    if row.get("status") != "ok":
        return row.get("reason") or "skipped"
    low = row.get("ndcg_ci_low")
    high = row.get("ndcg_ci_high")
    if low is None or high is None:
        return ""
    return f"{float(low):+.3f} to {float(high):+.3f}"
