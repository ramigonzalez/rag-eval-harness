"""Draw charts from committed result files. No hardcoded metric values.

    python scripts/make_charts.py

Reads results/scifact/metrics.csv, results/generation/metrics.json, and
results/generation-100/metrics.json. Writes:
    results/scifact/ndcg_vs_latency.png
    results/scifact/ndcg_vs_latency_linkedin.png   (1200 x 627)
    results/generation/faithfulness_hallucination.png          (20-query sample)
    results/generation-100/faithfulness_hallucination.png      (headline chart)
"""

from __future__ import annotations

import csv
import json
import struct
from pathlib import Path

CHUNKER_COLORS = {
    "whole": "#4C4C4C",
    "fixed": "#0072B2",
    "recursive": "#009E73",
    "semantic": "#E69F00",
}
RETRIEVER_MARKERS = {
    "bm25": "o",
    "dense": "s",
    "hybrid": "D",
    "rerank": "^",
    "hyde": "P",
    "rag_fusion": "X",
}
# LinkedIn thumbnail colors the retriever, not the chunker.
RETRIEVER_COLORS = {
    "bm25": "#0072B2",
    "dense": "#009E73",
    "hybrid": "#E69F00",
    "rerank": "#D55E00",
}
# Which measured points get a direct label on the thumbnail.
LINKEDIN_CALLOUTS = (
    ("fixed", "rerank"),
    ("fixed", "hybrid"),
    ("whole", "bm25"),
    ("whole", "rerank"),
)
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


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    scifact_csv = root / "results" / "scifact" / "metrics.csv"
    generation_json = root / "results" / "generation" / "metrics.json"
    generation_100_json = root / "results" / "generation-100" / "metrics.json"
    rows = _load_csv(scifact_csv)
    generation = json.loads(generation_json.read_text(encoding="utf-8"))
    generation_100 = json.loads(generation_100_json.read_text(encoding="utf-8"))
    _quality_latency(rows, root / "results" / "scifact" / "ndcg_vs_latency.png")
    _quality_latency_linkedin(
        rows,
        root / "results" / "scifact" / "ndcg_vs_latency_linkedin.png",
    )
    _faithfulness(
        generation,
        root / "results" / "generation" / "faithfulness_hallucination.png",
    )
    _faithfulness(
        generation_100,
        root / "results" / "generation-100" / "faithfulness_hallucination.png",
    )


def _load_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["status"] == "ok"]
    if not rows:
        raise SystemExit(f"No rows in {path}")
    return rows


def _quality_latency(rows: list[dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = [
        {
            "chunker": row["chunker"],
            "retriever": row["retriever"],
            "ndcg": float(row["ndcg@10"]),
            "latency": float(row["online_latency_ms"]),
        }
        for row in rows
    ]
    queries = rows[0]["num_queries"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    for point in points:
        ax.scatter(
            point["latency"],
            point["ndcg"],
            s=64,
            color=CHUNKER_COLORS.get(point["chunker"], "#333333"),
            marker=RETRIEVER_MARKERS.get(point["retriever"], "o"),
            zorder=3,
            linewidths=0.6,
            edgecolors="#222222",
        )
    _legend(ax, points)
    latencies = [point["latency"] for point in points]
    scores = [point["ndcg"] for point in points]
    ax.set_xscale("log")
    ax.set_xlim(min(latencies) * 0.55, max(latencies) * 1.9)
    pad = 0.012
    ax.set_ylim(min(scores) - pad, max(scores) + pad * 2.2)
    ax.set_xlabel("Online latency (ms, log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_title(_headline(points), loc="left", fontsize=11, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(
        0.01,
        0.01,
        f"SciFact, {queries} queries. Y-axis is zoomed to the measured range.",
        fontsize=8,
        color="#444444",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _quality_latency_linkedin(rows: list[dict], path: Path) -> None:
    """1200x627 thumbnail. Story points are colored by retriever and labeled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dpi = 100
    points = _points(rows)
    queries = rows[0]["num_queries"]
    fig = plt.figure(figsize=(12.0, 6.27), dpi=dpi)
    # Leave room for a large title and axis labels, and use the rest of the canvas.
    ax = fig.add_axes((0.10, 0.18, 0.86, 0.62))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    callouts = set(LINKEDIN_CALLOUTS)
    for point in points:
        featured = (point["chunker"], point["retriever"]) in callouts
        ax.scatter(
            point["latency"],
            point["ndcg"],
            s=220 if featured else 55,
            color=RETRIEVER_COLORS.get(point["retriever"], "#555555") if featured else "#C5C5C5",
            marker=RETRIEVER_MARKERS.get(point["retriever"], "o"),
            zorder=4 if featured else 2,
            linewidths=0.8,
            edgecolors="#222222" if featured else "#A0A0A0",
        )
    _linkedin_labels(ax, points, dpi)
    latencies = [point["latency"] for point in points]
    scores = [point["ndcg"] for point in points]
    ax.set_xscale("log")
    ax.set_xlim(min(latencies) * 0.55, max(latencies) * 1.35)
    ax.set_ylim(min(scores) - 0.012, max(scores) + 0.02)
    title_pt = _px(34, dpi)
    label_pt = _px(20, dpi)
    tick_pt = _px(17, dpi)
    ax.set_xlabel("Online latency (ms, log scale)", fontsize=label_pt, labelpad=6)
    ax.set_ylabel("nDCG@10", fontsize=label_pt, labelpad=4)
    ax.set_title(_linkedin_title(points), loc="left", fontsize=title_pt, pad=12)
    ax.tick_params(axis="both", labelsize=tick_pt, length=5, width=1.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.xaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.25)
    ax.set_axisbelow(True)
    fig.text(
        0.10,
        0.035,
        f"SciFact, {queries} queries. Gray points are the other chunker/retriever pairs.",
        fontsize=_px(15, dpi),
        color="#444444",
    )
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def _points(rows: list[dict]) -> list[dict]:
    return [
        {
            "chunker": row["chunker"],
            "retriever": row["retriever"],
            "ndcg": float(row["ndcg@10"]),
            "latency": float(row["online_latency_ms"]),
        }
        for row in rows
    ]


def _px(pixels: float, dpi: int) -> float:
    """Matplotlib font size, in points, that renders at about this many pixels."""
    return pixels * 72 / dpi


def _linkedin_title(points: list[dict]) -> str:
    fixed = [point for point in points if point["chunker"] == "fixed"]
    rerank = next(point for point in fixed if point["retriever"] == "rerank")
    hybrid = next(point for point in fixed if point["retriever"] == "hybrid")
    return (
        f"Fixed + rerank  {rerank['ndcg']:.3f} at {rerank['latency']:.0f} ms\n"
        f"Hybrid  {hybrid['ndcg']:.3f} at {hybrid['latency']:.0f} ms"
    )


def _linkedin_labels(ax, points: list[dict], dpi: int) -> None:
    by_key = {(point["chunker"], point["retriever"]): point for point in points}
    short_chunker = {"whole": "Whole", "fixed": "Fixed"}
    # Offsets are layout only. The numbers in the text come from the row.
    offsets = {
        ("whole", "bm25"): ((0, 14), "center", "bottom"),
        ("fixed", "hybrid"): ((-6, 14), "right", "bottom"),
        ("fixed", "rerank"): ((-28, 2), "right", "center"),
        ("whole", "rerank"): ((-12, -18), "right", "top"),
    }
    font = _px(18, dpi)
    for key, (xytext, ha, va) in offsets.items():
        point = by_key[key]
        chunker = short_chunker.get(point["chunker"], point["chunker"])
        retriever = RETRIEVER_LABELS.get(point["retriever"], point["retriever"])
        ax.annotate(
            f"{chunker} {retriever}\n{point['ndcg']:.3f} @ {point['latency']:.0f} ms",
            (point["latency"], point["ndcg"]),
            textcoords="offset points",
            xytext=xytext,
            ha=ha,
            va=va,
            fontsize=font,
            color="#1a1a1a",
            linespacing=1.05,
            zorder=5,
            annotation_clip=False,
        )


def _headline(points: list[dict]) -> str:
    fixed = [point for point in points if point["chunker"] == "fixed"]
    rerank = next(point for point in fixed if point["retriever"] == "rerank")
    hybrid = next(point for point in fixed if point["retriever"] == "hybrid")
    return (
        f"Fixed + rerank nDCG@10 {rerank['ndcg']:.3f} at {rerank['latency']:.1f} ms. "
        f"Hybrid {hybrid['ndcg']:.3f} at {hybrid['latency']:.1f} ms."
    )


def _legend(ax, points: list[dict]) -> None:
    """One legend row per measured point, using that point's color and marker."""
    import matplotlib.pyplot as plt

    ordered = sorted(points, key=lambda point: (-point["ndcg"], point["latency"]))
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker=RETRIEVER_MARKERS.get(point["retriever"], "o"),
            color="none",
            markerfacecolor=CHUNKER_COLORS.get(point["chunker"], "#333333"),
            markeredgecolor="#222222",
            markersize=7,
            label=(
                f"{CHUNKER_LABELS.get(point['chunker'], point['chunker'])} · "
                f"{RETRIEVER_LABELS.get(point['retriever'], point['retriever'])}"
            ),
        )
        for point in ordered
    ]
    ax.legend(
        handles=handles,
        frameon=False,
        fontsize=7,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        handletextpad=0.4,
        borderaxespad=0.0,
    )


def _faithfulness(payload: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pairs = payload["generation"]["pairs"]
    if not pairs:
        raise SystemExit("generation metrics.json has no judged pairs")
    answered_only = any("num_refusals" in pair for pair in pairs)
    if answered_only:
        labels = [
            f"{pair['chunker']}\n{pair['retriever']}\nn={pair['num_scored']}" for pair in pairs
        ]
    else:
        labels = [f"{pair['chunker']}\n{pair['retriever']}" for pair in pairs]
    faithfulness = [float(pair["faithfulness"]) for pair in pairs]
    hallucination = [float(pair["hallucination_rate"]) for pair in pairs]
    judged = payload["generation"]["num_queries"]
    figure, axis = plt.subplots(figsize=(8.6, 4.6), dpi=160)
    figure.patch.set_facecolor("white")
    axis.set_facecolor("white")
    positions = np.arange(len(pairs))
    width = 0.36
    axis.bar(
        positions - width / 2,
        faithfulness,
        width=width,
        color="#0072B2",
        label="Faithfulness",
    )
    axis.bar(
        positions + width / 2,
        hallucination,
        width=width,
        color="#D55E00",
        label="Hallucination rate",
    )
    axis.set_xticks(positions)
    axis.set_xticklabels(labels)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score")
    model = payload["generation"].get("model", "")
    if answered_only:
        title = f"Answered queries, refusals excluded, {model}"
    else:
        title = f"Judge scores, n={judged} queries, {model}"
    axis.set_title(title, loc="left", fontsize=12, pad=10)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")
    width, height = struct.unpack(">II", data[16:24])
    return width, height


if __name__ == "__main__":
    main()
