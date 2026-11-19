#!/usr/bin/env python3
"""Score human labels against the two LLM judges.

Reads a filled labels.xlsx or labels.csv plus the sibling key.csv written by
make_sheet.py. Maps yes=1, partial=0.5, no=0. Writes analysis.json.

Usage:
    python analyze.py labels.xlsx
    python analyze.py labels.csv --key key.csv --out analysis.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

LABEL_MAP = {
    "yes": 1.0,
    "partial": 0.5,
    "no": 0.0,
}
AGREEMENT_TOLERANCE = 0.25
BINARY_THRESHOLD = 0.5


def load_csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_label_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_csv_rows(path)
    if suffix in {".xlsx", ".xlsm"}:
        from openpyxl import load_workbook

        wb = load_workbook(path, data_only=True, read_only=True)
        if "label" in wb.sheetnames:
            ws = wb["label"]
        else:
            ws = wb.active
        rows_iter = ws.iter_rows(values_only=True)
        header = [str(v).strip() if v is not None else "" for v in next(rows_iter)]
        rows = []
        for raw in rows_iter:
            if all(v is None or str(v).strip() == "" for v in raw):
                continue
            rows.append(
                {
                    header[i]: ("" if raw[i] is None else raw[i])
                    for i in range(len(header))
                }
            )
        wb.close()
        return rows
    raise SystemExit(f"Unsupported labels file: {path}")


def parse_item(value) -> int:
    return int(float(str(value).strip()))


def parse_label(value):
    if value is None:
        return None
    text = str(value).strip().lower()
    if text == "":
        return None
    if text not in LABEL_MAP:
        raise SystemExit(
            f"Invalid label {value!r}. Expected yes, partial, or no."
        )
    return LABEL_MAP[text]


def parse_score(value) -> float:
    number = float(value)
    if math.isnan(number):
        raise ValueError("score is NaN")
    return number


def rankdata_average(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    n = len(indexed)
    while i < n:
        j = i
        while j + 1 < n and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        # 1-based average ranks for ties
        avg = 0.5 * ((i + 1) + (j + 1))
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg
        i = j + 1
    return ranks


def pearson(xs: list[float], ys: list[float]):
    n = len(xs)
    if n < 2:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x == 0 or var_y == 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / math.sqrt(var_x * var_y)


def spearman(xs: list[float], ys: list[float]):
    if len(xs) < 2:
        return None
    return pearson(rankdata_average(xs), rankdata_average(ys))


def metric_block(human: list[float], judge: list[float]) -> dict:
    n = len(human)
    if n == 0:
        return {
            "n": 0,
            "mae": None,
            "agreement_within_0.25": None,
            "spearman": None,
            "accuracy_at_0.5": None,
        }
    abs_err = [abs(h - j) for h, j in zip(human, judge)]
    mae = sum(abs_err) / n
    agreement = sum(err <= AGREEMENT_TOLERANCE for err in abs_err) / n
    acc = (
        sum(
            (h >= BINARY_THRESHOLD) == (j >= BINARY_THRESHOLD)
            for h, j in zip(human, judge)
        )
        / n
    )
    rho = spearman(human, judge)
    return {
        "n": n,
        "mae": round(mae, 6),
        "agreement_within_0.25": round(agreement, 6),
        "spearman": None if rho is None else round(rho, 6),
        "accuracy_at_0.5": round(acc, 6),
    }


def join_rows(labels: list[dict], keys: list[dict]) -> list[dict]:
    key_index = {parse_item(row["item"]): row for row in keys}
    joined = []
    skipped_unlabeled = 0
    for row in labels:
        if "item" not in row or str(row["item"]).strip() == "":
            continue
        item = parse_item(row["item"])
        if item not in key_index:
            raise SystemExit(f"Item {item} is missing from key.csv")
        faithful = parse_label(row.get("faithful"))
        relevant = parse_label(row.get("relevant"))
        if faithful is None or relevant is None:
            skipped_unlabeled += 1
            continue
        key = key_index[item]
        joined.append(
            {
                "item": item,
                "human_faithful": faithful,
                "human_relevant": relevant,
                "ours_faithfulness": parse_score(key["ours_faithfulness"]),
                "ragas_faithfulness": parse_score(key["ragas_faithfulness"]),
                "ours_answer_relevancy": parse_score(key["ours_answer_relevancy"]),
                "ragas_answer_relevancy": parse_score(key["ragas_answer_relevancy"]),
            }
        )
    return joined, skipped_unlabeled


def analyze(joined: list[dict], skipped_unlabeled: int, n_label_rows: int) -> dict:
    human_f = [row["human_faithful"] for row in joined]
    human_r = [row["human_relevant"] for row in joined]
    return {
        "n_label_rows": n_label_rows,
        "n_labeled": len(joined),
        "n_unlabeled": skipped_unlabeled,
        "label_map": {"yes": 1.0, "partial": 0.5, "no": 0.0},
        "agreement_tolerance": AGREEMENT_TOLERANCE,
        "binary_threshold": BINARY_THRESHOLD,
        "accuracy_at_0.5_note": (
            "Both human and judge scores are treated as positive if >= 0.5 "
            "(yes or partial vs no; judge score >= 0.5 vs < 0.5)."
        ),
        "faithfulness": {
            "ours": metric_block(human_f, [row["ours_faithfulness"] for row in joined]),
            "ragas": metric_block(human_f, [row["ragas_faithfulness"] for row in joined]),
        },
        "relevancy": {
            "ours": metric_block(human_r, [row["ours_answer_relevancy"] for row in joined]),
            "ragas": metric_block(human_r, [row["ragas_answer_relevancy"] for row in joined]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path, help="Filled labels.xlsx or labels.csv")
    parser.add_argument(
        "--key",
        type=Path,
        default=None,
        help="Unblinding key (default: key.csv next to the labels file)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON (default: analysis.json next to the labels file)",
    )
    args = parser.parse_args()

    labels_path = args.labels.resolve()
    key_path = (args.key or labels_path.parent / "key.csv").resolve()
    out_path = (args.out or labels_path.parent / "analysis.json").resolve()

    if not labels_path.exists():
        raise SystemExit(f"Labels file not found: {labels_path}")
    if not key_path.exists():
        raise SystemExit(f"key.csv not found: {key_path}")

    label_rows = load_label_rows(labels_path)
    key_rows = load_csv_rows(key_path)
    joined, skipped = join_rows(label_rows, key_rows)
    result = analyze(joined, skipped, n_label_rows=len(label_rows))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    print(f"n_labeled: {result['n_labeled']} (unlabeled: {result['n_unlabeled']})")
    for family in ("faithfulness", "relevancy"):
        for judge, block in result[family].items():
            print(
                f"{family}/{judge}: mae={block['mae']} "
                f"agree@0.25={block['agreement_within_0.25']} "
                f"spearman={block['spearman']} "
                f"acc@0.5={block['accuracy_at_0.5']}"
            )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
