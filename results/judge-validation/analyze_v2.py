#!/usr/bin/env python3
"""Compare old judge (v1), new judge (v2), and RAGAS faithfulness to reference labels.

Reuses analyze.py's metric code. v2 scores come from a scores.jsonl written by
scripts/rescore_faithfulness.py (joined on query_id, chunker, retriever).

    python analyze_v2.py reference_labels_opus.csv --v2 path/to/scores.jsonl --out analysis_opus_v2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import (  # noqa: E402
    LABEL_MAP,
    load_csv_rows,
    load_label_rows,
    metric_block,
    parse_item,
    parse_label,
)


CONTRADICTION_ITEMS = [3, 7, 22, 35, 36]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("labels", type=Path)
    ap.add_argument("--key", type=Path, default=Path(__file__).with_name("key.csv"))
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    keys = {parse_item(r["item"]): r for r in load_csv_rows(args.key)}
    v2 = {}
    for line in args.v2.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            v2[(str(rec["query_id"]), rec["chunker"], rec["retriever"])] = rec
    rows = []
    for lab in load_label_rows(args.labels):
        item = parse_item(lab["item"])
        k = keys[item]
        rec = v2.get((k["query_id"], k["chunker"], k["retriever"]))
        if rec is None or rec.get("faithfulness") is None:
            raise SystemExit(f"item {item} has no v2 faithfulness score")
        rows.append({
            "item": item,
            "reference_faithful": parse_label(lab["faithful"]),
            "reference_label": str(lab["faithful"]).strip().lower(),
            "old": float(k["ours_faithfulness"]),
            "new": float(rec["faithfulness"]),
            "ragas": float(k["ragas_faithfulness"]),
            "new_claims": rec["claims"],
        })
    ref = [r["reference_faithful"] for r in rows]
    result = {
        "reference": "reference_labels_opus.csv (Claude Opus labels, one LLM, not human)",
        "v2_scores": str(args.v2),
        "label_map": LABEL_MAP,
        "faithfulness": {
            "old_judge_v1": metric_block(ref, [r["old"] for r in rows]),
            "new_judge_v2": metric_block(ref, [r["new"] for r in rows]),
            "ragas": metric_block(ref, [r["ragas"] for r in rows]),
        },
        "contradiction_items": [
            {k: r[k] for k in ("item", "reference_label", "old", "new", "ragas", "new_claims")}
            for r in rows if r["item"] in CONTRADICTION_ITEMS
        ],
        "false_alarms_v2": [
            {k: r[k] for k in ("item", "reference_label", "old", "new", "ragas", "new_claims")}
            for r in rows if r["reference_label"] == "yes" and r["new"] < 0.5
        ],
        "false_alarms_v1": [r["item"] for r in rows if r["reference_label"] == "yes" and r["old"] < 0.5],
        "partial_items": [
            {k: r[k] for k in ("item", "old", "new", "ragas")} for r in rows if r["reference_label"] == "partial"
        ],
        "per_item": [{k: r[k] for k in ("item", "reference_label", "old", "new", "ragas")} for r in rows],
    }
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, b in result["faithfulness"].items():
        print(f"{name}: mae={b['mae']} agree@0.25={b['agreement_within_0.25']} spearman={b['spearman']} acc@0.5={b['accuracy_at_0.5']}")
    for r in result["contradiction_items"]:
        print("contradiction", r["item"], r["old"], "->", round(r["new"], 3), [c["verdict"] for c in r["new_claims"]])
    print("false alarms v2:", [(r["item"], round(r["new"], 3)) for r in result["false_alarms_v2"]])
    print("false alarms v1:", result["false_alarms_v1"])
    print("partials:", [(r["item"], round(r["old"], 2), round(r["new"], 2)) for r in result["partial_items"]])


if __name__ == "__main__":
    main()
