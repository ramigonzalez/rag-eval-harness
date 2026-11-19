"""Build answers.jsonl, metrics, and results.md for a rescored faithfulness run.

    python scripts/summarize_rescore.py \
        --answers results/generation-100/answers.jsonl \
        --out-dir results/gen-100-judge-v2
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ABSTENTION = "I do not know."
IN_USD = 0.15
OUT_USD = 0.60
ORDER = [
    ("whole", "bm25"),
    ("whole", "rerank"),
    ("fixed", "rerank"),
    ("recursive", "rerank"),
    ("semantic", "rerank"),
]


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = [
        json.loads(line)
        for line in args.answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scores = {
        (score["query_id"], score["chunker"], score["retriever"]): score
        for score in (
            json.loads(line)
            for line in (args.out_dir / "scores.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    merged = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        record = dict(row)
        record["abstained"] = bool(row.get("answer")) and row["answer"].strip() == ABSTENTION
        record["faithfulness_v1"] = row.get("faithfulness")
        record["claims_v1"] = row.get("claims")
        score = scores.get((row["query_id"], row["chunker"], row["retriever"]))
        if record["abstained"] or not row.get("answer"):
            record["faithfulness"] = None
            record["hallucination_rate"] = None
            record["claims"] = []
            if record["abstained"]:
                record["answer_relevancy"] = None
        else:
            record["faithfulness"] = score["faithfulness"]
            record["hallucination_rate"] = score["hallucination_rate"]
            record["claims"] = score["claims"]
        record["faithfulness_judge"] = "v2"
        for key in (
            "prompt_tokens",
            "completion_tokens",
            "llm_calls",
            "estimated_usd_cumulative",
        ):
            record.pop(key, None)
        merged.append(record)
        groups[(row["chunker"], row["retriever"])].append(record)
    with (args.out_dir / "answers.jsonl").open("w", encoding="utf-8") as handle:
        for record in merged:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    pairs = []
    for chunker, retriever in ORDER:
        group = groups[(chunker, retriever)]
        answered = [row for row in group if row.get("answer") and not row["abstained"]]
        faith = [row["faithfulness"] for row in answered if row["faithfulness"] is not None]
        faith_v1 = [
            row["faithfulness_v1"] for row in answered if row["faithfulness_v1"] is not None
        ]
        relevancy = [
            row["answer_relevancy"] for row in answered if row["answer_relevancy"] is not None
        ]
        pairs.append(
            {
                "chunker": chunker,
                "retriever": retriever,
                "num_queries": len(group),
                "num_abstained": sum(row["abstained"] for row in group),
                "refusal_rate": sum(row["abstained"] for row in group) / len(group),
                "num_answered": len(answered),
                "num_faithfulness_scored": len(faith),
                "faithfulness": mean(faith),
                "hallucination_rate": None if not faith else 1 - mean(faith),
                "answer_relevancy": mean(relevancy),
                "faithfulness_v1_answered_only": mean(faith_v1),
                "num_faithfulness_v1_scored": len(faith_v1),
            }
        )
    scored = list(scores.values())
    prompt_tokens = sum(row["prompt_tokens"] for row in scored)
    completion_tokens = sum(row["completion_tokens"] for row in scored)
    calls = sum(row["llm_calls"] for row in scored)
    usage = {
        "model": scored[0]["model"],
        "judge": "v2",
        "scored_rows": len(scored),
        "llm_calls": calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "input_usd_per_million": IN_USD,
        "output_usd_per_million": OUT_USD,
        "estimated_usd": round(prompt_tokens * IN_USD / 1e6 + completion_tokens * OUT_USD / 1e6, 6),
        "note": (
            "Token counts from OpenRouter responses. Identical prompts were served from an "
            "in-process cache and cost nothing. USD uses list rates, not the OpenRouter invoice."
        ),
    }
    (args.out_dir / "usage.json").write_text(json.dumps(usage, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "source_answers": str(args.answers),
        "faithfulness_judge": "v2",
        "judge_model": usage["model"],
        "temperature": 0,
        "context_k": 5,
        "abstention": (
            f"exact answer '{ABSTENTION}' excluded from faithfulness, hallucination rate, relevancy"
        ),
        "answer_relevancy": (
            "unchanged from the v1 run (same answers, same proposed-question step); "
            "abstentions excluded"
        ),
        "pairs": pairs,
        "usage": usage,
    }
    (args.out_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    with (args.out_dir / "metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(pairs[0].keys()))
        writer.writeheader()
        writer.writerows(pairs)

    def fmt(value: float | None) -> str:
        if value is None:
            return "-"
        return f"{value:.3f}"

    lines = [
        "# Generation, faithfulness judge v2",
        "",
        (
            "Same 500 answers as `results/generation-100/answers.jsonl` "
            "(100 SciFact queries x 5 stacks, context_k=5). "
            "Only faithfulness was re-scored, with judge v2 "
            f"(`{usage['model']}`, temperature 0): claims keep the answer's yes/no "
            "verdict and negations, then each claim is checked against the context "
            "as supported, contradicted, or not_found. "
            "Answer relevancy is carried over from the v1 run."
        ),
        "",
        (
            "Answered-only: an exact `I do not know.` is an abstention and is excluded "
            "from faithfulness, hallucination rate, and relevancy."
        ),
        "",
        (
            "| Chunker | Retriever | Refusal rate | Answered | Faithfulness v2 | "
            "Hallucination rate v2 | Answer relevancy | Faithfulness v1 |"
        ),
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in pairs:
        lines.append(
            "| {chunker} | {retriever} | {refusal_rate:.2f} | {num_answered} | "
            "{faith} | {hall} | {relevancy} | {faith_v1} |".format(
                chunker=pair["chunker"],
                retriever=pair["retriever"],
                refusal_rate=pair["refusal_rate"],
                num_answered=pair["num_answered"],
                faith=fmt(pair["faithfulness"]),
                hall=fmt(pair["hallucination_rate"]),
                relevancy=fmt(pair["answer_relevancy"]),
                faith_v1=fmt(pair["faithfulness_v1_answered_only"]),
            )
        )
    lines.extend(
        [
            "",
            (
                f"Re-scoring cost: {calls} LLM calls, {prompt_tokens} input and "
                f"{completion_tokens} output tokens, about ${usage['estimated_usd']:.4f} "
                "at $0.15/$0.60 per million."
            ),
            "",
        ]
    )
    text = "\n".join(lines)
    (args.out_dir / "results.md").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
