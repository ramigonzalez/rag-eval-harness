"""Re-score faithfulness on saved answers with a chosen judge version.

Reads an answers.jsonl written by run_generation, keeps each answer, its
contexts, and its answer_relevancy, and replaces the claims and faithfulness
with the chosen judge. Exact "I do not know." answers are abstentions and are
not scored. Rows already in the output file are reused, so the run resumes.

    python scripts/rescore_faithfulness.py \
        --answers results/generation-100/answers.jsonl \
        --out-dir results/gen-100-judge-v2 --judge v2 --max-usd 0.9
    # smoke test on a subset: --only key.csv (columns query_id, chunker, retriever)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from raglab.generation import format_context_texts, judge_claims
from raglab.llm import BudgetExceeded, LLMClient, approximate_tokens

ABSTENTION = "I do not know."
IN_USD = 0.15
OUT_USD = 0.60


def is_abstention(answer: str | None) -> bool:
    return answer is not None and answer.strip() == ABSTENTION


def key_of(row: dict) -> tuple[str, str, str]:
    return (str(row["query_id"]), row["chunker"], row["retriever"])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--judge", default="v2")
    parser.add_argument("--only", type=Path, default=None)
    parser.add_argument("--max-usd", type=float, default=0.9)
    parser.add_argument("--estimate", action="store_true")
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.answers.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    todo = [row for row in rows if row.get("answer") and not is_abstention(row["answer"])]
    if args.only:
        wanted = {
            (row["query_id"], row["chunker"], row["retriever"])
            for row in csv.DictReader(args.only.open(encoding="utf-8"))
        }
        todo = [row for row in todo if key_of(row) in wanted]

    out_path = args.out_dir / "scores.jsonl"
    done: dict = {}
    if out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("error") is None:
                done[key_of(record)] = record
    pending = [row for row in todo if key_of(row) not in done]
    spent_before = sum(
        record["prompt_tokens"] * IN_USD / 1e6 + record["completion_tokens"] * OUT_USD / 1e6
        for record in done.values()
    )

    # Rough estimate: the verify call carries the context; the claim call does not.
    estimate_in = sum(
        approximate_tokens(format_context_texts(row["contexts"]))
        + 2 * approximate_tokens(row["answer"])
        + 650
        for row in pending
    )
    estimate_out = 250 * len(pending)
    estimate = estimate_in * IN_USD / 1e6 + estimate_out * OUT_USD / 1e6
    print(
        f"pending={len(pending)} done={len(done)} "
        f"spent_so_far=${spent_before:.4f} estimate=${estimate:.4f}"
    )
    if args.estimate:
        return 0
    if spent_before + estimate > args.max_usd:
        print("estimate above cap, stopping", file=sys.stderr)
        return 2

    llm = LLMClient()
    llm.set_budget(max(args.max_usd - spent_before, 0.0), IN_USD, OUT_USD)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(pending, 1):
            before = (llm.usage.prompt_tokens, llm.usage.completion_tokens, llm.usage.calls)
            record = {
                "query_id": row["query_id"],
                "chunker": row["chunker"],
                "retriever": row["retriever"],
                "judge": args.judge,
                "model": llm.model,
                "claims": [],
                "faithfulness": None,
                "hallucination_rate": None,
                "faithfulness_v1": row.get("faithfulness"),
                "answer_relevancy": row.get("answer_relevancy"),
                "error": None,
            }
            try:
                claims = judge_claims(
                    llm,
                    row["question"],
                    format_context_texts(row["contexts"]),
                    row["answer"],
                    args.judge,
                )
                record["claims"] = claims
                if claims:
                    score = sum(claim["supported"] for claim in claims) / len(claims)
                    record["faithfulness"] = score
                    record["hallucination_rate"] = 1.0 - score
            except BudgetExceeded as exc:
                print(f"budget cap hit: {exc}", file=sys.stderr)
                break
            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
            record["prompt_tokens"] = llm.usage.prompt_tokens - before[0]
            record["completion_tokens"] = llm.usage.completion_tokens - before[1]
            record["llm_calls"] = llm.usage.calls - before[2]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                spent = llm.estimated_usd()
                print(f"{index}/{len(pending)} spent this run ${spent:.4f}", flush=True)
    print(
        "done. this run: "
        f"calls={llm.usage.calls} in={llm.usage.prompt_tokens} "
        f"out={llm.usage.completion_tokens} usd=${llm.estimated_usd():.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
