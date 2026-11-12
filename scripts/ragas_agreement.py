"""Compare this repo's judge to RAGAS on a saved answer file.

``results/generation-100/answers.jsonl`` stores one object per judged query
from the 100-query run. The earlier ``results/generation`` run does not.
Point ``--answers`` at a JSONL file with one object per line:

    {"question": "...", "answer": "...", "contexts": ["..."],
     "faithfulness": 0.8, "answer_relevancy": 0.5}

``faithfulness`` and ``answer_relevancy`` are this repo's scores. The script
asks RAGAS for the same two metrics and prints the mean absolute difference.
It does not run without an API key, without the ``ragas`` package, or without
the answers file, and it never invents scores. Agreement on the 100-query
file has not been computed in this repository.

    export OPENROUTER_API_KEY=...   # or OPENAI_API_KEY
    python scripts/ragas_agreement.py --answers results/generation-100/answers.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare our judge to RAGAS on saved answers.")
    parser.add_argument("--answers", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.answers.exists():
        print(
            f"No answers file at {args.answers}. "
            "Pass results/generation-100/answers.jsonl to score the 100-query run. "
            "No numbers were computed.",
            file=sys.stderr,
        )
        return 0
    from raglab.llm import llm_configured

    if not llm_configured():
        print(
            "No API key is set (OPENROUTER_API_KEY, OPENAI_API_KEY, or RAGBENCH_API_KEY). "
            "RAGAS was not called. No numbers were computed.",
            file=sys.stderr,
        )
        return 0
    try:
        import ragas  # noqa: F401
    except ImportError:
        print(
            "The ragas package is not installed. Install with pip install ragas. "
            "No numbers were computed.",
            file=sys.stderr,
        )
        return 0
    rows = _load_answers(args.answers)
    report = score_with_ragas(rows)
    print(json.dumps(report, indent=2))
    return 0


def _load_answers(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"{path} has no answer rows")
    return rows


def agreement(ours: list[float], other: list[float]) -> dict:
    if len(ours) != len(other) or not ours:
        raise ValueError("agreement needs two equal non-empty score lists")
    diffs = [abs(left - right) for left, right in zip(ours, other, strict=True)]
    return {"n": len(diffs), "mean_abs_diff": sum(diffs) / len(diffs)}


def score_with_ragas(rows: list[dict]) -> dict:
    """Call RAGAS. Imported only after the key and package checks above."""
    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, faithfulness

    dataset = Dataset.from_dict(
        {
            "question": [row["question"] for row in rows],
            "answer": [row["answer"] for row in rows],
            "contexts": [row["contexts"] for row in rows],
        }
    )
    scored = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
    frame = scored.to_pandas()
    faith = agreement(
        [float(row["faithfulness"]) for row in rows],
        [float(value) for value in frame["faithfulness"]],
    )
    relevancy = agreement(
        [float(row["answer_relevancy"]) for row in rows],
        [float(value) for value in frame["answer_relevancy"]],
    )
    return {"faithfulness": faith, "answer_relevancy": relevancy}


if __name__ == "__main__":
    raise SystemExit(main())
