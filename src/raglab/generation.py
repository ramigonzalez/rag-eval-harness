"""LLM-as-judge generation metrics, in the style of RAGAS.

Faithfulness is the fraction of answer claims the retrieved context supports.
Hallucination rate is the fraction the context does not support, so on the same
claims it is 1 minus faithfulness. Answer relevancy is the mean cosine
similarity between the user question and questions an LLM says the answer is
addressing. Similarities are clipped at 0 so the score stays in [0, 1].
Embeddings for that step are local. The LLM is used to write the answer, the
claims, and the proposed questions.

When ``answers_path`` is set, each judged query is appended to that JSONL file
(question, contexts, answer, claims, verdicts, and the three scores). A later
run with the same path reuses those rows instead of calling the model again.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from raglab.llm import BudgetExceeded, parse_json_object
from raglab.retrieval.index import SearchIndex, Services
from raglab.retrieval.registry import build_retriever
from raglab.types import Hit, Query

logger = logging.getLogger("raglab")

ANSWER_PROMPT = """Task: answer.

Answer the question using only the context. If the context is not enough,
say you do not know. Do not add facts from outside the context.

Question: {question}

Context:
{context}
"""

CLAIMS_PROMPT = """Task: extract claims.

Split the answer into short factual claims. Set supported to true only when
the context states that claim. Return JSON only:
{{"claims": [{{"claim": "...", "supported": true}}]}}

Question: {question}

Context:
{context}

Answer: {answer}
"""

QUESTIONS_PROMPT = """Task: propose questions.

Write 3 questions that this answer is addressing. Return JSON only:
{{"questions": ["...", "...", "..."]}}

Answer: {answer}
"""


def run_generation(
    indexes: dict[str, SearchIndex],
    queries: list[Query],
    pairs: list[tuple[str, str]],
    services: Services,
    *,
    k: int,
    max_queries: int,
    seed: int,
    answers_path: Path | None = None,
) -> dict:
    if services.llm is None or services.embedder is None:
        raise RuntimeError("generation metrics need an LLM and an embedder")
    chosen = _sample(queries, max_queries, seed)
    prior = _load_prior_answers(answers_path)
    answers_file = None
    if answers_path is not None:
        answers_path.parent.mkdir(parents=True, exist_ok=True)
        answers_file = answers_path.open("a", encoding="utf-8")
        logger.info("Writing per-query answers to %s (%s prior rows)", answers_path, len(prior))
    pair_reports = []
    budget_stopped = False
    budget_message = None
    try:
        for chunker_name, retriever_name in pairs:
            if chunker_name not in indexes:
                raise RuntimeError(f"generation pair uses unknown chunker '{chunker_name}'")
            import raglab.retrieval  # noqa: F401  registers built-in retrievers

            retriever = build_retriever(retriever_name)
            faithfulness: list[float] = []
            hallucination: list[float] = []
            relevancy: list[float] = []
            errors = 0
            empty_claims = 0
            skipped_budget = 0
            for query in chosen:
                key = (chunker_name, retriever_name, query.query_id)
                if key in prior:
                    record = prior[key]
                    if record.get("faithfulness") is not None:
                        faithfulness.append(float(record["faithfulness"]))
                    if record.get("hallucination_rate") is not None:
                        hallucination.append(float(record["hallucination_rate"]))
                    if record.get("answer_relevancy") is not None:
                        relevancy.append(float(record["answer_relevancy"]))
                    if record.get("error"):
                        errors += 1
                    if record.get("claims") == []:
                        empty_claims += 1
                    continue
                hits = retriever.retrieve(indexes[chunker_name], query.text, k, services)
                contexts = [hit.text for hit in hits]
                context = _format_context(hits)
                tokens_before = _usage_snapshot(services.llm)
                record = {
                    "query_id": query.query_id,
                    "chunker": chunker_name,
                    "retriever": retriever_name,
                    "question": query.text,
                    "contexts": contexts,
                    "answer": None,
                    "claims": [],
                    "proposed_questions": [],
                    "faithfulness": None,
                    "hallucination_rate": None,
                    "answer_relevancy": None,
                    "error": None,
                }
                try:
                    answer = services.llm.complete(
                        ANSWER_PROMPT.format(question=query.text, context=context)
                    ).strip()
                    record["answer"] = answer
                    claims = _parse_claims(
                        services.llm.complete(
                            CLAIMS_PROMPT.format(
                                question=query.text, context=context, answer=answer
                            )
                        )
                    )
                    record["claims"] = claims
                    if not claims:
                        empty_claims += 1
                    else:
                        supported = sum(1 for claim in claims if claim["supported"])
                        score = supported / len(claims)
                        record["faithfulness"] = score
                        record["hallucination_rate"] = 1.0 - score
                        faithfulness.append(score)
                        hallucination.append(1.0 - score)
                    questions = _questions(
                        services.llm.complete(QUESTIONS_PROMPT.format(answer=answer))
                    )
                    record["proposed_questions"] = questions
                    if questions:
                        rel = _answer_relevancy(query.text, questions, services.embedder)
                        record["answer_relevancy"] = rel
                        relevancy.append(rel)
                except BudgetExceeded as exc:
                    record["error"] = str(exc)
                    errors += 1
                    skipped_budget += 1
                    budget_stopped = True
                    budget_message = str(exc)
                    logger.warning(
                        "Budget cap hit during %s/%s query %s: %s",
                        chunker_name,
                        retriever_name,
                        query.query_id,
                        exc,
                    )
                    _finalize_record(record, services.llm, tokens_before)
                    _write_record(answers_file, record)
                    break
                except Exception as exc:
                    record["error"] = f"{type(exc).__name__}: {exc}"
                    errors += 1
                    logger.warning("Generation judge failed for %s (%s)", query.query_id, exc)
                _finalize_record(record, services.llm, tokens_before)
                _write_record(answers_file, record)
                prior[key] = record
            pair_reports.append(
                {
                    "chunker": chunker_name,
                    "retriever": retriever_name,
                    "num_queries": len(chosen),
                    "num_scored": len(faithfulness),
                    "num_errors": errors,
                    "num_empty_claims": empty_claims,
                    "num_skipped_budget": skipped_budget,
                    "faithfulness": _mean(faithfulness),
                    "answer_relevancy": _mean(relevancy),
                    "hallucination_rate": _mean(hallucination),
                    "context_k": k,
                }
            )
            if budget_stopped:
                break
    finally:
        if answers_file is not None:
            answers_file.close()
    status = "budget_exceeded" if budget_stopped else "ok"
    return {
        "status": status,
        "budget_message": budget_message,
        "model": getattr(services.llm, "model", None),
        "num_queries": len(chosen),
        "query_ids": [query.query_id for query in chosen],
        "context_k": k,
        "answers_path": str(answers_path) if answers_path is not None else None,
        "pairs": pair_reports,
    }


def _usage_snapshot(llm: object) -> dict:
    usage = getattr(llm, "usage", None)
    if usage is None:
        return {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cache_hits": 0}
    return {
        "calls": int(usage.calls),
        "prompt_tokens": int(usage.prompt_tokens),
        "completion_tokens": int(usage.completion_tokens),
        "cache_hits": int(usage.cache_hits),
    }


def _finalize_record(record: dict, llm: object, before: dict) -> None:
    after = _usage_snapshot(llm)
    record["prompt_tokens"] = after["prompt_tokens"] - before["prompt_tokens"]
    record["completion_tokens"] = after["completion_tokens"] - before["completion_tokens"]
    record["llm_calls"] = after["calls"] - before["calls"]
    estimated = getattr(llm, "estimated_usd", None)
    record["estimated_usd_cumulative"] = estimated() if callable(estimated) else None


def _write_record(handle, record: dict) -> None:
    if handle is None:
        return
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _load_prior_answers(path: Path | None) -> dict[tuple[str, str, str], dict]:
    prior: dict[tuple[str, str, str], dict] = {}
    if path is None or not path.exists():
        return prior
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["chunker"], row["retriever"], row["query_id"])
        prior[key] = row
    return prior


def _sample(queries: list[Query], max_queries: int, seed: int) -> list[Query]:
    if max_queries < 1:
        raise ValueError("generation max_queries must be >= 1")
    if len(queries) <= max_queries:
        return list(queries)
    import random

    rng = random.Random(seed + 1)
    chosen = rng.sample(list(queries), max_queries)
    chosen.sort(key=lambda query: query.query_id)
    return chosen


def _format_context(hits: list[Hit]) -> str:
    if not hits:
        return "(no context retrieved)"
    return "\n\n".join(f"[{index}] {hit.text}" for index, hit in enumerate(hits, start=1))


def _parse_claims(text: str) -> list[dict]:
    payload = parse_json_object(text)
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims was not a list")
    parsed: list[dict] = []
    for claim in claims:
        if not isinstance(claim, dict) or "supported" not in claim:
            raise ValueError("a claim is missing the supported field")
        claim_text = claim.get("claim", "")
        if not isinstance(claim_text, str):
            claim_text = str(claim_text)
        parsed.append({"claim": claim_text, "supported": _as_bool(claim["supported"])})
    return parsed


def _questions(text: str) -> list[str]:
    payload = parse_json_object(text)
    questions = payload.get("questions", [])
    if not isinstance(questions, list):
        raise ValueError("questions was not a list")
    return [
        question.strip() for question in questions if isinstance(question, str) and question.strip()
    ]


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "yes", "supported"}:
        return True
    if isinstance(value, str) and value.strip().lower() in {"false", "no", "unsupported"}:
        return False
    raise ValueError(f"cannot read supported value {value!r}")


def _answer_relevancy(question: str, questions: list[str], embedder: object) -> float:
    query_vector = np.asarray(embedder.embed_query(question), dtype=np.float32)
    query_vector = query_vector / max(float(np.linalg.norm(query_vector)), 1e-12)
    vectors = np.asarray(embedder.embed_documents(questions), dtype=np.float32)
    sims: list[float] = []
    for vector in vectors:
        vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
        sims.append(max(0.0, float(vector @ query_vector)))
    return sum(sims) / len(sims)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)
