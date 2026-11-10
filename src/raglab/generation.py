"""LLM-as-judge generation metrics, in the style of RAGAS.

Faithfulness is the fraction of answer claims the retrieved context supports.
Hallucination rate is the fraction the context does not support, so on the same
claims it is 1 minus faithfulness. Answer relevancy is the mean cosine
similarity between the user question and questions an LLM says the answer is
addressing. Similarities are clipped at 0 so the score stays in [0, 1].
Embeddings for that step are local. The LLM is used to write the answer, the
claims, and the proposed questions.
"""

from __future__ import annotations

import logging

import numpy as np

from raglab.llm import parse_json_object
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
) -> dict:
    if services.llm is None or services.embedder is None:
        raise RuntimeError("generation metrics need an LLM and an embedder")
    chosen = _sample(queries, max_queries, seed)
    pair_reports = []
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
        for query in chosen:
            hits = retriever.retrieve(indexes[chunker_name], query.text, k, services)
            context = _format_context(hits)
            try:
                answer = services.llm.complete(
                    ANSWER_PROMPT.format(question=query.text, context=context)
                ).strip()
                claims = _claims(
                    services.llm.complete(
                        CLAIMS_PROMPT.format(question=query.text, context=context, answer=answer)
                    )
                )
                if not claims:
                    empty_claims += 1
                else:
                    supported = sum(1 for claim in claims if claim)
                    score = supported / len(claims)
                    faithfulness.append(score)
                    hallucination.append(1.0 - score)
                questions = _questions(
                    services.llm.complete(QUESTIONS_PROMPT.format(answer=answer))
                )
                if questions:
                    relevancy.append(_answer_relevancy(query.text, questions, services.embedder))
            except Exception as exc:
                errors += 1
                logger.warning("Generation judge failed for %s (%s)", query.query_id, exc)
        pair_reports.append(
            {
                "chunker": chunker_name,
                "retriever": retriever_name,
                "num_queries": len(chosen),
                "num_scored": len(faithfulness),
                "num_errors": errors,
                "num_empty_claims": empty_claims,
                "faithfulness": _mean(faithfulness),
                "answer_relevancy": _mean(relevancy),
                "hallucination_rate": _mean(hallucination),
                "context_k": k,
            }
        )
    return {
        "status": "ok",
        "model": getattr(services.llm, "model", None),
        "num_queries": len(chosen),
        "query_ids": [query.query_id for query in chosen],
        "context_k": k,
        "pairs": pair_reports,
    }


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


def _claims(text: str) -> list[bool]:
    payload = parse_json_object(text)
    claims = payload.get("claims", [])
    if not isinstance(claims, list):
        raise ValueError("claims was not a list")
    verdicts: list[bool] = []
    for claim in claims:
        if not isinstance(claim, dict) or "supported" not in claim:
            raise ValueError("a claim is missing the supported field")
        verdicts.append(_as_bool(claim["supported"]))
    return verdicts


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
