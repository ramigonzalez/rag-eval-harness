"""Call counts and a pre-run token estimate for the LLM strategies.

HyDE and RAG-Fusion prompts depend only on the question, so the cache calls
the model once per query even though every chunker searches with the result.
Generation metrics are three calls per judged query per pair: answer, claims,
and the questions used for answer relevancy.
"""

from __future__ import annotations

from raglab.generation import ANSWER_PROMPT, CLAIMS_PROMPT, QUESTIONS_PROMPT
from raglab.llm import approximate_tokens
from raglab.retrieval.strategies import FUSION_PROMPT, HYDE_PROMPT

# Assumptions for the pre-run estimate. The logged line names them.
# A SciFact claim is a short sentence. A judged chunk is on the order of
# 120 words. The hypothetical passage is capped at 120 words in the prompt.
_QUERY_CHARS = 80
_CHUNK_CHARS = 600
_ANSWER_CHARS = 400
_HYDE_OUTPUT_CHARS = 600
_FUSION_OUTPUT_CHARS = 180
_CLAIMS_OUTPUT_CHARS = 500
_QUESTIONS_OUTPUT_CHARS = 180


def plan_llm_calls(
    *,
    retrieval_queries: int,
    generation_queries: int,
    retrievers: list[str],
    generation_pairs: list[tuple[str, str]],
    generation_enabled: bool,
) -> dict:
    hyde = retrieval_queries if "hyde" in retrievers else 0
    rag_fusion = retrieval_queries if "rag_fusion" in retrievers else 0
    judge = generation_queries * len(generation_pairs) * 3 if generation_enabled else 0
    return {
        "retrieval_queries": retrieval_queries,
        "generation_queries": generation_queries if generation_enabled else 0,
        "generation_pairs": len(generation_pairs) if generation_enabled else 0,
        "hyde_calls": hyde,
        "rag_fusion_calls": rag_fusion,
        "generation_calls": judge,
        "total_calls": hyde + rag_fusion + judge,
    }


def estimate_pre_run(
    plan: dict,
    *,
    context_k: int,
    rag_fusion_queries: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> dict:
    """Character-per-4 estimate of the planned calls. Not a measured bill."""
    question = "q" * _QUERY_CHARS
    context = "\n\n".join(f"[{index}] {'c' * _CHUNK_CHARS}" for index in range(1, context_k + 1))
    answer = "a" * _ANSWER_CHARS
    prompt_tokens = 0
    completion_tokens = 0
    if plan["hyde_calls"]:
        prompt_tokens += plan["hyde_calls"] * approximate_tokens(HYDE_PROMPT.format(query=question))
        completion_tokens += plan["hyde_calls"] * approximate_tokens("h" * _HYDE_OUTPUT_CHARS)
    if plan["rag_fusion_calls"]:
        prompt = FUSION_PROMPT.format(n=rag_fusion_queries, query=question)
        prompt_tokens += plan["rag_fusion_calls"] * approximate_tokens(prompt)
        completion_tokens += plan["rag_fusion_calls"] * approximate_tokens(
            "f" * _FUSION_OUTPUT_CHARS
        )
    per_pair = plan["generation_queries"]
    if per_pair and plan["generation_pairs"]:
        answer_prompt = ANSWER_PROMPT.format(question=question, context=context)
        claims_prompt = CLAIMS_PROMPT.format(question=question, context=context, answer=answer)
        questions_prompt = QUESTIONS_PROMPT.format(answer=answer)
        examples = per_pair * plan["generation_pairs"]
        prompt_tokens += examples * (
            approximate_tokens(answer_prompt)
            + approximate_tokens(claims_prompt)
            + approximate_tokens(questions_prompt)
        )
        completion_tokens += examples * (
            approximate_tokens(answer)
            + approximate_tokens("c" * _CLAIMS_OUTPUT_CHARS)
            + approximate_tokens("q" * _QUESTIONS_OUTPUT_CHARS)
        )
    estimated_usd = (prompt_tokens / 1_000_000) * input_usd_per_million + (
        completion_tokens / 1_000_000
    ) * output_usd_per_million
    return {
        "method": "characters_per_4",
        "assumptions": (
            f"{_QUERY_CHARS}-character questions, {_CHUNK_CHARS}-character chunks, "
            f"context_k={context_k}, about 4 characters per token"
        ),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "estimated_usd": round(estimated_usd, 6),
        "input_usd_per_million": input_usd_per_million,
        "output_usd_per_million": output_usd_per_million,
        **plan,
    }


def format_pre_run(estimate: dict) -> str:
    return (
        "Pre-run LLM estimate: "
        f"{estimate['total_calls']} calls "
        f"(HyDE {estimate['hyde_calls']}, RAG-Fusion {estimate['rag_fusion_calls']}, "
        f"generation {estimate['generation_calls']}) on "
        f"{estimate['retrieval_queries']} retrieval queries and "
        f"{estimate['generation_queries']} judged queries "
        f"across {estimate['generation_pairs']} pairs. "
        f"About {estimate['prompt_tokens']:,} input tokens and "
        f"{estimate['completion_tokens']:,} output tokens "
        f"(~${estimate['estimated_usd']:.4f} at "
        f"${estimate['input_usd_per_million']:.2f}/"
        f"${estimate['output_usd_per_million']:.2f} per million). "
        f"Method: {estimate['assumptions']}. Not a bill."
    )


def format_actual(usage: dict, provider: str, model: str) -> str:
    source = "includes character estimates" if usage["estimated_calls"] else "provider usage"
    return (
        f"LLM usage ({provider}, {model}): {usage['calls']} calls, "
        f"{usage['cache_hits']} cache hits, "
        f"{usage['prompt_tokens']:,} input tokens, "
        f"{usage['completion_tokens']:,} output tokens, "
        f"~${usage['estimated_usd']:.4f} at "
        f"${usage['input_usd_per_million']:.2f}/"
        f"${usage['output_usd_per_million']:.2f} per million ({source})."
    )
