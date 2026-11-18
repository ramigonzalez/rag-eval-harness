import json
import math
from pathlib import Path

import numpy as np

from raglab.generation import is_abstention, run_generation, summarize_answer_records
from raglab.llm import SKIP_REASON, BudgetExceeded, parse_json_object
from raglab.retrieval.index import SearchIndex, Services
from raglab.types import Chunk, Query


class _ScriptedLLM:
    model = "fake-model"

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if "Task: answer." in prompt:
            question = prompt.split("Question:", 1)[1].split("Context:", 1)[0]
            if "The ice is old." in question:
                return "The ice is old."
            return "The capital of Mars is dust."
        if "Task: extract claims." in prompt:
            supported = "The ice is old." in prompt.split("Answer:", 1)[-1] and (
                "The ice is old." in prompt.split("Answer:", 1)[0]
            )
            return '{"claims": [{"claim": "The ice is old.", "supported": %s}]}' % (
                "true" if supported else "false"
            )
        if "Task: propose questions." in prompt:
            return '{"questions": ["How old is the ice?", "What do ice cores show?"]}'
        raise AssertionError(prompt)


class _Embedder:
    model_name = "hash"
    dim = 32

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                matrix[row, hash(token) % self.dim] += 1.0
        return matrix

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


def test_parse_json_object_accepts_a_fence():
    payload = parse_json_object('```json\n{"queries": ["a"]}\n```')
    assert payload == {"queries": ["a"]}


def test_judge_scores_supported_and_unsupported_claims():
    chunks = [Chunk("ice::c0", "ice", "The ice is old.", 0)]
    index = SearchIndex(
        chunks=chunks,
        embeddings=np.ones((1, 32), dtype=np.float32),
        bm25=None,
    )
    services = Services(embedder=_Embedder(), llm=_ScriptedLLM())
    supported = run_generation(
        {"whole": index},
        [Query("q1", "The ice is old.")],
        [("whole", "dense")],
        services,
        k=1,
        max_queries=5,
        seed=1,
    )
    pair = supported["pairs"][0]
    assert pair["faithfulness"] == 1.0
    assert pair["hallucination_rate"] == 0.0
    assert 0.0 <= pair["answer_relevancy"] <= 1.0

    unsupported = run_generation(
        {"whole": index},
        [Query("q2", "unrelated question about ships")],
        [("whole", "dense")],
        services,
        k=1,
        max_queries=5,
        seed=1,
    )
    assert unsupported["pairs"][0]["faithfulness"] == 0.0
    assert unsupported["pairs"][0]["hallucination_rate"] == 1.0


def test_skip_reason_tells_the_reader_to_set_a_key():
    assert "run with API key" in SKIP_REASON


def test_answers_jsonl_is_written_and_reused(tmp_path):
    chunks = [Chunk("ice::c0", "ice", "The ice is old.", 0)]
    index = SearchIndex(
        chunks=chunks,
        embeddings=np.ones((1, 32), dtype=np.float32),
        bm25=None,
    )
    services = Services(embedder=_Embedder(), llm=_ScriptedLLM())
    path = tmp_path / "answers.jsonl"
    queries = [Query("q1", "The ice is old.")]
    first = run_generation(
        {"whole": index},
        queries,
        [("whole", "dense")],
        services,
        k=1,
        max_queries=5,
        seed=1,
        answers_path=path,
    )
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 1
    assert rows[0]["question"] == "The ice is old."
    assert rows[0]["contexts"] == ["The ice is old."]
    assert rows[0]["claims"][0]["claim"]
    assert rows[0]["claims"][0]["supported"] is True
    assert rows[0]["answer"] == "The ice is old."
    assert first["answers_path"] == str(path)

    class _ExplodingLLM(_ScriptedLLM):
        def complete(self, prompt: str, temperature: float = 0.0) -> str:
            raise AssertionError("a saved answer should not call the model again")

    second = run_generation(
        {"whole": index},
        queries,
        [("whole", "dense")],
        Services(embedder=_Embedder(), llm=_ExplodingLLM()),
        k=1,
        max_queries=5,
        seed=1,
        answers_path=path,
    )
    assert second["pairs"][0]["faithfulness"] == 1.0
    again = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(again) == 1


def test_budget_cap_stops_the_judge_and_keeps_the_row(tmp_path):
    chunks = [Chunk("ice::c0", "ice", "The ice is old.", 0)]
    index = SearchIndex(
        chunks=chunks,
        embeddings=np.ones((1, 32), dtype=np.float32),
        bm25=None,
    )

    class _CappedLLM(_ScriptedLLM):
        def complete(self, prompt: str, temperature: float = 0.0) -> str:
            raise BudgetExceeded("Estimated spend $2.10 passed the $2.00 cap")

    path = tmp_path / "answers.jsonl"
    result = run_generation(
        {"whole": index},
        [Query("q1", "The ice is old."), Query("q2", "another claim")],
        [("whole", "dense")],
        Services(embedder=_Embedder(), llm=_CappedLLM()),
        k=1,
        max_queries=5,
        seed=1,
        answers_path=path,
    )
    assert result["status"] == "budget_exceeded"
    assert result["pairs"][0]["num_errors"] == 1
    assert result["pairs"][0]["num_skipped_budget"] == 1
    assert result["pairs"][0]["faithfulness"] is None
    rows = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    assert len(rows) == 1
    assert "cap" in rows[0]["error"]


def test_exact_refusal_is_an_abstention_and_skips_the_claim_judge(tmp_path):
    chunks = [Chunk("ice::c0", "ice", "The ice is old.", 0)]
    index = SearchIndex(
        chunks=chunks,
        embeddings=np.ones((1, 32), dtype=np.float32),
        bm25=None,
    )

    class _RefusalLLM(_ScriptedLLM):
        def complete(self, prompt: str, temperature: float = 0.0) -> str:
            if "Task: answer." in prompt:
                if "ships" in prompt:
                    return "I do not know."
                return "The ice is old."
            if "I do not know." in prompt:
                raise AssertionError("an abstention must not be split into claims or questions")
            return super().complete(prompt, temperature)

    result = run_generation(
        {"whole": index},
        [Query("q1", "The ice is old."), Query("q2", "unrelated question about ships")],
        [("whole", "dense")],
        Services(embedder=_Embedder(), llm=_RefusalLLM()),
        k=1,
        max_queries=5,
        seed=1,
        answers_path=tmp_path / "answers.jsonl",
    )
    pair = result["pairs"][0]
    assert pair["num_refusals"] == 1
    assert pair["refusal_rate"] == 0.5
    assert pair["num_answered"] == 1
    assert pair["num_scored"] == 1
    assert pair["faithfulness"] == 1.0
    assert pair["hallucination_rate"] == 0.0
    rows = [
        json.loads(line)
        for line in (tmp_path / "answers.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    refusal = next(row for row in rows if row["query_id"] == "q2")
    assert refusal["abstained"] is True
    assert refusal["claims"] == []
    assert refusal["faithfulness"] is None
    assert is_abstention("I do not know.")
    assert not is_abstention("I do not know")


def test_stored_refusal_scores_stay_out_of_the_answered_mean():
    """Older rows scored 'I do not know.' as unfaithful. Those scores are all_rows only."""
    records = [
        {
            "answer": "I do not know.",
            "claims": [{"claim": "restated", "supported": False}],
            "faithfulness": 0.0,
            "hallucination_rate": 1.0,
            "answer_relevancy": 0.02,
        },
        {
            "answer": "The ice is old.",
            "claims": [{"claim": "The ice is old.", "supported": True}],
            "faithfulness": 1.0,
            "hallucination_rate": 0.0,
            "answer_relevancy": 0.8,
        },
    ]
    pair = summarize_answer_records(
        records,
        chunker="whole",
        retriever="bm25",
        num_queries=2,
        context_k=5,
    )
    assert pair["faithfulness"] == 1.0
    assert pair["hallucination_rate"] == 0.0
    assert pair["answer_relevancy"] == 0.8
    assert pair["num_scored"] == 1
    assert pair["num_refusals"] == 1
    assert pair["all_rows"]["faithfulness"] == 0.5
    assert pair["all_rows"]["hallucination_rate"] == 0.5
    assert pair["all_rows"]["num_scored"] == 2


def test_published_generation_100_matches_saved_verdicts():
    root = Path(__file__).resolve().parents[1] / "results" / "generation-100"
    recorded = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    grouped: dict[tuple[str, str], list[dict]] = {}
    for line in (root / "answers.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        grouped.setdefault((row["chunker"], row["retriever"]), []).append(row)
    for pair in recorded["generation"]["pairs"]:
        summary = summarize_answer_records(
            grouped[(pair["chunker"], pair["retriever"])],
            chunker=pair["chunker"],
            retriever=pair["retriever"],
            num_queries=pair["num_queries"],
            context_k=pair["context_k"],
            num_skipped_budget=pair["num_skipped_budget"],
        )
        assert _same_number(pair["faithfulness"], summary["faithfulness"])
        assert _same_number(pair["hallucination_rate"], summary["hallucination_rate"])
        assert _same_number(pair["answer_relevancy"], summary["answer_relevancy"])
        assert pair["num_scored"] == summary["num_scored"]
        assert pair["num_refusals"] == summary["num_refusals"]
        assert _same_number(pair["refusal_rate"], summary["refusal_rate"])
        assert pair["all_rows"].keys() == summary["all_rows"].keys()
        for key, value in summary["all_rows"].items():
            assert _same_number(pair["all_rows"][key], value)


def _same_number(stored: object, fresh: object) -> bool:
    """JSON round-trip can move a mean by one ulp across Python versions."""
    if isinstance(fresh, float):
        return math.isclose(float(stored), fresh, rel_tol=0.0, abs_tol=1e-12)
    return stored == fresh
