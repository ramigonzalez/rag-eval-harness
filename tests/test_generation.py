import json

import numpy as np

from raglab.generation import run_generation
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
