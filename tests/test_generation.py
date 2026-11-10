import numpy as np

from raglab.generation import run_generation
from raglab.llm import SKIP_REASON, parse_json_object
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
