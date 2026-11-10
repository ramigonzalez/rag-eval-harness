import json
import urllib.request

from raglab.cost import estimate_pre_run, format_pre_run, plan_llm_calls
from raglab.llm import LLMClient, approximate_tokens, resolve_llm_settings


def test_openrouter_key_selects_the_openrouter_endpoint(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    settings = resolve_llm_settings()
    assert settings.provider == "openrouter"
    assert settings.api_base == "https://openrouter.ai/api/v1"
    assert settings.model == "openai/gpt-4o-mini"
    assert settings.api_key == "sk-or-test"


def test_openai_key_keeps_the_openai_endpoint(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings = resolve_llm_settings()
    assert settings.provider == "openai"
    assert settings.api_base == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"


def test_openrouter_wins_when_both_provider_keys_are_set(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    settings = resolve_llm_settings()
    assert settings.provider == "openrouter"
    assert settings.api_key == "sk-or-test"


def test_env_overrides_change_model_and_base(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("RAGBENCH_LLM_MODEL", "openai/gpt-4.1-mini")
    monkeypatch.setenv("RAGBENCH_API_BASE", "https://example.test/v1")
    settings = resolve_llm_settings()
    assert settings.model == "openai/gpt-4.1-mini"
    assert settings.api_base == "https://example.test/v1"


def test_complete_records_provider_usage_and_caches(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(
                {
                    "choices": [{"message": {"content": "a glacier core"}}],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4},
                }
            ).encode()

    def _urlopen(request, timeout=0):
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())
        captured["headers"] = {key.lower(): value for key, value in request.header_items()}
        captured["calls"] = captured.get("calls", 0) + 1
        return _Response()

    monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
    client = LLMClient()
    assert client.complete("How old is the ice?") == "a glacier core"
    assert client.complete("How old is the ice?") == "a glacier core"
    assert captured["calls"] == 1
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["body"]["model"] == "openai/gpt-4o-mini"
    assert captured["headers"]["authorization"] == "Bearer sk-or-test"
    assert captured["headers"]["x-title"] == "RAG Lab"
    assert client.usage.calls == 1
    assert client.usage.cache_hits == 1
    assert client.usage.prompt_tokens == 12
    assert client.usage.completion_tokens == 4
    assert client.usage.estimated_calls == 0


def test_missing_usage_falls_back_to_character_estimate(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "abcd"}}]}).encode()

    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout=0: _Response())
    client = LLMClient()
    assert client.complete("hello") == "abcd"
    assert client.usage.estimated_calls == 1
    assert client.usage.completion_tokens == approximate_tokens("abcd")


def test_plan_counts_one_hyde_call_per_query_not_per_chunker():
    plan = plan_llm_calls(
        retrieval_queries=50,
        generation_queries=20,
        retrievers=["bm25", "dense", "hybrid", "rerank", "hyde", "rag_fusion"],
        generation_pairs=[("whole", "bm25"), ("fixed", "rerank")],
        generation_enabled=True,
    )
    assert plan["hyde_calls"] == 50
    assert plan["rag_fusion_calls"] == 50
    assert plan["generation_calls"] == 20 * 2 * 3
    assert plan["total_calls"] == 50 + 50 + 120
    estimate = estimate_pre_run(
        plan,
        context_k=5,
        rag_fusion_queries=3,
        input_usd_per_million=0.15,
        output_usd_per_million=0.60,
    )
    assert estimate["prompt_tokens"] > 0
    assert estimate["estimated_usd"] > 0
    text = format_pre_run(estimate)
    assert "50" in text
    assert "Not a bill" in text
