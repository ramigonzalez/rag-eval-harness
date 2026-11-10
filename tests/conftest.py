"""Keep unit tests off the network even if a key is present in the shell."""

import pytest

_LLM_ENV = (
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "RAGBENCH_API_KEY",
    "RAGBENCH_API_BASE",
    "RAGBENCH_LLM_MODEL",
    "RAGBENCH_HTTP_REFERER",
)


@pytest.fixture(autouse=True)
def _clear_llm_env(monkeypatch):
    for name in _LLM_ENV:
        monkeypatch.delenv(name, raising=False)
