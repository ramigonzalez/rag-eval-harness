"""OpenAI-compatible chat client used only when an API key is set.

No SDK dependency: the benchmark stays installable, and runnable, without a key.
Set ``RAGBENCH_API_KEY`` or ``OPENAI_API_KEY``. Optional ``RAGBENCH_API_BASE``
(default ``https://api.openai.com/v1``) and ``RAGBENCH_LLM_MODEL`` (default
``gpt-4o-mini``).
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


SKIP_REASON = (
    "Generation metrics, HyDE, and RAG-Fusion did not run. "
    "run with API key: set RAGBENCH_API_KEY or OPENAI_API_KEY. "
    "Optional RAGBENCH_API_BASE (default https://api.openai.com/v1) and "
    "RAGBENCH_LLM_MODEL (default gpt-4o-mini)."
)


def llm_configured() -> bool:
    return bool(os.environ.get("RAGBENCH_API_KEY") or os.environ.get("OPENAI_API_KEY"))


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 60,
    ) -> None:
        self.model = model or os.environ.get("RAGBENCH_LLM_MODEL", "gpt-4o-mini")
        self.api_key = (
            api_key or os.environ.get("RAGBENCH_API_KEY") or os.environ.get("OPENAI_API_KEY")
        )
        self.api_base = (
            api_base or os.environ.get("RAGBENCH_API_BASE") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        if not self.api_key:
            raise LLMNotConfigured(SKIP_REASON)
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": 800,
            "messages": [
                {
                    "role": "system",
                    "content": "Follow the user's output format exactly.",
                },
                {"role": "user", "content": prompt},
            ],
        }
        request = urllib.request.Request(
            f"{self.api_base}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read().decode())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise LLMError(f"LLM request failed ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"LLM request failed: {exc.reason}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response did not include a message") from exc


def parse_json_object(text: str) -> dict:
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if match is None:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value
