"""OpenAI-compatible chat client used only when an API key is set.

No SDK dependency. The benchmark stays installable without a key.

Key selection, first match wins:

- ``RAGBENCH_API_KEY`` uses ``RAGBENCH_API_BASE`` or ``https://api.openai.com/v1``,
  and ``RAGBENCH_LLM_MODEL`` or ``gpt-4o-mini``.
- ``OPENROUTER_API_KEY`` uses ``https://openrouter.ai/api/v1`` and
  ``openai/gpt-4o-mini`` unless those two variables override them.
- ``OPENAI_API_KEY`` uses ``https://api.openai.com/v1`` and ``gpt-4o-mini``.

``RAGBENCH_API_BASE`` and ``RAGBENCH_LLM_MODEL`` override the provider defaults
whenever they are set.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMNotConfigured(RuntimeError):
    pass


class LLMError(RuntimeError):
    pass


class BudgetExceeded(LLMError):
    pass


OPENAI_BASE = "https://api.openai.com/v1"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"
OPENAI_DEFAULT_MODEL = "gpt-4o-mini"
OPENROUTER_DEFAULT_MODEL = "openai/gpt-4o-mini"

SKIP_REASON = (
    "Generation metrics, HyDE, and RAG-Fusion did not run. "
    "run with API key: set OPENROUTER_API_KEY "
    "(https://openrouter.ai/api/v1, default model openai/gpt-4o-mini) "
    "or OPENAI_API_KEY. "
    "RAGBENCH_API_KEY, RAGBENCH_API_BASE, and RAGBENCH_LLM_MODEL override those defaults."
)


@dataclass
class TokenUsage:
    """Token counts from the provider. Estimates are used only if usage is missing."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_calls: int = 0

    def as_dict(self, input_usd_per_million: float, output_usd_per_million: float) -> dict:
        estimated_usd = (self.prompt_tokens / 1_000_000) * input_usd_per_million + (
            self.completion_tokens / 1_000_000
        ) * output_usd_per_million
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "estimated_calls": self.estimated_calls,
            "input_usd_per_million": input_usd_per_million,
            "output_usd_per_million": output_usd_per_million,
            "estimated_usd": round(estimated_usd, 6),
        }


@dataclass(frozen=True)
class LLMSettings:
    api_key: str | None
    api_base: str
    model: str
    provider: str


def llm_configured() -> bool:
    return resolve_llm_settings().api_key is not None


def resolve_llm_settings(
    model: str | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> LLMSettings:
    env_model = os.environ.get("RAGBENCH_LLM_MODEL") or None
    env_base = os.environ.get("RAGBENCH_API_BASE") or None
    ragbench_key = os.environ.get("RAGBENCH_API_KEY") or None
    openrouter_key = os.environ.get("OPENROUTER_API_KEY") or None
    openai_key = os.environ.get("OPENAI_API_KEY") or None

    if api_key:
        provider = "explicit"
        key = api_key
        default_base = OPENAI_BASE
        default_model = OPENAI_DEFAULT_MODEL
    elif ragbench_key:
        provider = "ragbench"
        key = ragbench_key
        default_base = OPENAI_BASE
        default_model = OPENAI_DEFAULT_MODEL
    elif openrouter_key:
        provider = "openrouter"
        key = openrouter_key
        default_base = OPENROUTER_BASE
        default_model = OPENROUTER_DEFAULT_MODEL
    elif openai_key:
        provider = "openai"
        key = openai_key
        default_base = OPENAI_BASE
        default_model = OPENAI_DEFAULT_MODEL
    else:
        provider = "unconfigured"
        key = None
        default_base = OPENAI_BASE
        default_model = OPENAI_DEFAULT_MODEL

    resolved_base = (api_base or env_base or default_base).rstrip("/")
    resolved_model = model or env_model or default_model
    return LLMSettings(
        api_key=key,
        api_base=resolved_base,
        model=resolved_model,
        provider=provider,
    )


def approximate_tokens(text: str) -> int:
    """English rule of thumb: about four characters per token."""
    return max(1, (len(text) + 3) // 4)


class LLMClient:
    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        api_base: str | None = None,
        timeout: float = 90,
    ) -> None:
        settings = resolve_llm_settings(model=model, api_key=api_key, api_base=api_base)
        self.model = settings.model
        self.api_key = settings.api_key
        self.api_base = settings.api_base
        self.provider = settings.provider
        self.timeout = timeout
        self.usage = TokenUsage()
        self._cache: dict[tuple[str, float, str], str] = {}
        self.max_estimated_usd: float | None = None
        self.input_usd_per_million = 0.15
        self.output_usd_per_million = 0.60

    def set_budget(
        self,
        max_estimated_usd: float | None,
        input_usd_per_million: float,
        output_usd_per_million: float,
    ) -> None:
        self.max_estimated_usd = max_estimated_usd
        self.input_usd_per_million = input_usd_per_million
        self.output_usd_per_million = output_usd_per_million

    def estimated_usd(self) -> float:
        return (self.usage.prompt_tokens / 1_000_000) * self.input_usd_per_million + (
            self.usage.completion_tokens / 1_000_000
        ) * self.output_usd_per_million

    def _raise_if_over_budget(self) -> None:
        if self.max_estimated_usd is None:
            return
        spent = self.estimated_usd()
        if spent > self.max_estimated_usd:
            raise BudgetExceeded(
                f"Estimated spend ${spent:.4f} passed the ${self.max_estimated_usd:.2f} cap "
                f"after {self.usage.calls} calls "
                f"({self.usage.prompt_tokens} input, {self.usage.completion_tokens} output tokens)."
            )

    def complete(self, prompt: str, temperature: float = 0.0, *, use_cache: bool = True) -> str:
        if not self.api_key:
            raise LLMNotConfigured(SKIP_REASON)
        self._raise_if_over_budget()
        cache_key = (self.model, float(temperature), prompt)
        if use_cache and cache_key in self._cache:
            self.usage.cache_hits += 1
            return self._cache[cache_key]
        text, prompt_tokens, completion_tokens, estimated = self._post(prompt, temperature)
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        if estimated:
            self.usage.estimated_calls += 1
        if use_cache:
            self._cache[cache_key] = text
        self._raise_if_over_budget()
        return text

    def _post(self, prompt: str, temperature: float) -> tuple[str, int, int, bool]:
        system = "Follow the user's output format exactly."
        payload = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": 800,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        if self.provider == "openrouter" or self.api_base.startswith("https://openrouter.ai"):
            headers["X-Title"] = "RAG Lab"
            referer = os.environ.get("RAGBENCH_HTTP_REFERER")
            if referer:
                headers["HTTP-Referer"] = referer
        body = None
        last_error: Exception | None = None
        for attempt in range(6):
            request = urllib.request.Request(
                f"{self.api_base}/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:500]
                last_error = LLMError(f"LLM request failed ({exc.code}): {detail}")
                if exc.code in {429, 500, 502, 503, 529} and attempt < 5:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise last_error from exc
            except urllib.error.URLError as exc:
                last_error = LLMError(f"LLM request failed: {exc.reason}")
                if attempt < 5:
                    time.sleep(min(2**attempt, 30))
                    continue
                raise last_error from exc
        if body is None:
            raise last_error or LLMError("LLM request failed")
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("LLM response did not include a message") from exc
        usage = body.get("usage") if isinstance(body, dict) else None
        if isinstance(usage, dict) and "prompt_tokens" in usage and "completion_tokens" in usage:
            return text, int(usage["prompt_tokens"]), int(usage["completion_tokens"]), False
        return (
            text,
            approximate_tokens(system) + approximate_tokens(prompt),
            approximate_tokens(text),
            True,
        )


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
