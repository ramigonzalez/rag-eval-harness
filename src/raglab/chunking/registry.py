"""Chunker registry. Importing a chunker module registers it."""

from __future__ import annotations

from typing import Any

CHUNKERS: dict[str, type] = {}


def register_chunker(cls: type) -> type:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} needs a name")
    CHUNKERS[name] = cls
    return cls


def build_chunker(name: str, options: dict[str, Any] | None = None) -> Any:
    if name not in CHUNKERS:
        known = ", ".join(sorted(CHUNKERS)) or "(none)"
        raise KeyError(f"Unknown chunker '{name}'. Known: {known}.")
    return CHUNKERS[name].from_config(options or {})
