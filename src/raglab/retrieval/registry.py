"""Retriever registry. A framework adapter registers itself the same way."""

from __future__ import annotations

RETRIEVERS: dict[str, type] = {}


def register_retriever(cls: type) -> type:
    name = getattr(cls, "name", None)
    if not name:
        raise ValueError(f"{cls.__name__} needs a name")
    RETRIEVERS[name] = cls
    return cls


def build_retriever(name: str) -> object:
    if name not in RETRIEVERS:
        known = ", ".join(sorted(RETRIEVERS)) or "(none)"
        raise KeyError(f"Unknown retriever '{name}'. Known: {known}.")
    return RETRIEVERS[name]()
