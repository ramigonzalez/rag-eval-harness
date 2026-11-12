"""Match retrieved context text back to corpus documents.

Chunkers do not all preserve the original whitespace. Fixed windows rejoin
words with single spaces, and a recursive split can drop the space after a
sentence boundary. Matching on alphanumeric tokens still finds the source
document, which is what a gold-label check needs.
"""

from __future__ import annotations

import re
from collections import defaultdict

_TOKEN = re.compile(r"[A-Za-z0-9]+")


def tokens(text: str) -> tuple[str, ...]:
    return tuple(match.group(0).lower() for match in _TOKEN.finditer(text))


class DocTokenIndex:
    """Lookup from a context string to the documents that contain it."""

    def __init__(self, documents: list[tuple[str, str]]) -> None:
        self._tokens: dict[str, tuple[str, ...]] = {}
        self._index: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for doc_id, text in documents:
            doc_tokens = tokens(text)
            self._tokens[doc_id] = doc_tokens
            if len(doc_tokens) < 5:
                self._index[doc_tokens].append(doc_id)
                continue
            seen: set[tuple[str, ...]] = set()
            for start in range(len(doc_tokens) - 4):
                key = doc_tokens[start : start + 5]
                if key in seen:
                    continue
                seen.add(key)
                self._index[key].append(doc_id)

    def match(self, context: str) -> list[str]:
        wanted = tokens(context)
        if not wanted:
            return []
        if len(wanted) < 5:
            return [
                doc_id
                for doc_id, doc_tokens in self._tokens.items()
                if _contains(doc_tokens, wanted)
            ]
        key = wanted[:5]
        hits: list[str] = []
        seen: set[str] = set()
        for doc_id in self._index.get(key, []):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if _contains(self._tokens[doc_id], wanted):
                hits.append(doc_id)
        return hits


def contexts_include_gold(
    contexts: list[str],
    gold_ids: set[str],
    index: DocTokenIndex,
) -> bool:
    for context in contexts:
        if any(doc_id in gold_ids for doc_id in index.match(context)):
            return True
    return False


def _contains(document: tuple[str, ...], chunk: tuple[str, ...]) -> bool:
    if not chunk or len(chunk) > len(document):
        return False
    span = len(chunk)
    start = 0
    first = chunk[0]
    while True:
        try:
            found = document.index(first, start)
        except ValueError:
            return False
        if document[found : found + span] == chunk:
            return True
        start = found + 1
