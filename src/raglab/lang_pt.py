"""Portuguese tokens for BM25.

Stopwords are removed before stemming, so a function word is never reduced
to a content stem and then kept. Other languages fall back to the default
ASCII tokenizer.
"""

from __future__ import annotations

import re

from raglab.textutil import tokenize

_LETTERS = re.compile(r"[^\W\d_]+", re.UNICODE)

# Function words. Written here so the search does not download a word list.
STOPWORDS = frozenset(
    {
        "a",
        "ao",
        "aos",
        "as",
        "ate",
        "até",
        "com",
        "como",
        "da",
        "das",
        "de",
        "dela",
        "dele",
        "do",
        "dos",
        "e",
        "ela",
        "elas",
        "ele",
        "eles",
        "em",
        "entre",
        "era",
        "essa",
        "esse",
        "esta",
        "este",
        "eu",
        "foi",
        "ha",
        "há",
        "isso",
        "isto",
        "ja",
        "já",
        "lhe",
        "mais",
        "mas",
        "me",
        "mesmo",
        "meu",
        "minha",
        "muito",
        "na",
        "nao",
        "não",
        "nas",
        "nem",
        "no",
        "nos",
        "nós",
        "o",
        "os",
        "ou",
        "para",
        "pela",
        "pelas",
        "pelo",
        "pelos",
        "por",
        "qual",
        "quando",
        "que",
        "se",
        "sem",
        "seu",
        "sua",
        "são",
        "sao",
        "também",
        "tambem",
        "te",
        "tem",
        "um",
        "uma",
        "você",
        "voce",
    }
)

# Longest suffix first. Applied only after stopwords are gone.
_SUFFIXES = (
    ("mente", ""),
    ("ações", "ação"),
    ("ões", "ão"),
    ("ães", "ão"),
    ("inhos", ""),
    ("inhas", ""),
    ("inho", ""),
    ("inha", ""),
    ("adoras", "ador"),
    ("adores", "ador"),
    ("s", ""),
)


def stem_pt(word: str) -> str:
    if len(word) <= 3:
        return word
    for suffix, replacement in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


def tokenize_pt(text: str) -> list[str]:
    tokens = [match.group(0).lower() for match in _LETTERS.finditer(text)]
    kept = [token for token in tokens if token not in STOPWORDS]
    return [stem_pt(token) for token in kept]


def tokenize_for_language(text: str, language: str) -> list[str]:
    if language.lower() in {"pt", "pt-br", "portuguese"}:
        return tokenize_pt(text)
    return tokenize(text)
