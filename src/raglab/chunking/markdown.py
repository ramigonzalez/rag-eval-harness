"""Split on Markdown headings. A document with no headings stays one chunk."""

from __future__ import annotations

import re

from raglab.chunking.base import Chunker
from raglab.chunking.registry import register_chunker

_HEADING = re.compile(r"(?m)^#{1,6}[ \t]+\S.*$")


@register_chunker
class MarkdownChunker(Chunker):
    name = "markdown"

    @classmethod
    def from_config(cls, options: dict) -> MarkdownChunker:
        if options:
            raise ValueError(f"markdown chunker takes no options, got {sorted(options)}")
        return cls()

    def split_text(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        matches = list(_HEADING.finditer(text))
        if not matches:
            return [text]
        parts: list[str] = []
        if matches[0].start() > 0:
            preamble = text[: matches[0].start()].strip()
            if preamble:
                parts.append(preamble)
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            section = text[match.start() : end].strip()
            if section:
                parts.append(section)
        return parts
