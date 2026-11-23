"""Turn a manifest of PDFs, Markdown, HTML, or text into page blocks.

The files themselves are cached under ``cache_dir`` and are not committed.
Each block keeps the source id and the 1-based page it came from.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path

import yaml

# A heading is a short title, not a table cell or a one-word column label.
_HEADINGISH = re.compile(r"^[A-ZÁÉÍÓÚÂÊÔÃÕÇ][A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9 \-:.]{8,79}$")


class IngestError(RuntimeError):
    pass


@dataclass(frozen=True)
class Block:
    source_id: str
    page: int
    text: str
    title: str = ""


@dataclass(frozen=True)
class SourceSpec:
    source_id: str
    title: str
    url: str
    filename: str
    role: str
    download: bool
    sha256: str
    language: str


def load_manifest(path: Path) -> list[SourceSpec]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = payload.get("sources") or payload.get("documents") or []
    specs = []
    for row in rows:
        specs.append(
            SourceSpec(
                source_id=str(row.get("id") or row.get("source_id")),
                title=str(row.get("title") or ""),
                url=str(row.get("url") or ""),
                filename=str(row.get("filename") or f"{row.get('id')}.bin"),
                role=str(row.get("role") or "core"),
                download=bool(row.get("download", True)),
                sha256=str(row.get("sha256") or ""),
                language=str(row.get("language") or ""),
            )
        )
    return specs


def select_sources(specs: list[SourceSpec], roles: list[str] | None) -> list[SourceSpec]:
    chosen = []
    for spec in specs:
        if roles and spec.role not in roles:
            continue
        if not spec.download:
            continue
        chosen.append(spec)
    return chosen


def fetch_bytes(url: str, dest: Path, expected_sha256: str = "") -> bytes:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        cached = dest.read_bytes()
        if _checksum_ok(cached, expected_sha256):
            return cached
        dest.unlink()
    request = urllib.request.Request(url, headers={"User-Agent": "raglab/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response:
        data = response.read()
    if not _checksum_ok(data, expected_sha256):
        digest = hashlib.sha256(data).hexdigest()
        raise IngestError(f"Checksum mismatch for {dest.name}: {digest} != {expected_sha256}")
    dest.write_bytes(data)
    return data


def _checksum_ok(data: bytes, expected_sha256: str) -> bool:
    if not expected_sha256:
        return True
    return hashlib.sha256(data).hexdigest() == expected_sha256


def blocks_from_path(path: Path, source_id: str, title: str = "") -> list[Block]:
    suffix = path.suffix.lower()
    data = path.read_bytes()
    if suffix == ".pdf":
        return blocks_from_pdf(data, source_id, title)
    if suffix in {".html", ".htm"}:
        text = html_to_markdown(data.decode("utf-8", errors="replace"))
        return [Block(source_id, 1, text, title)]
    if suffix in {".md", ".txt", ""}:
        text = data.decode("utf-8", errors="replace")
        if suffix == ".txt":
            text = plain_to_markdown(text)
        return [Block(source_id, 1, text, title)]
    raise IngestError(f"Unsupported file type {suffix} for {path.name}")


def blocks_from_pdf(data: bytes, source_id: str, title: str = "") -> list[Block]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise IngestError("Reading PDFs needs the optional pypdf package") from exc
    import io

    reader = PdfReader(io.BytesIO(data))
    blocks = []
    for index, page in enumerate(reader.pages, start=1):
        raw = page.extract_text() or ""
        blocks.append(Block(source_id, index, plain_to_markdown(raw), title if index == 1 else ""))
    return blocks


def _is_heading(line: str) -> bool:
    """True for a short title line. Table rows, numbers, and one-word labels are not titles."""
    if not _HEADINGISH.match(line):
        return False
    if "  " in line or any(mark in line for mark in "|•·∙"):
        return False
    words = line.split()
    if not 2 <= len(words) <= 8:
        return False
    letters = sum(character.isalpha() for character in line)
    digits = sum(character.isdigit() for character in line)
    if letters < 6 or (digits and digits * 2 >= letters):
        return False
    return True


def plain_to_markdown(text: str) -> str:
    marked: list[tuple[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            marked.append(("blank", ""))
        elif _is_heading(line):
            marked.append(("heading", line))
        else:
            marked.append(("text", line))
    merged: list[tuple[str, str]] = []
    for kind, line in marked:
        if kind == "heading" and merged and merged[-1][0] == "heading":
            merged[-1] = ("heading", f"{merged[-1][1]} {line}")
        else:
            merged.append((kind, line))
    lines = []
    for kind, line in merged:
        if kind == "blank":
            lines.append("")
        elif kind == "heading":
            lines.append(f"## {line}")
        else:
            lines.append(line)
    return "\n".join(lines).strip()


class _HTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self._skip += 1
        if self._skip:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            level = int(tag[1])
            self.parts.append("\n" + ("#" * level) + " ")
        elif tag in {"p", "br", "li", "div"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if self._skip:
            return
        text = data.strip()
        if text:
            self.parts.append(text + " ")


def html_to_markdown(html: str) -> str:
    parser = _HTMLToMarkdown()
    parser.feed(html)
    return re.sub(r"\n{3,}", "\n\n", "".join(parser.parts)).strip()
