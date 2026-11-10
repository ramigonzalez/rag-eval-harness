"""Dataset loading.

SciFact is the public BEIR retrieval set used for the published run: scientific
claims against paper abstracts, with binary evidence labels. The smoke fixture
is original text checked into the repo so CI does not download anything except
the local embedding model.
"""

from __future__ import annotations

import hashlib
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

from raglab.types import Dataset, Document, Query

logger = logging.getLogger("raglab")

SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SHA256 = "536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165"


def load_dataset(name: str, data_dir: Path, fixture_path: Path | None) -> Dataset:
    if name == "scifact":
        return load_scifact(data_dir)
    if name == "fixture":
        if fixture_path is None:
            raise ValueError("fixture dataset needs fixture_path")
        return load_fixture(fixture_path)
    raise ValueError(f"Unknown dataset '{name}'. Use 'scifact' or 'fixture'.")


def load_fixture(path: Path) -> Dataset:
    payload = json.loads(path.read_text(encoding="utf-8"))
    documents = [
        Document(
            doc_id=str(row["doc_id"]),
            title=str(row.get("title", "")),
            text=str(row.get("text", "")),
        )
        for row in payload["documents"]
    ]
    queries = [
        Query(query_id=str(row["query_id"]), text=str(row["text"])) for row in payload["queries"]
    ]
    qrels: dict[str, dict[str, int]] = {}
    for row in payload["qrels"]:
        qrels.setdefault(str(row["query_id"]), {})[str(row["doc_id"])] = int(row["score"])
    return Dataset(
        name=str(payload.get("name", path.stem)),
        documents=documents,
        queries=queries,
        qrels=qrels,
    )


def load_scifact(data_dir: Path) -> Dataset:
    root = data_dir / "beir" / "scifact"
    corpus_path = root / "corpus.jsonl"
    if not corpus_path.exists():
        _download_scifact(data_dir / "beir")
    if not corpus_path.exists():
        raise RuntimeError(f"SciFact corpus is missing at {corpus_path}")
    documents = []
    for line in corpus_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        documents.append(
            Document(
                doc_id=str(row["_id"]),
                title=str(row.get("title") or ""),
                text=str(row.get("text") or ""),
            )
        )
    queries = []
    for line in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        text = str(row.get("text") or "").strip()
        if text:
            queries.append(Query(query_id=str(row["_id"]), text=text))
    qrels = _load_qrels(root / "qrels" / "test.tsv")
    logger.info(
        "Loaded SciFact: %s documents, %s queries in the file, %s labeled test queries",
        f"{len(documents):,}",
        f"{len(queries):,}",
        f"{len(qrels):,}",
    )
    return Dataset(name="scifact", documents=documents, queries=queries, qrels=qrels)


def _load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        if line_no == 0 and not parts[-1].lstrip("-").isdigit():
            continue
        query_id, doc_id, score = parts[0], parts[1], int(parts[2])
        qrels.setdefault(query_id, {})[doc_id] = score
    return qrels


def _download_scifact(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "scifact.zip"
    logger.info("Downloading SciFact (~8 MB) from %s", SCIFACT_URL)
    digest = _download(SCIFACT_URL, zip_path)
    if digest != SCIFACT_SHA256:
        zip_path.unlink(missing_ok=True)
        raise RuntimeError(
            "SciFact checksum mismatch. "
            f"Got {digest}, expected {SCIFACT_SHA256}. "
            "Refusing to evaluate a file that is not the pinned BEIR release."
        )
    _safe_extract(zip_path, dest)
    logger.info("SciFact extracted to %s", dest / "scifact")


def _download(url: str, dest: Path) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "raglab/0.1 (benchmark)"})
    hasher = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                hasher.update(block)
                handle.write(block)
    except Exception:
        dest.unlink(missing_ok=True)
        raise
    return hasher.hexdigest()


def _safe_extract(zip_path: Path, dest: Path) -> None:
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            if name.is_absolute() or ".." in name.parts:
                raise RuntimeError(f"Unsafe path in zip: {info.filename}")
            target = dest / name
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as handle:
                handle.write(source.read())
