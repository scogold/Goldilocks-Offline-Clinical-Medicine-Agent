"""Build the local retrieval index from manifest-approved PDF documents."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import ollama
from pypdf import PdfReader
from pypdf.errors import PdfReadError

import config as cfg

REQUIRED_MANIFEST_COLUMNS = {"filename", "title", "approved"}
TRUE_VALUES = {"1", "true", "yes", "y", "si", "sí"}


class IngestionError(RuntimeError):
    """Raised when an index cannot be built safely."""


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise IngestionError(f"Expected a 2-D embedding array, got shape {vectors.shape}.")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise IngestionError("The embedding model returned a zero-length vector.")
    return vectors / norms


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split normalized page text while keeping chunks on the same cited page."""
    text = " ".join(text.split())
    if not text:
        return []
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller.")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        next_start = max(end - overlap, start + 1)
        word_boundary = text.rfind(" ", start + 1, next_start + 1)
        if word_boundary >= 0:
            next_start = word_boundary + 1
        start = next_start
    return chunks


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_approved_documents(manifest_path: Path, document_dir: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise IngestionError(f"Manifest not found: {manifest_path}")

    with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_MANIFEST_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise IngestionError(f"Manifest is missing columns: {', '.join(sorted(missing))}")
        rows = [
            {key: (value or "").strip() for key, value in row.items()}
            for row in reader
            if (row.get("approved") or "").strip().lower() in TRUE_VALUES
        ]

    if not rows:
        raise IngestionError("The manifest contains no approved documents.")

    seen: set[str] = set()
    for row in rows:
        filename = row["filename"]
        if not filename or Path(filename).name != filename:
            raise IngestionError(f"Unsafe or invalid manifest filename: {filename!r}")
        if filename in seen:
            raise IngestionError(f"Duplicate approved filename in manifest: {filename}")
        seen.add(filename)

        path = document_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise IngestionError(f"Approved document is missing or empty: {path}")
        expected_hash = row.get("sha256", "").lower()
        if expected_hash and sha256_file(path) != expected_hash:
            raise IngestionError(f"SHA-256 mismatch for approved document: {filename}")
    return rows


def extract_records(documents: Iterable[dict[str, str]], document_dir: Path) -> list[dict]:
    records: list[dict] = []
    for document in documents:
        pdf_path = document_dir / document["filename"]
        try:
            reader = PdfReader(pdf_path)
            if reader.is_encrypted:
                raise IngestionError(f"Encrypted PDFs are not supported: {pdf_path.name}")
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = page.extract_text() or ""
                for chunk_number, text in enumerate(
                    chunk_text(page_text, cfg.CHUNK_SIZE, cfg.CHUNK_OVERLAP), start=1
                ):
                    records.append(
                        {
                            "id": f"{pdf_path.name}:p{page_number}:c{chunk_number}",
                            "filename": pdf_path.name,
                            "title": document["title"],
                            "source": document.get("source", ""),
                            "version": document.get("version", ""),
                            "language": document.get("language", ""),
                            "page": page_number,
                            "chunk": chunk_number,
                            "text": text,
                        }
                    )
        except (PdfReadError, OSError) as exc:
            raise IngestionError(f"Could not read {pdf_path.name}: {exc}") from exc

    if not records:
        raise IngestionError("No extractable text was found in the approved documents.")
    return records


def embed_records(records: list[dict], model: str, batch_size: int) -> np.ndarray:
    batches: list[np.ndarray] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        inputs = [f"title: {item['title']} | text: {item['text']}" for item in batch]
        try:
            response = ollama.embed(model=model, input=inputs)
        except Exception as exc:
            raise IngestionError(
                f"Ollama embedding failed. Is Ollama running and is '{model}' installed? {exc}"
            ) from exc
        vectors = np.asarray(response.embeddings, dtype=np.float32)
        if len(vectors) != len(batch):
            raise IngestionError("Ollama returned a different number of embeddings than inputs.")
        batches.append(vectors)
        done = min(start + batch_size, len(records))
        print(f"Embedded {done}/{len(records)} passages.", flush=True)
    return normalize_rows(np.vstack(batches).astype(np.float32, copy=False))


def atomic_write_json(path: Path, value: object) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, path)


def atomic_write_npy(path: Path, value: np.ndarray) -> None:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("wb") as handle:
        np.save(handle, value)
    os.replace(temp_path, path)


def build_index(manifest_path: Path, document_dir: Path, data_dir: Path) -> tuple[int, int]:
    documents = load_approved_documents(manifest_path, document_dir)
    records = extract_records(documents, document_dir)
    embeddings = embed_records(records, cfg.EMBEDDING_MODEL_NAME, cfg.EMBED_BATCH_SIZE)

    data_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_npy(data_dir / "embeddings.npy", embeddings)
    atomic_write_json(data_dir / "chunks.json", records)
    atomic_write_json(
        data_dir / "index_meta.json",
        {
            "embedding_model": cfg.EMBEDDING_MODEL_NAME,
            "embedding_dimensions": int(embeddings.shape[1]),
            "document_count": len(documents),
            "passage_count": len(records),
            "chunk_size": cfg.CHUNK_SIZE,
            "chunk_overlap": cfg.CHUNK_OVERLAP,
        },
    )
    return len(documents), len(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=cfg.MANIFEST_PATH)
    parser.add_argument("--documents", type=Path, default=cfg.DOCUMENT_DIR)
    parser.add_argument("--data", type=Path, default=cfg.DATA_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        document_count, passage_count = build_index(args.manifest, args.documents, args.data)
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}")
        return 1
    print(f"Indexed {passage_count} passages from {document_count} approved documents.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
