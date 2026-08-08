import json
from pathlib import Path

import numpy as np
import ollama
from pypdf import PdfReader
import config as cfg


DOCUMENT_DIR = Path("documents/approved")
DATA_DIR = Path("data")
EMBEDDING_MODEL = cfg.EMBEDDING_MODEL_NAME


def normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def chunk_text(text, chunk_size=1600, overlap=250):
    text = " ".join(text.split())
    chunks = []

    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])

        if end == len(text):
            break

        start = end - overlap

    return chunks


records = []

for pdf_path in DOCUMENT_DIR.glob("MedicalLibrary/*.pdf"):
    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        for chunk_number, text in enumerate(chunk_text(page_text), start=1):
            records.append({
                "filename": pdf_path.name,
                "title": pdf_path.stem,
                "page": page_number,
                "chunk": chunk_number,
                "text": text,
            })

embedding_inputs = [
    f"title: {record['title']} | text: {record['text']}"
    for record in records
]

response = ollama.embed(
    model=EMBEDDING_MODEL,
    input=embedding_inputs,
)

embeddings = np.asarray(response.embeddings, dtype=np.float32)
embeddings = np.asarray([normalize(vector) for vector in embeddings])

DATA_DIR.mkdir(exist_ok=True)

np.save(DATA_DIR / "embeddings.npy", embeddings)

with open(DATA_DIR / "chunks.json", "w", encoding="utf-8") as file:
    json.dump(records, file, ensure_ascii=False, indent=2)

print(f"Indexed {len(records)} passages.")