"""Retrieval and grounded answer generation for Goldilocks."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ollama

import config as cfg
from prompts import (
    NO_EVIDENCE_MESSAGES,
    RESTRICTED_REQUEST_NOTICES,
    SYSTEM_PROMPT,
    build_user_prompt,
    is_patient_specific_dose_request,
    is_restricted_request,
    normalize_language,
)


class RagError(RuntimeError):
    """Raised when retrieval or local generation cannot proceed."""


@dataclass(frozen=True)
class Passage:
    filename: str
    title: str
    page: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        return f"[{self.title}, p. {self.page}]"


@dataclass(frozen=True)
class Answer:
    text: str
    passages: list[Passage]


class GoldilocksRAG:
    def __init__(self, data_dir: Path = cfg.DATA_DIR) -> None:
        chunks_path = data_dir / "chunks.json"
        embeddings_path = data_dir / "embeddings.npy"
        if not chunks_path.exists() or not embeddings_path.exists():
            raise RagError("Index files are missing. Run `python ingest.py` first.")

        try:
            with chunks_path.open(encoding="utf-8") as handle:
                self.records = json.load(handle)
            self.embeddings = np.load(embeddings_path, allow_pickle=False)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise RagError(f"The local index could not be loaded: {exc}") from exc

        if not self.records:
            raise RagError("The index is empty. Correct the document setup and run ingestion again.")
        if self.embeddings.ndim != 2 or len(self.embeddings) != len(self.records):
            raise RagError("Index mismatch: chunks and embeddings do not have compatible shapes.")

    def retrieve(self, question: str, top_k: int = cfg.DEFAULT_TOP_K) -> list[Passage]:
        question = question.strip()
        if not question:
            raise RagError("The question cannot be empty.")
        if top_k < 1:
            raise RagError("top_k must be at least 1.")

        try:
            response = ollama.embed(
                model=cfg.EMBEDDING_MODEL_NAME,
                input=[f"query: {question}"],
            )
        except Exception as exc:
            raise RagError(
                f"Could not query Ollama with '{cfg.EMBEDDING_MODEL_NAME}': {exc}"
            ) from exc

        query = np.asarray(response.embeddings[0], dtype=np.float32)
        norm = np.linalg.norm(query)
        if norm == 0 or query.shape[0] != self.embeddings.shape[1]:
            raise RagError("The query embedding is empty or incompatible with the index.")
        scores = self.embeddings @ (query / norm)
        candidate_indices = np.argsort(scores)[::-1][: min(top_k, len(scores))]

        passages: list[Passage] = []
        for index in candidate_indices:
            score = float(scores[index])
            if score < cfg.MIN_SIMILARITY:
                continue
            record = self.records[int(index)]
            passages.append(
                Passage(
                    filename=record["filename"],
                    title=record["title"],
                    page=int(record["page"]),
                    text=record["text"],
                    score=score,
                )
            )
        return passages

    @staticmethod
    def format_context(
        passages: list[Passage], redact_numeric_values: bool = False
    ) -> str:
        sections: list[str] = []
        used_chars = 0
        for number, passage in enumerate(passages, start=1):
            passage_text = passage.text
            if redact_numeric_values:
                passage_text = re.sub(
                    r"\d+(?:[.,]\d+)?",
                    "[valor numérico omitido en modo restringido]",
                    passage_text,
                )
            section = f"FRAGMENTO {number} {passage.citation}\n{passage_text}"
            remaining = cfg.MAX_CONTEXT_CHARS - used_chars
            if remaining <= 0:
                break
            section = section[:remaining]
            sections.append(section)
            used_chars += len(section)
        return "\n\n".join(sections)

    def answer(
        self,
        question: str,
        top_k: int = cfg.DEFAULT_TOP_K,
        answer_language: str = "es",
    ) -> Answer:
        try:
            answer_language = normalize_language(answer_language)
        except ValueError as exc:
            raise RagError(str(exc)) from exc
        restricted = is_restricted_request(question)
        passages = self.retrieve(question, top_k=top_k)
        if not passages:
            text = NO_EVIDENCE_MESSAGES[answer_language]
            if restricted:
                text = f"{RESTRICTED_REQUEST_NOTICES[answer_language]}\n\n{text}"
            return Answer(text, [])

        context = self.format_context(
            passages,
            redact_numeric_values=is_patient_specific_dose_request(question),
        )
        try:
            response = ollama.chat(
                model=cfg.CHAT_MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": build_user_prompt(
                            question,
                            context,
                            restricted=restricted,
                            answer_language=answer_language,
                        ),
                    },
                ],
                think=False,
                options={"temperature": 0.1, "num_ctx": 8_192, "num_predict": 500},
            )
        except Exception as exc:
            raise RagError(f"Could not generate an answer with '{cfg.CHAT_MODEL_NAME}': {exc}") from exc

        text = response.message.content.strip()
        if not text:
            raise RagError("The chat model returned an empty answer.")
        if restricted:
            text = f"{RESTRICTED_REQUEST_NOTICES[answer_language]}\n\n{text}"
        return Answer(text, passages)
