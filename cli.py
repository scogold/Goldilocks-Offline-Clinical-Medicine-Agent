"""Command-line interface for the offline Goldilocks clinical assistant."""

from __future__ import annotations

import argparse

import config as cfg
from rag import GoldilocksRAG, RagError


DISCLAIMER = (
    "Goldilocks consulta protocolos aprobados; no diagnostica, prescribe, calcula dosis, "
    "interpreta pruebas ni decide la disposición de un paciente."
)


def display_answer(
    rag: GoldilocksRAG,
    question: str,
    top_k: int,
    show_context: bool,
    language: str,
) -> None:
    result = rag.answer(question, top_k=top_k, answer_language=language)
    print(f"\n{result.text}\n")
    if result.passages:
        print("Fuentes recuperadas:")
        for passage in result.passages:
            print(f"- {passage.citation} ({passage.filename}; similitud {passage.score:.3f})")
            if show_context:
                suffix = "…" if len(passage.text) > 500 else ""
                print(f"  {passage.text[:500]}{suffix}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-q", "--question", help="Ask one question and exit.")
    parser.add_argument("--top-k", type=int, default=cfg.DEFAULT_TOP_K)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument("--language", choices=("es", "en"), default="es")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print("Goldilocks — asistente local de protocolos clínicos")
    print(DISCLAIMER)
    try:
        rag = GoldilocksRAG()
        if args.question:
            display_answer(rag, args.question, args.top_k, args.show_context, args.language)
            return 0

        print("\nEscriba una pregunta en español o inglés. Use 'salir' para terminar.")
        while True:
            try:
                question = input("\nPregunta> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if question.lower() in {"salir", "exit", "quit"}:
                return 0
            if not question:
                continue
            display_answer(rag, question, args.top_k, args.show_context, args.language)
    except RagError as exc:
        print(f"\nError: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
