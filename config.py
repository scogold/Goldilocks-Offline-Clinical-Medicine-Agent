"""Central configuration for the local Goldilocks application."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DOCUMENT_DIR = BASE_DIR / "documents" / "approved"
MANIFEST_PATH = BASE_DIR / "documents" / "manifest.csv"
DATA_DIR = BASE_DIR / "data"

# Streamlit serves this folder at /app/static/ when enableStaticServing is on,
# letting the UI link straight to an approved PDF instead of embedding a copy.
STATIC_DIR = BASE_DIR / "static"
STATIC_DOCUMENT_DIR = STATIC_DIR / "documents"
STATIC_DOCUMENT_URL_PREFIX = "app/static/documents"

CHAT_MODEL_NAME = "gemma4:12b"
EMBEDDING_MODEL_NAME = "embeddinggemma"

CHUNK_SIZE = 1_400
CHUNK_OVERLAP = 200
EMBED_BATCH_SIZE = 32
DEFAULT_TOP_K = 5
MIN_SIMILARITY = 0.15
MAX_CONTEXT_CHARS = 10_000
