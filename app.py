"""Local bilingual Streamlit web interface for Goldilocks."""

from __future__ import annotations

import json

import streamlit as st

import config as cfg
from rag import GoldilocksRAG, RagError


st.set_page_config(
    page_title="Goldilocks | Clinical Protocols",
    page_icon="🐻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --gold-ink: #173a34;
        --gold-green: #1f5d50;
        --gold-mint: #e7f1ed;
        --gold-cream: #fbf8f1;
        --gold-amber: #d99128;
    }
    .stApp { background: var(--gold-cream); color: var(--gold-ink); }
    [data-testid="stHeader"] { background: rgba(251, 248, 241, 0.88); }
    [data-testid="stSidebar"] { background: #153d35; }
    [data-testid="stSidebar"] * { color: #f7f3e8; }
    [data-testid="stSidebar"] hr { border-color: rgba(255,255,255,.18); }
    .hero {
        padding: 1.2rem 1.4rem 1rem;
        border: 1px solid #c9ddd5;
        border-radius: 20px;
        background: linear-gradient(135deg, #f8f3e7 0%, #e3f0ea 100%);
        box-shadow: 0 12px 30px rgba(23, 58, 52, .07);
        margin-bottom: 1rem;
    }
    .eyebrow { color: var(--gold-green); font-size: .76rem; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .hero h1 { color: var(--gold-ink); font-size: clamp(2rem, 5vw, 3.35rem); line-height: .96; margin: .4rem 0 .7rem; }
    .hero p { color: #41645d; max-width: 780px; font-size: 1.02rem; margin: 0; }
    .safety-note {
        border-left: 4px solid var(--gold-amber);
        background: #fff8e8;
        color: #5d451d;
        border-radius: 8px;
        padding: .8rem 1rem;
        margin: .7rem 0 1.1rem;
    }
    [data-testid="stChatMessage"] {
        background: rgba(255,255,255,.68);
        border: 1px solid rgba(31,93,80,.12);
        border-radius: 16px;
        padding: .45rem .75rem;
        margin-bottom: .65rem;
    }
    .source-chip {
        display: inline-block;
        color: #1f5d50;
        background: #e7f1ed;
        border-radius: 999px;
        padding: .16rem .55rem;
        font-size: .78rem;
        font-weight: 700;
        margin-right: .25rem;
    }
    .status-dot { color: #76d9ae; font-size: .85rem; font-weight: 700; }
    .stButton > button { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)


TEXT = {
    "es": {
        "library": "Biblioteca clínica local",
        "status": "Índice local disponible",
        "passages": "Pasajes indexados",
        "documents": "documentos aprobados",
        "index_error": "No se pudo leer el resumen del índice.",
        "models": "Modelos locales",
        "chat_model": "Chat",
        "search_model": "Búsqueda",
        "source_count": "Cantidad de fuentes",
        "source_help": "Número máximo de fragmentos enviados al modelo.",
        "clear": "Limpiar conversación",
        "local": "🔒 Se ejecuta en este equipo mediante Ollama.",
        "privacy": "No introduzca identificadores de pacientes salvo que la política local lo permita.",
        "eyebrow": "Asistente de protocolos · Español / English",
        "headline": "La respuesta correcta empieza<br>en la fuente correcta.",
        "description": "Consulte la biblioteca médica aprobada y reciba respuestas concisas en español, acompañadas por el documento y la página de origen.",
        "safety_title": "Apoyo informativo, no atención autónoma.",
        "safety": "Goldilocks no sustituye el juicio clínico. Las decisiones corresponden al profesional responsable.",
        "load_error": "No se pudo cargar la biblioteca",
        "hello": "Hola. Puedo buscar información en los protocolos aprobados. Pregúnteme en español o inglés; por ejemplo: **¿Qué significan las categorías AWaRe?**",
        "input": "Pregunte sobre los protocolos aprobados…",
        "spinner": "Buscando en los documentos aprobados…",
        "answer_error": "No pude completar la consulta",
        "view_sources": "Ver {count} fuentes recuperadas",
        "page": "p.",
        "similarity": "similitud",
    },
    "en": {
        "library": "Local clinical library",
        "status": "Local index available",
        "passages": "Indexed passages",
        "documents": "approved documents",
        "index_error": "The index summary could not be read.",
        "models": "Local models",
        "chat_model": "Chat",
        "search_model": "Search",
        "source_count": "Number of sources",
        "source_help": "Maximum number of excerpts sent to the model.",
        "clear": "Clear conversation",
        "local": "🔒 Runs on this computer through Ollama.",
        "privacy": "Do not enter patient identifiers unless local policy permits it.",
        "eyebrow": "Protocol assistant · English / Español",
        "headline": "The right answer starts<br>with the right source.",
        "description": "Search the approved medical library and receive concise English answers with the originating document and page.",
        "safety_title": "Informational support, not autonomous care.",
        "safety": "Goldilocks does not replace clinical judgment. Decisions remain with the responsible professional.",
        "load_error": "The library could not be loaded",
        "hello": "Hello. I can search the approved clinical protocols. Ask in English or Spanish; for example: **What do the AWaRe categories mean?**",
        "input": "Ask about the approved protocols…",
        "spinner": "Searching the approved documents…",
        "answer_error": "I could not complete the request",
        "view_sources": "View {count} retrieved sources",
        "page": "p.",
        "similarity": "similarity",
    },
}


@st.cache_resource(show_spinner=False)
def load_rag(index_fingerprint: tuple[int, int, int, int]) -> GoldilocksRAG:
    return GoldilocksRAG()


def current_index_fingerprint() -> tuple[int, int, int, int]:
    """Invalidate Streamlit's cached index whenever either generated file changes."""
    chunks = cfg.DATA_DIR / "chunks.json"
    embeddings = cfg.DATA_DIR / "embeddings.npy"
    return (
        chunks.stat().st_mtime_ns,
        chunks.stat().st_size,
        embeddings.stat().st_mtime_ns,
        embeddings.stat().st_size,
    )


def passage_to_dict(passage) -> dict:
    return {
        "filename": passage.filename,
        "title": passage.title,
        "page": passage.page,
        "text": passage.text,
        "score": passage.score,
    }


def render_sources(sources: list[dict], text: dict[str, str]) -> None:
    if not sources:
        return
    with st.expander(text["view_sources"].format(count=len(sources)), expanded=False):
        for source_index, source in enumerate(sources):
            st.markdown(
                f'<span class="source-chip">{text["page"]} {source["page"]}</span> '
                f'**{source["title"]}** · {text["similarity"]} `{source["score"]:.3f}`',
                unsafe_allow_html=True,
            )
            st.caption(source["filename"])
            st.write(source["text"])
            if source_index < len(sources) - 1:
                st.divider()


with st.sidebar:
    language_choice = st.selectbox(
        "Idioma / Language",
        options=("Español", "English"),
        key="interface_language",
    )
    language = "es" if language_choice == "Español" else "en"
    t = TEXT[language]

    st.markdown("## 🐻 Goldilocks")
    st.caption(t["library"])
    st.markdown(f'<div class="status-dot">● {t["status"]}</div>', unsafe_allow_html=True)

    try:
        with (cfg.DATA_DIR / "index_meta.json").open(encoding="utf-8") as handle:
            index_meta = json.load(handle)
        st.metric(t["passages"], f'{index_meta["passage_count"]:,}')
        st.caption(f'{index_meta["document_count"]} {t["documents"]}')
    except (OSError, KeyError, json.JSONDecodeError):
        st.warning(t["index_error"])

    st.divider()
    st.markdown(f'**{t["models"]}**')
    st.caption(f'{t["chat_model"]} · `{cfg.CHAT_MODEL_NAME}`')
    st.caption(f'{t["search_model"]} · `{cfg.EMBEDDING_MODEL_NAME}`')
    st.divider()

    top_k = st.slider(
        t["source_count"],
        min_value=2,
        max_value=8,
        value=cfg.DEFAULT_TOP_K,
        help=t["source_help"],
    )
    if st.button(t["clear"], use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption(t["local"])
    st.caption(t["privacy"])


st.markdown(
    f"""
    <section class="hero">
      <div class="eyebrow">{t["eyebrow"]}</div>
      <h1>{t["headline"]}</h1>
      <p>{t["description"]}</p>
    </section>
    <div class="safety-note"><strong>{t["safety_title"]}</strong> {t["safety"]}</div>
    """,
    unsafe_allow_html=True,
)

try:
    rag = load_rag(current_index_fingerprint())
except (OSError, RagError) as exc:
    st.error(f'{t["load_error"]}: {exc}')
    st.code("python ingest.py", language="bash")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    with st.chat_message("assistant", avatar="🐻"):
        st.markdown(t["hello"])

for message in st.session_state.messages:
    avatar = "🐻" if message["role"] == "assistant" else "🩺"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        render_sources(message.get("sources", []), t)

question = st.chat_input(t["input"])
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🩺"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🐻"):
        try:
            with st.spinner(t["spinner"]):
                result = rag.answer(question, top_k=top_k, answer_language=language)
            sources = [passage_to_dict(passage) for passage in result.passages]
            st.markdown(result.text)
            render_sources(sources, t)
            st.session_state.messages.append(
                {"role": "assistant", "content": result.text, "sources": sources}
            )
        except RagError as exc:
            error_text = f'{t["answer_error"]}: {exc}'
            st.error(error_text)
            st.session_state.messages.append({"role": "assistant", "content": error_text})
