"""One-click launcher: prepares the environment and starts the Goldilocks app.

Invoked by Start_Goldilocks.bat so a non-technical user can double-click one
file. Creates an isolated virtual environment, installs Python dependencies,
makes sure Ollama and its models are available, rebuilds the local index only
when the approved documents changed, then opens the app in the browser.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
import venv
import webbrowser
from pathlib import Path

import config as cfg

VENV_DIR = cfg.BASE_DIR / ".goldivenv"
APP_URL = "http://127.0.0.1:8501"
REQUIRED_MODELS = (cfg.EMBEDDING_MODEL_NAME, cfg.CHAT_MODEL_NAME)


def say(es: str, en: str) -> None:
    print(f"{es} / {en}", flush=True)


def venv_python() -> Path:
    if sys.platform == "win32":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def ensure_venv() -> Path:
    python = venv_python()
    if not python.exists():
        say(
            "Preparando el entorno de Python (solo la primera vez)...",
            "Setting up the Python environment (first run only)...",
        )
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    return python


def ensure_dependencies(python: Path) -> None:
    marker = VENV_DIR / ".requirements_installed"
    requirements = cfg.BASE_DIR / "requirements.txt"
    if marker.exists() and marker.stat().st_mtime >= requirements.stat().st_mtime:
        return
    say("Instalando dependencias de Python...", "Installing Python dependencies...")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "-r", str(requirements)],
        check=True,
    )
    marker.write_text("ok", encoding="utf-8")


def _ollama_responsive() -> bool:
    try:
        subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=5, check=True
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        return False


def ensure_ollama_running() -> None:
    if shutil.which("ollama") is None:
        say(
            "No se encontró Ollama. Instálelo desde https://ollama.com/download y vuelva a intentarlo.",
            "Ollama was not found. Install it from https://ollama.com/download and try again.",
        )
        raise SystemExit(1)

    if _ollama_responsive():
        return

    say("Iniciando Ollama...", "Starting Ollama...")
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline:
        if _ollama_responsive():
            return
        time.sleep(1)

    say(
        "No se pudo iniciar Ollama automáticamente. Ábralo manualmente y vuelva a intentarlo.",
        "Could not start Ollama automatically. Please open it manually and try again.",
    )
    raise SystemExit(1)


def installed_ollama_models() -> set[str]:
    result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    names: set[str] = set()
    for line in result.stdout.splitlines()[1:]:
        if line.strip():
            names.add(line.split()[0])
    return names


def ensure_models() -> None:
    installed = installed_ollama_models()
    for model in REQUIRED_MODELS:
        base_name = model.split(":")[0]
        if any(name == model or name.split(":")[0] == base_name for name in installed):
            continue
        say(
            f"Descargando el modelo {model} (puede tardar varios minutos)...",
            f"Downloading model {model} (this can take several minutes)...",
        )
        subprocess.run(["ollama", "pull", model], check=True)


def index_is_stale() -> bool:
    meta = cfg.DATA_DIR / "index_meta.json"
    chunks = cfg.DATA_DIR / "chunks.json"
    embeddings = cfg.DATA_DIR / "embeddings.npy"
    if not (meta.exists() and chunks.exists() and embeddings.exists()):
        return True

    index_time = meta.stat().st_mtime
    if cfg.MANIFEST_PATH.exists() and cfg.MANIFEST_PATH.stat().st_mtime > index_time:
        return True
    if cfg.DOCUMENT_DIR.is_dir():
        for pdf in cfg.DOCUMENT_DIR.glob("*.pdf"):
            if pdf.stat().st_mtime > index_time:
                return True
    return False


def ensure_index(python: Path) -> None:
    if not index_is_stale():
        return
    say(
        "Actualizando la biblioteca local (esto puede tardar unos minutos)...",
        "Updating the local library (this can take a few minutes)...",
    )
    subprocess.run([str(python), str(cfg.BASE_DIR / "ingest.py")], check=True, cwd=cfg.BASE_DIR)


def wait_and_open_browser(timeout: float = 45.0) -> None:
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(APP_URL, timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    webbrowser.open(APP_URL)


def launch_app(python: Path) -> None:
    say("Iniciando Goldilocks...", "Starting Goldilocks...")
    process = subprocess.Popen(
        [str(python), "-m", "streamlit", "run", str(cfg.BASE_DIR / "app.py")],
        cwd=cfg.BASE_DIR,
    )
    wait_and_open_browser()
    say(
        "Goldilocks se está ejecutando. Cierre esta ventana para detener la aplicación.",
        "Goldilocks is running. Close this window to stop the application.",
    )
    process.wait()


def main() -> int:
    try:
        python = ensure_venv()
        ensure_dependencies(python)
        ensure_ollama_running()
        ensure_models()
        ensure_index(python)
        launch_app(python)
    except SystemExit as exc:
        return int(exc.code or 0)
    except subprocess.CalledProcessError as exc:
        say(f"Ocurrió un error: {exc}", f"Something went wrong: {exc}")
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
