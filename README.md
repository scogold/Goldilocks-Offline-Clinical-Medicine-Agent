# Goldilocks — Offline Clinical Protocol Assistant

Goldilocks helps authorized clinic personnel find and understand information in an approved local medical-protocol library. It accepts questions in Spanish or English, retrieves relevant PDF passages, and uses local Ollama models to produce a concise Spanish response with document and page citations.

Goldilocks is decision support, not an autonomous medical provider. It does not diagnose patients, prescribe treatment, calculate doses, interpret imaging or laboratory results, or determine patient disposition. Clinical decisions remain with qualified healthcare professionals.

When a user directly requests one of those patient-care decisions, Goldilocks does not silently stop. It displays a prominent scope warning and may summarize relevant passages from approved documents, with citations, without applying them to the patient. It never falls back to uncited model knowledge for clinical decisions.

## Project layout

```text
.
├── Start_Goldilocks.bat      # Windows one-click launcher (double-click to run)
├── launch.py                 # Launcher logic: setup, checks, ingest, start app
├── app.py                    # Local Streamlit web interface
├── cli.py                    # Interactive and single-question terminal interface
├── config.py                 # Models, paths, and retrieval settings
├── ingest.py                 # Manifest validation, PDF extraction, embeddings
├── prompts.py                # Grounding and clinical safety boundaries
├── rag.py                    # Retrieval and local answer generation
├── requirements.txt
├── documents/
│   ├── manifest.csv          # Explicit approval list and document hashes
│   ├── approved/             # Only approved source PDFs
│   ├── pending_review/       # Downloaded sources awaiting review or OCR
│   └── quarantine/           # Invalid or unapproved files
├── data/                     # Generated local index; ignored by Git
└── evaluation/
    └── test_cases.csv
```

## Quick start (Windows, no command line)

Double-click **`Start_Goldilocks.bat`**. It sets up an isolated Python environment, installs dependencies, checks that Ollama is running and pulls the required models if missing, rebuilds the local index only when the approved documents changed, and opens the app in your browser. Python 3.10+ and Ollama still need to be installed first (see Setup below); the first run also needs internet access to download dependencies and models. Close the console window it opens to stop the app.

## Setup

Python 3.10+ and Ollama must be installed. The configured local models are `embeddinggemma` for retrieval and `gemma4:12b` for answers.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull embeddinggemma
ollama pull gemma4:12b
```

After the models have been downloaded, ingestion and use are fully local and do not require internet connectivity.

## Approve and index documents

Place PDFs in `documents/approved/` and add one row per file to `documents/manifest.csv`. An approved row should include a SHA-256 digest so unexpected file changes stop ingestion.

```bash
shasum -a 256 documents/approved/example.pdf
python ingest.py
```

The indexer refuses missing, empty, duplicate, modified, or unreadable approved PDFs. It writes `data/chunks.json`, `data/embeddings.npy`, and `data/index_meta.json` only after extraction and embedding succeed.

### Current Guatemala-focused sources

Active for retrieval:

- WHO/ICRC/MSF Spanish Interagency Integrated Triage Tool for adults, children, and high-risk reference criteria. This requires local workflow validation before operational triage use.
- OPS/OMS Guatemala and MSPAS 2024 field reference cards for primary-care outreach equipment.
- Current MSPAS Guatemala Module 1 for pregnancy, delivery, postpartum care, obstetric triage, danger signs, stabilization, and referral. Local referral routes and emergency workflows still require clinic validation.
- MSF January 2026 Essential Drugs guides in Spanish and English, including professional dosing, formulation, contraindication, precaution, and adverse-effect references. Check MSF updates and validate against the Guatemala formulary and clinic policy before medication use.

Stored in `documents/pending_review/` and excluded from answers:

- MSPAS 2018 comprehensive first- and second-level care standards, because a newer 2025 modular edition appears to supersede it.
- MSPAS 2023 under-five growth monitoring manual, because it is image-heavy and requires Spanish OCR before reliable retrieval.

## Use the web interface

Start the private local web app:

```bash
streamlit run app.py
```

Then open `http://127.0.0.1:8501` in a browser. Use the **Idioma / Language** selector to switch the entire interface between Spanish and English. New answers follow the selected language, regardless of the language used in the question. The interface also includes conversation history, expandable source passages, retrieval controls, index status, and clinical-scope notices. Streamlit telemetry is disabled and the server binds only to localhost.

## Ask questions from the terminal

Interactive mode:

```bash
python cli.py
```

One question:

```bash
python cli.py --question "¿Qué dice el protocolo sobre los antibióticos de reserva?"
```

Inspect retrieved evidence during development:

```bash
python cli.py --question "What is the AWaRe classification?" --show-context
```

Choose the terminal answer language with `--language es` or `--language en`:

```bash
python cli.py --question "¿Qué significa AWaRe?" --language en
```

## Updating the library

Treat `manifest.csv` as the retrieval-approval boundary. `approved=true` means a document may be indexed; it does not replace local clinical validation. Review the source, version, language, licensing, clinical applicability, and `clinical_review` status before operational use. Keep superseded, image-only, or unreviewed documents in `documents/pending_review/`. Recalculate the hash and rerun `python ingest.py` whenever an approved PDF changes. Generated index files are intentionally excluded from version control because they can be rebuilt locally.

## Privacy and limitations

- Ollama requests are sent only to the local Ollama service.
- Do not enter patient identifiers unless local clinic policy explicitly permits it.
- PDF text extraction does not perform OCR; scanned documents need an offline OCR step before indexing.
- Retrieval scores indicate semantic similarity, not clinical correctness.
- Always verify important details in the cited source page.
