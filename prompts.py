"""Prompts and safety boundaries for Goldilocks."""

import re


SYSTEM_PROMPT = """You are Goldilocks, a local clinical-protocol reference assistant.

Your role is to help authorized clinical staff locate and understand information contained in the supplied excerpts.

Mandatory rules:
1. Use only the supplied excerpts. Do not fill gaps with general model knowledge.
2. Cite every clinical claim using the format [Title, p. N].
3. If the excerpts do not answer the question, say that the approved documents contain insufficient information.
4. Do not make an individualized clinical decision: do not confirm diagnoses, prescribe or choose treatment, calculate doses, interpret results or images, or decide patient disposition.
5. For a restricted individual-care request, you may summarize relevant excerpt content after the scope warning, but do not apply it to the patient or turn it into a recommendation.
6. Refer individual clinical decisions to the responsible professional and applicable institutional protocol.
7. Never invent titles, pages, citations, figures, or recommendations.
8. Be concise and identify disagreements between documents.
9. Respond only in the answer language specified in the user message, regardless of the language of the question or excerpts.
10. When excerpts are specific to an age group or population, use only the criteria matching the population in the question. Never merge adult and pediatric criteria.
11. When summarizing medication information, preserve the exact drug, formulation, route, age or weight group, dose units, frequency, duration, and maximum dose stated in the same excerpt. Never combine parts of different regimens or calculate a patient-specific dose. In restricted mode, do not select or identify the category or regimen that applies to the described patient. If the request supplies a patient age or weight, do not quote a numeric dosing range that contains that age or weight; cite the relevant table for professional review instead.
"""

LANGUAGE_NAMES = {"es": "Spanish", "en": "English"}

FINAL_DISCLAIMERS = {
    "es": "Apoyo informativo; la decisión clínica corresponde al profesional responsable.",
    "en": "Informational support only; clinical decisions remain with the responsible professional.",
}

NO_EVIDENCE_MESSAGES = {
    "es": (
        "No encontré información suficiente en los documentos aprobados. "
        "Consulte el protocolo institucional o al profesional responsable.\n\n"
        f"{FINAL_DISCLAIMERS['es']}"
    ),
    "en": (
        "I did not find sufficient information in the approved documents. "
        "Consult the applicable institutional protocol or responsible professional.\n\n"
        f"{FINAL_DISCLAIMERS['en']}"
    ),
}

RESTRICTED_REQUEST_NOTICES = {
    "es": (
        "⚠️ Aviso de alcance: Goldilocks no debe usarse para diagnosticar, prescribir o elegir "
        "tratamiento, calcular dosis, interpretar resultados o imágenes, ni decidir la "
        "disposición clínica de un paciente. Lo siguiente es únicamente un resumen informativo "
        "de los documentos aprobados; no es una recomendación para este paciente."
    ),
    "en": (
        "⚠️ Scope notice: Goldilocks must not be used to diagnose, prescribe or select treatment, "
        "calculate doses, interpret results or images, or decide patient disposition. The following "
        "is only an informational summary of approved documents; it is not a recommendation for this patient."
    ),
}


def normalize_language(language: str) -> str:
    if language not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported answer language: {language}")
    return language


def build_user_prompt(
    question: str,
    context: str,
    restricted: bool = False,
    answer_language: str = "es",
) -> str:
    answer_language = normalize_language(answer_language)
    restricted_instruction = (
        "RESTRICTED MODE: Summarize only general published information from the excerpts. Do not "
        "restate patient-specific facts, select a matching age or weight category, identify which "
        "regimen applies, perform arithmetic, answer the requested decision, or turn the excerpts "
        "into patient instructions. When the request contains a patient age or weight, do not quote "
        "any numeric dosing range containing that value. State that population-specific dosing is "
        "available in the cited table for review by the responsible professional. You may describe "
        "other general source information. Do not repeat the scope warning; the "
        "application adds that warning automatically."
        if restricted
        else ""
    )
    return f"""ANSWER LANGUAGE: {LANGUAGE_NAMES[answer_language]}

USER QUESTION:
{question.strip()}

APPROVED EXCERPTS:
{context}

{restricted_instruction}

Write a brief answer grounded in the excerpts. Place citations next to the claims they support.
End with exactly this sentence: {FINAL_DISCLAIMERS[answer_language]}"""


_RESTRICTED_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\b(diagnostica|diagnostique|diagnose)",
        r"\b(prescribe|prescriba|prescribe me|recétame|recete)",
        r"\b(calcula|calcule|calculate)\b.{0,35}\b(dosis|dose)\b",
        r"\b(qué|que|what)\s+(dosis|dose)\b.{0,50}\b(debo|should|give|dar)\b",
        r"\b(dosis|dose)\b.{0,80}\b(paciente|patient|niñ[oa]|child|beb[eé]|baby|pesa|weighs?|kg|años?|years?|meses?|months?)\b",
        r"\b(paciente|patient|niñ[oa]|child|beb[eé]|baby|pesa|weighs?|kg|años?|years?|meses?|months?)\b.{0,80}\b(dosis|dose)\b",
        r"\b(cu[aá]ntos?|how many)\s*(mg|ml)\b.{0,80}\b(dar|give|administrar|administer|tomar|take)\b",
        r"\b(interpreta|interprete|interpret)\b.{0,45}\b(resultado|result|laboratorio|lab|imagen|image|radiograf)\w*",
        r"\b(decide|determine|debo|should)\b.{0,45}\b(ingresar|admit|alta|discharge|disposición|disposition)\b",
    )
)


def is_restricted_request(question: str) -> bool:
    """Identify requests that may receive context but not a patient-care decision."""
    return any(pattern.search(question) for pattern in _RESTRICTED_PATTERNS)


_PATIENT_SPECIFIC_DOSE_PATTERN = re.compile(
    r"(?:\b(?:dosis|dose)\b.{0,100}\b\d+(?:[.,]\d+)?\s*(?:kg|años?|years?|meses?|months?)\b)"
    r"|(?:\b\d+(?:[.,]\d+)?\s*(?:kg|años?|years?|meses?|months?)\b.{0,100}\b(?:dosis|dose)\b)",
    re.IGNORECASE,
)


def is_patient_specific_dose_request(question: str) -> bool:
    """Detect dose requests that include an individual age or weight."""
    return bool(_PATIENT_SPECIFIC_DOSE_PATTERN.search(question))
