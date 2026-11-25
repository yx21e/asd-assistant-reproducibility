from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


DEFAULT_AUTISM_BASE_QUERY = (
    '("autism spectrum disorder" OR autism OR autistic OR "autistic disorder" '
    'OR "Asperger syndrome" OR "pervasive developmental disorder" '
    'OR "childhood autism" OR "early infantile autism") '
    'NOT ("atrial septal defect" OR "heart septal defects, atrial" '
    'OR "adjacent segment degeneration" OR "adult spinal deformity" '
    'OR "acute stress disorder")'
)


_AUTISM_POSITIVE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\bautism spectrum disorder\b",
        r"\bautism spectrum disorders\b",
        r"\bautistic disorder\b",
        r"\bautistic\b",
        r"\basperger(?:'s)? syndrome\b",
        r"\bpervasive developmental disorder\b",
        r"\bchildhood autism\b",
        r"\bearly infantile autism\b",
        r"\bautism\b",
    )
]

_AUTISM_NEGATIVE_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in (
        r"\batrial septal defect\b",
        r"\bheart septal defects,\s*atrial\b",
        r"\badjacent segment degeneration\b",
        r"\badult spinal deformity\b",
        r"\bacute stress disorder\b",
        r"\bostium secundum\b",
        r"\bcongenital heart disease\b",
        r"\bechocardiograph",
        r"\bseptal occluder\b",
        r"\bpatent foramen ovale\b",
    )
]

_TOPIC_PATTERNS: Dict[str, tuple[re.Pattern[str], ...]] = {
    "diagnosis_screening": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bdiagnos",
            r"\bscreen",
            r"\bassess",
            r"\bADOS\b",
            r"\bADI-R\b",
            r"\bM-CHAT\b",
            r"\bquestionnaire\b",
            r"\bscale\b",
            r"\binstrument\b",
            r"\bvalidation\b",
        )
    ),
    "biomarkers": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bbiomarker",
            r"\bseroton",
            r"\bmicrobiome\b",
            r"\bmetabol",
            r"\bEEG\b",
            r"\bMRI\b",
            r"\bfunctional MRI\b",
            r"\bblood\b",
            r"\bproteom",
            r"\bmetabolom",
        )
    ),
    "epidemiology_population": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bprevalence\b",
            r"\bincidence\b",
            r"\bepidemiolog",
            r"\bpopulation\b",
            r"\bcohort\b",
            r"\bsex differen",
            r"\bgender\b",
            r"\bdisparit",
            r"\bunderdiagnos",
            r"\badults?\b",
        )
    ),
    "methods_study_design": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bsystematic review\b",
            r"\bmeta-analysis\b",
            r"\bmeta analysis\b",
            r"\breview\b",
            r"\btrial\b",
            r"\brandomized\b",
            r"\bmachine learning\b",
            r"\bdeep learning\b",
            r"\bstudy design\b",
            r"\bmethod(?:s|ology)?\b",
        )
    ),
    "risk_pathogenesis": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\brisk factor",
            r"\betiolog",
            r"\bpathogenes",
            r"\bmechanis",
            r"\bgenetic",
            r"\bprenatal\b",
            r"\benvironment",
            r"\bimmune\b",
            r"\bmicrogl",
            r"\bgut-brain\b",
        )
    ),
    "clinical_features": tuple(
        re.compile(pattern, flags=re.IGNORECASE)
        for pattern in (
            r"\bclinical\b",
            r"\bphenotyp",
            r"\bsymptom",
            r"\bdevelopmental\b",
            r"\bsocial communication\b",
            r"\brepetitive behavio",
            r"\blanguage\b",
            r"\bsensory\b",
            r"\bpragmatic\b",
            r"\bfunction(?:al)? profile\b",
        )
    ),
}


def normalize_task3_text(*parts: object) -> str:
    return re.sub(r"\s+", " ", " ".join(str(part or "") for part in parts)).strip()


def has_autism_signal(
    title: object = "",
    abstract: object = "",
    keywords: object = "",
    publication_type: object = "",
) -> bool:
    text = normalize_task3_text(title, abstract, keywords, publication_type)
    if not text:
        return False
    return any(pattern.search(text) for pattern in _AUTISM_POSITIVE_PATTERNS)


def is_probable_non_autism_record(
    title: object = "",
    abstract: object = "",
    keywords: object = "",
    publication_type: object = "",
) -> bool:
    text = normalize_task3_text(title, abstract, keywords, publication_type)
    if not text:
        return True
    has_positive = has_autism_signal(title=title, abstract=abstract, keywords=keywords, publication_type=publication_type)
    has_negative = any(pattern.search(text) for pattern in _AUTISM_NEGATIVE_PATTERNS)
    if has_negative and not has_positive:
        return True
    if re.search(r"\bASD\b", text, flags=re.IGNORECASE) and not has_positive:
        return True
    return False


def infer_task3_topic(
    title: object = "",
    abstract: object = "",
    keywords: object = "",
    publication_type: object = "",
) -> str:
    text = normalize_task3_text(title, abstract, keywords, publication_type)
    if not text:
        return "general_autism"

    scores = {}
    for topic, patterns in _TOPIC_PATTERNS.items():
        scores[topic] = sum(1 for pattern in patterns if pattern.search(text))

    best_topic = "general_autism"
    best_score = 0
    for topic in (
        "diagnosis_screening",
        "biomarkers",
        "epidemiology_population",
        "methods_study_design",
        "risk_pathogenesis",
        "clinical_features",
    ):
        score = scores.get(topic, 0)
        if score > best_score:
            best_topic = topic
            best_score = score
    return best_topic


def topic_anchor_terms(topic: str) -> list[str]:
    mapping = {
        "clinical_features": ["clinical", "development", "social", "behavior", "language", "sensory"],
        "diagnosis_screening": ["screen", "diagnos", "assessment", "questionnaire", "instrument"],
        "biomarkers": ["biomarker", "seroton", "microbiome", "eeg", "mri", "blood"],
        "epidemiology_population": ["prevalence", "incidence", "population", "cohort", "sex", "gender", "adult", "disparit", "underdiagnos", "demograph"],
        "methods_study_design": ["review", "meta", "trial", "cohort", "method", "study"],
        "risk_pathogenesis": ["risk", "etiology", "mechanism", "genetic", "prenatal", "environment"],
        "general_autism": ["autism", "spectrum", "disorder"],
    }
    return mapping.get(topic, mapping["general_autism"])


def count_anchor_hits(text: str, anchors: Iterable[str]) -> int:
    raw = normalize_task3_text(text).lower()
    return sum(anchor.lower() in raw for anchor in anchors)


def has_review_signal(
    publication_type: object = "",
    title: object = "",
    keywords: object = "",
) -> bool:
    text = normalize_task3_text(publication_type, title, keywords).lower()
    if not text:
        return False
    review_markers = (
        "review",
        "systematic review",
        "meta-analysis",
        "meta analysis",
        "scoping review",
        "umbrella review",
        "narrative review",
    )
    return any(marker in text for marker in review_markers)


def wants_review_style(question: str) -> bool:
    q = normalize_task3_text(question).lower()
    cues = (
        "summarize",
        "overview",
        "what is known",
        "literature",
        "evidence",
        "review",
        "broad themes",
        "high-level",
    )
    return any(cue in q for cue in cues)


def wants_recent_evidence(question: str) -> bool:
    q = normalize_task3_text(question).lower()
    return any(token in q for token in ("recent", "current", "latest", "modern"))


def infer_topic_from_question(question: str) -> Optional[str]:
    q = normalize_task3_text(question).lower()
    if not q:
        return None

    scores = {}
    for topic, patterns in _TOPIC_PATTERNS.items():
        scores[topic] = sum(1 for pattern in patterns if pattern.search(q))

    if any(token in q for token in ("underdiagnos", "disparit", "demograph", "race", "racial", "ethnic")):
        scores["epidemiology_population"] = scores.get("epidemiology_population", 0) + 2
    if any(token in q for token in ("research direction", "research directions", "broad theme", "broad themes", "high-level overview")):
        scores["general_autism"] = scores.get("general_autism", 0) + 2
    if any(token in q for token in ("ados", "adi-r", "m-chat", "questionnaire", "instrument")):
        scores["diagnosis_screening"] = scores.get("diagnosis_screening", 0) + 2

    best_topic = None
    best_score = 0
    for topic in (
        "epidemiology_population",
        "diagnosis_screening",
        "biomarkers",
        "methods_study_design",
        "risk_pathogenesis",
        "clinical_features",
        "general_autism",
    ):
        score = scores.get(topic, 0)
        if score > best_score:
            best_topic = topic
            best_score = score

    if best_topic and best_score > 0:
        return best_topic
    if "autism" in q:
        return "general_autism"
    return None
