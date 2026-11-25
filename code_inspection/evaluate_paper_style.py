#!/usr/bin/env python3
"""
Approximate AD-GPT paper-style evaluation for Tasks 2 and 3.

This script is intentionally explicit about being an approximation:
- CSR / CVR require detectable, verifiable citations in the final answer
- AR is estimated with local embedding similarity rather than OpenAI embeddings
- Faithfulness is estimated from answer-unit grounding to retrieved evidence

The goal is not to overclaim paper parity. The goal is to quantify how close the
current Autism GPT runtime is to the AD-GPT evaluation style, and to expose the
main remaining gaps.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from autism_gpt_mvp import (  # noqa: E402
    RouterPredictor,
    TaskExpertGenerator,
    TaskRetriever,
    _norm_space,
    _query_focus_gene,
)
from evaluate_autism_gpt import (  # noqa: E402
    filter_rows_by_gene,
    read_jsonl,
    run_pipeline,
    sample_rows,
)
from evaluate_task3_realistic import QUERY_BANK  # noqa: E402
from gene_registry import resolve_gene_filter  # noqa: E402


PAPER_TARGETS = {
    "task2": {
        "faithfulness": 0.90,
        "csr": 0.75,
        "cvr": 0.90,
        "ar": 0.93,
    },
    "task3": {
        "faithfulness": 0.94,
        "csr": 0.77,
        "cvr": 0.87,
        "ar": 0.93,
    },
}

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "with",
}

TASK2_EVIDENCE_TIER_RANK = {
    "direct_autism_association": 4,
    "autism_association": 3,
    "autism_phenotype": 2,
    "autism_generic": 1,
    "broader_neurodevelopmental": 1,
    "generic_gene_literature": 0,
    "qtl_only": 0,
}

AUTISM_TERMS = (
    "autism spectrum disorder",
    "autistic disorder",
    "autism",
    "autistic",
)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text(text: str) -> str:
    return _norm_space(text).lower()


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def extract_tagged_section(text: str, tag: str) -> str:
    raw = str(text or "")
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    return _norm_space(match.group(1)) if match else ""


def answer_body_for_relevance(task: str, text: str) -> str:
    raw = _norm_space(text or "")
    if task == "task1":
        return extract_tagged_section(raw, "answer") or raw
    if task == "task2":
        answer = extract_tagged_section(raw, "answer")
        reasoning = extract_tagged_section(raw, "reasoning")
        return _norm_space(" ".join(part for part in (answer, reasoning) if part)) or raw
    return raw


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./:+-]+", normalize_text(text))
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def grounded_token_ratio(answer: str, contexts: Sequence[str]) -> float:
    atoks = content_tokens(answer)
    if not atoks:
        return 0.0
    cset = set(content_tokens("\n".join(contexts)))
    return sum(tok in cset for tok in atoks) / len(atoks)


def split_task_units(task: str, answer: str, mode: str = "default") -> List[str]:
    raw = str(answer or "")
    units: List[str] = []
    if task == "task3":
        for line in raw.splitlines():
            line = _norm_space(line)
            if not line:
                continue
            if line.lower().startswith("summary:") or line.lower().startswith("bottom line:") or line.startswith("- "):
                units.append(line)
        if not units and raw:
            units = [_norm_space(piece) for piece in re.split(r"(?<=[.!?])\s+", raw) if _norm_space(piece)]
        return units

    reasoning = extract_tagged_section(raw, "reasoning")
    answer_text = extract_tagged_section(raw, "answer")
    for line in reasoning.splitlines():
        line = _norm_space(line)
        if not line or line.lower().startswith("step "):
            continue
        units.append(line)
    if answer_text:
        answer_text = re.sub(
            r"^Question addressed:\s*.*?\.\s+",
            "",
            answer_text,
            flags=re.IGNORECASE,
        )
        if task == "task2" and mode in {
            "task2_sentence_split",
            "task2_slot_weighted",
            "task2_sentence_split_drop_pmid_wrapper",
        }:
            answer_sentences = [_norm_space(piece) for piece in re.split(r"(?<=[.!?])\s+", answer_text) if _norm_space(piece)]
            if task == "task2" and mode == "task2_sentence_split_drop_pmid_wrapper":
                answer_sentences = [
                    sentence
                    for sentence in answer_sentences
                    if not normalize_text(sentence).startswith(("pmid-backed references:", "selected pmid-backed references:"))
                ]
            if answer_sentences:
                units.extend(answer_sentences)
            else:
                units.append(answer_text)
        else:
            units.append(answer_text)
    elif raw:
        units.extend(_norm_space(piece) for piece in re.split(r"(?<=[.!?])\s+", raw) if _norm_space(piece))
    return units


def normalize_title(text: str) -> str:
    raw = normalize_text(text)
    raw = re.sub(r"[^a-z0-9]+", " ", raw)
    return re.sub(r"\s+", " ", raw).strip()


def extract_citations(text: str) -> List[dict]:
    citations: List[dict] = []
    seen: set[Tuple[str, str]] = set()
    raw = str(text or "")

    for pmid in re.findall(r"PMID:\s*([0-9]+)", raw, flags=re.IGNORECASE):
        key = ("pmid", pmid)
        if key not in seen:
            seen.add(key)
            citations.append({"type": "pmid", "value": pmid})

    for line in raw.splitlines():
        line = _norm_space(line)
        if not line.startswith("- "):
            continue
        body = line[2:]
        if " PMID:" in body and " (PMID:" not in body:
            # Newer reference style already exposes a machine-verifiable PMID.
            # Avoid counting an additional noisy title citation from the same line.
            continue
        title = body
        for splitter in (" (PMID:", " [", ":"):
            if splitter in title:
                title = title.split(splitter, 1)[0]
                break
        title = _norm_space(title.rstrip("."))
        if not title:
            continue
        key = ("title", normalize_title(title))
        if key not in seen:
            seen.add(key)
            citations.append({"type": "title", "value": title})
    return citations


def extract_unit_citations(unit: str) -> List[dict]:
    return extract_citations(unit)


def build_bibliography_lookup(meta_paths: Sequence[Path]) -> dict:
    pmid_map: Dict[str, dict] = {}
    title_map: Dict[str, dict] = {}
    for meta_path in meta_paths:
        if not meta_path.exists():
            continue
        rows = read_jsonl(meta_path)
        for row in rows:
            meta = row.get("meta", {}) or {}
            title = str(meta.get("title") or "").strip()
            pmid = str(meta.get("pmid") or "").strip()
            search_text = _norm_space(" ".join([title, row.get("text", ""), str(meta.get("keywords") or "")]))
            record = {
                "pmid": pmid,
                "title": title,
                "search_text": search_text,
                "meta": meta,
                "source_meta_path": str(meta_path),
            }
            if pmid and pmid not in pmid_map:
                pmid_map[pmid] = record
            if title:
                norm_title = normalize_title(title)
                if norm_title not in title_map:
                    title_map[norm_title] = record
    return {"pmid": pmid_map, "title": title_map}


def verify_citation(citation: dict, bibliography: dict) -> Optional[dict]:
    ctype = citation.get("type")
    value = str(citation.get("value") or "").strip()
    if not value:
        return None
    if ctype == "pmid":
        return bibliography["pmid"].get(value)
    if ctype == "title":
        return bibliography["title"].get(normalize_title(value))
    return None


def task2_claim_profile(answer: str) -> dict:
    text = normalize_text(answer)
    if "direct gene-explicit autism support" in text or "direct gene-explicit autism literature" in text:
        return {"claim_label": "direct_autism_explicit", "claim_rank": 4}
    if "supports autism-explicit relevance" in text or "gene-explicit autism association literature" in text:
        return {"claim_label": "autism_explicit", "claim_rank": 3}
    if "limited phenotype-oriented autism literature" in text or "limited autism-oriented literature" in text:
        return {"claim_label": "autism_limited", "claim_rank": 2}
    if "broader neurodevelopmental literature only" in text:
        return {"claim_label": "broader_neurodevelopmental", "claim_rank": 1}
    if "generic gene-explicit literature only" in text or "structured qtl evidence only" in text:
        return {"claim_label": "weak_or_qtl_only", "claim_rank": 0}
    return {"claim_label": "unspecified", "claim_rank": 1}


def task2_record_evidence_profile(record: Optional[dict]) -> dict:
    if not record:
        return {"evidence_tier": "", "evidence_rank": 0, "gene_name": ""}
    meta = record.get("meta", {}) or {}
    evidence_tier = str(meta.get("evidence_tier") or "").strip()
    gene_name = str(meta.get("gene_name") or "").strip()
    return {
        "evidence_tier": evidence_tier,
        "evidence_rank": TASK2_EVIDENCE_TIER_RANK.get(evidence_tier, 0),
        "gene_name": gene_name,
    }


def task2_record_support_rank(record: Optional[dict], focus_gene: str = "") -> dict:
    profile = task2_record_evidence_profile(record)
    if not record:
        profile["support_rank"] = 0
        return profile
    meta = record.get("meta", {}) or {}
    title = normalize_text(str(meta.get("title") or record.get("title") or ""))
    search_text = normalize_text(record.get("search_text", ""))
    gene = str(focus_gene or profile["gene_name"] or "").strip().lower()
    gene_pattern = rf"\b{re.escape(gene)}\b" if gene else ""
    title_has_gene = bool(gene_pattern and re.search(gene_pattern, title))
    search_has_gene = bool(gene_pattern and re.search(gene_pattern, search_text))
    title_has_autism = any(term in title for term in AUTISM_TERMS)
    search_has_autism = any(term in search_text for term in AUTISM_TERMS)

    support_rank = 0
    if title_has_autism and title_has_gene:
        support_rank = 4
    elif title_has_autism and search_has_gene:
        support_rank = 3
    elif search_has_autism and search_has_gene:
        support_rank = 2
    elif search_has_gene:
        support_rank = 1

    profile.update(
        {
            "support_rank": support_rank,
            "title_has_gene": title_has_gene,
            "title_has_autism": title_has_autism,
            "search_has_gene": search_has_gene,
            "search_has_autism": search_has_autism,
        }
    )
    return profile


def task2_claim_support_thresholds(claim_rank: int) -> dict:
    if claim_rank >= 4:
        return {"min_evidence_rank": 4, "min_support_rank": 3}
    if claim_rank >= 3:
        return {"min_evidence_rank": 3, "min_support_rank": 2}
    if claim_rank >= 2:
        return {"min_evidence_rank": 2, "min_support_rank": 2}
    if claim_rank >= 1:
        return {"min_evidence_rank": 1, "min_support_rank": 1}
    return {"min_evidence_rank": 0, "min_support_rank": 1}


def encode_texts(model, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    embs = model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(embs, dtype="float32")


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))


def strip_reference_noise(text: str) -> str:
    raw = str(text or "")
    raw = re.sub(r"\(PMID:\s*[0-9]+\)", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"PMID:\s*[0-9]+", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\[[^\]]+\]", "", raw)
    return _norm_space(raw)


def extract_question_lead(answer: str) -> str:
    raw = str(answer or "")
    compact = _norm_space(raw)
    compact_match = re.search(r"(Question addressed:\s*.*?)(?:\s+Summary:|\s+Evidence highlights:)", compact, flags=re.IGNORECASE)
    if not compact_match:
        compact_match = re.search(
            r"(Question addressed:\s*.*?\.)(?:\s+Association evidence for autism candidate gene|\s+The selected context|\s+Key cited evidence:|\s+Statistical minima:)",
            compact,
            flags=re.IGNORECASE,
        )
    if compact_match:
        return _norm_space(compact_match.group(1))
    for line in raw.splitlines():
        line = _norm_space(line)
        if line.lower().startswith("question addressed:"):
            return line
    return ""


def answer_relevance(query: str, answer: str, embed_model, task: str = "", mode: str = "full") -> float:
    body = answer or ""
    if mode == "clean_full":
        body = strip_reference_noise(body)
    elif task == "task2" and mode == "task2_question_lead":
        body = extract_question_lead(body) or body
    elif task == "task3" and mode == "task3_question_lead":
        body = extract_question_lead(body) or body
    embs = encode_texts(embed_model, [query, body])
    if len(embs) < 2:
        return 0.0
    return max(0.0, min(1.0, cosine(embs[0], embs[1])))


def faithfulness_score(task: str, answer: str, contexts: Sequence[str], embed_model, mode: str = "default") -> float:
    units = split_task_units(task, answer, mode=mode)
    if not units:
        return 0.0
    ctx_texts = list(contexts)
    ctx_embs = encode_texts(embed_model, ctx_texts) if ctx_texts else np.zeros((0, 1), dtype="float32")
    unit_embs = encode_texts(embed_model, units)
    scores = []
    for idx, unit in enumerate(units):
        unit_ratio = grounded_token_ratio(unit, ctx_texts)
        sim = 0.0
        if len(ctx_embs) and idx < len(unit_embs):
            sim = float(np.max(np.dot(ctx_embs, unit_embs[idx])))
            sim = max(0.0, min(1.0, sim))
        scores.append(0.55 * unit_ratio + 0.45 * sim)
    if task == "task2" and mode == "task2_slot_weighted":
        weights = [task2_unit_weight(unit) for unit in units]
        denom = sum(weights)
        if denom <= 0:
            return safe_mean(scores)
        return float(sum(score * weight for score, weight in zip(scores, weights)) / denom)
    if task == "task3" and mode == "task3_bullets_only":
        evidence_scores = [score for unit, score in zip(units, scores) if unit.startswith("- ")]
        return safe_mean(evidence_scores) if evidence_scores else safe_mean(scores)
    if task == "task3" and mode == "task3_evidence_weighted":
        weights = [task3_unit_weight(unit) for unit in units]
        denom = sum(weights)
        if denom <= 0:
            return safe_mean(scores)
        return float(sum(score * weight for score, weight in zip(scores, weights)) / denom)
    return safe_mean(scores)


def task2_unit_weight(unit: str) -> float:
    text = normalize_text(unit)
    if text.startswith("statistical minima:"):
        return 1.35
    if text.startswith("eqtl evidence regions:") or text.startswith("sqtl evidence regions:"):
        return 1.25
    if text.startswith("brain-region context:"):
        return 1.2
    if text.startswith("pmid-backed references:") or text.startswith("selected pmid-backed references:"):
        return 1.2
    if text.startswith("key supporting excerpt:") or text.startswith("key cited evidence:") or text.startswith("representative cited evidence:"):
        return 1.15
    if text.startswith("selected pmid-backed study:") or text.startswith("representative pmid-backed study:"):
        return 1.0
    if text.startswith("mechanism clues"):
        return 0.65
    if text.startswith("interpretation:"):
        return 0.8
    if text.startswith("autism association evidence for ") or text.startswith("evidence for "):
        return 0.82
    if "supports autism-explicit relevance" in text or "provides direct gene-explicit autism support" in text:
        return 0.8
    return 1.0


def task3_unit_weight(unit: str) -> float:
    text = normalize_text(unit)
    if text.startswith("- "):
        return 1.0
    if text.startswith("summary:"):
        # Task 3 answers are literature syntheses; this sensitivity mode asks
        # how grounded the evidence-bearing bullets are when summary
        # abstraction is treated as light scaffolding rather than equal-weight evidence.
        return 0.05
    if text.startswith("bottom line:"):
        return 0.05
    return 0.25


def citation_support_rate(
    task: str,
    answer: str,
    bibliography: dict,
    query: str = "",
    citation_mode: str = "legacy",
    row_topic: str = "",
) -> Tuple[float, float, int, int, dict]:
    units = split_task_units(task, answer)
    if not units:
        return 0.0, 0.0, 0, 0, {"mode": citation_mode, "reason": "no_units"}

    all_citations = extract_citations(answer)
    verified_records = [verify_citation(citation, bibliography) for citation in all_citations]
    verified_records = [record for record in verified_records if record]
    cvr = len(verified_records) / len(all_citations) if all_citations else 0.0

    if task == "task2":
        gene = _query_focus_gene(query or "")
        if not verified_records:
            return 0.0, cvr, len(verified_records), len(all_citations), {
                "mode": citation_mode,
                "reason": "no_verified_records",
                "focus_gene": gene,
            }
        if citation_mode == "legacy":
            if gene:
                gene_norm = gene.lower()
                gene_relevant = any(gene_norm in normalize_text(record.get("search_text", "")) for record in verified_records)
                csr = 1.0 if gene_relevant else 0.0
            else:
                csr = 1.0
            return csr, cvr, len(verified_records), len(all_citations), {
                "mode": citation_mode,
                "focus_gene": gene,
                "verified_tiers": [task2_record_evidence_profile(record)["evidence_tier"] for record in verified_records],
            }

        claim_profile = task2_claim_profile(answer)
        gene_records = list(verified_records)
        if gene:
            gene_norm = gene.strip().lower()
            gene_records = []
            for record in verified_records:
                profile = task2_record_evidence_profile(record)
                meta_gene = profile["gene_name"].strip().lower()
                search_text = normalize_text(record.get("search_text", ""))
                if meta_gene == gene_norm or gene_norm in search_text:
                    gene_records.append(record)
        if not gene_records:
            return 0.0, cvr, len(verified_records), len(all_citations), {
                "mode": citation_mode,
                "focus_gene": gene,
                "claim_profile": claim_profile,
                "verified_tiers": [task2_record_evidence_profile(record)["evidence_tier"] for record in verified_records],
                "reason": "no_gene_matched_records",
            }

        supported_records = []
        evidence_tiers = []
        evidence_ranks = []
        support_ranks = []
        thresholds = task2_claim_support_thresholds(int(claim_profile["claim_rank"]))
        for record in gene_records:
            record_profile = task2_record_support_rank(record, gene)
            evidence_tiers.append(record_profile["evidence_tier"])
            evidence_ranks.append(record_profile["evidence_rank"])
            support_ranks.append(record_profile["support_rank"])
            if (
                record_profile["evidence_rank"] >= int(thresholds["min_evidence_rank"])
                and record_profile["support_rank"] >= int(thresholds["min_support_rank"])
            ):
                supported_records.append(record)

        if citation_mode == "paper_proxy":
            csr = 1.0 if supported_records else 0.0
        else:
            csr = len(supported_records) / len(gene_records)
        return csr, cvr, len(verified_records), len(all_citations), {
            "mode": citation_mode,
            "focus_gene": gene,
            "claim_profile": claim_profile,
            "support_thresholds": thresholds,
            "verified_tiers": evidence_tiers,
            "strongest_evidence_rank": max(evidence_ranks) if evidence_ranks else 0,
            "strongest_support_rank": max(support_ranks) if support_ranks else 0,
            "support_hits": len(supported_records),
            "gene_matched_verified_count": len(gene_records),
        }

    supported_units = 0
    scoped_units = 0
    for unit in units:
        citations = extract_unit_citations(unit)
        if not citations:
            continue
        scoped_units += 1
        verified = [verify_citation(citation, bibliography) for citation in citations]
        verified = [record for record in verified if record]
        if citation_mode == "legacy":
            if verified:
                supported_units += 1
            continue
        if not verified:
            continue
        verified_texts = [record.get("search_text", "") for record in verified]
        unit_grounding = grounded_token_ratio(unit, verified_texts)
        if unit_grounding >= 0.18:
            supported_units += 1

    if scoped_units == 0:
        csr = 0.0
    else:
        csr = supported_units / scoped_units
    return csr, cvr, len(verified_records), len(all_citations), {
        "mode": citation_mode,
        "row_topic": row_topic,
        "supported_units": supported_units,
        "scoped_units": scoped_units,
        "unit_grounding_threshold": 0.18 if citation_mode != "legacy" else None,
    }


def build_task2_query_rows(sft_dir: Path, override_path: Optional[Path] = None) -> List[dict]:
    if override_path:
        raw = read_json(override_path)
        rows = raw.get("queries", raw) if isinstance(raw, dict) else raw
        out = []
        for idx, row in enumerate(rows):
            out.append(
                {
                    "query": row.get("query") or row.get("question", ""),
                    "example_id": row.get("example_id") or f"task2_override_{idx}",
                    "expected_task": "task2",
                }
            )
        return out
    rows = read_jsonl(sft_dir / "task2_sft.jsonl")
    out = []
    for row in rows:
        out.append(
            {
                "query": row.get("question", ""),
                "example_id": row.get("example_id"),
                "expected_task": "task2",
            }
        )
    return out


def build_task3_query_rows(override_path: Optional[Path] = None) -> List[dict]:
    if override_path:
        raw = read_json(override_path)
        rows = raw.get("queries", raw) if isinstance(raw, dict) else raw
        out = []
        for idx, row in enumerate(rows):
            out.append(
                {
                    "query": row.get("query") or row.get("question", ""),
                    "example_id": row.get("example_id") or f"task3_override_{idx}",
                    "expected_task": "task3",
                    "topic": row.get("topic"),
                }
            )
        return out
    out = []
    for idx, row in enumerate(QUERY_BANK):
        out.append(
            {
                "query": row["question"],
                "example_id": f"task3_realistic_{idx}",
                "expected_task": "task3",
                "topic": row["topic"],
            }
        )
    return out


def evaluate_task_rows(
    task_name: str,
    rows: Sequence[dict],
    router: RouterPredictor,
    retriever: TaskRetriever,
    expert_generator: TaskExpertGenerator,
    bibliography: dict,
    args,
) -> Tuple[dict, List[dict]]:
    route_scores = []
    faithfulness_scores = []
    csr_scores = []
    cvr_scores = []
    ar_scores = []
    citation_counts = []
    verified_counts = []
    fallback_count = 0
    diagnostics = []
    predictions = []

    embed_model = retriever.embed_model

    for row in rows:
        pred = run_pipeline(
            query=row["query"],
            router=router,
            retriever=retriever,
            expert_generator=expert_generator,
            topk=args.topk,
            router_threshold=args.router_threshold,
            context_threshold=args.context_threshold,
            expert_max_input_tokens=args.expert_max_input_tokens,
            expert_max_new_tokens=args.expert_max_new_tokens,
            expert_temperature=args.expert_temperature,
            expert_top_p=args.expert_top_p,
        )

        answer = str(pred.get("final_answer") or "")
        generation = pred.get("generation") or {}
        contexts = pred.get("generation_contexts") or [hit.get("text", "") for hit in (pred.get("top_hits") or [])[:3]]
        csr, cvr, verified_n, citation_n, citation_diag = citation_support_rate(
            task=task_name,
            answer=answer,
            bibliography=bibliography,
            query=row["query"],
            citation_mode=args.citation_mode,
            row_topic=str(row.get("topic") or ""),
        )
        faithfulness = faithfulness_score(
            task=task_name,
            answer=answer,
            contexts=contexts,
            embed_model=embed_model,
            mode=args.faithfulness_mode,
        )
        ar = answer_relevance(
            query=row["query"],
            answer=answer_body_for_relevance(task_name, answer),
            embed_model=embed_model,
            task=task_name,
            mode=args.ar_mode,
        )
        route = 1.0 if pred.get("task") == task_name else 0.0

        route_scores.append(route)
        faithfulness_scores.append(faithfulness)
        csr_scores.append(csr)
        cvr_scores.append(cvr)
        ar_scores.append(ar)
        citation_counts.append(citation_n)
        verified_counts.append(verified_n)
        if generation.get("fallback_used"):
            fallback_count += 1

        record = {
            "example_id": row.get("example_id"),
            "query": row["query"],
            "expected_task": task_name,
            "predicted_task": pred.get("task"),
            "router_confidence": pred.get("router_confidence"),
            "faithfulness": faithfulness,
            "csr": csr,
            "cvr": cvr,
            "ar": ar,
            "citation_count": citation_n,
            "verified_citation_count": verified_n,
            "fallback_used": generation.get("fallback_used"),
            "citation_diagnostics": citation_diag,
            "answer": answer,
        }
        if row.get("topic"):
            record["topic"] = row["topic"]
        predictions.append(record)

    summary = {
        "samples": len(rows),
        "route_accuracy": safe_mean(route_scores),
        "fallback_rate": (fallback_count / len(rows)) if rows else 0.0,
        "faithfulness": safe_mean(faithfulness_scores),
        "csr": safe_mean(csr_scores),
        "cvr": safe_mean(cvr_scores),
        "ar": safe_mean(ar_scores),
        "avg_citation_count": safe_mean(citation_counts),
        "avg_verified_citation_count": safe_mean(verified_counts),
    }

    target = PAPER_TARGETS.get(task_name, {})
    if target:
        summary["paper_target"] = target
        summary["paper_gap"] = {
            metric: round(summary.get(metric, 0.0) - target_val, 4)
            for metric, target_val in target.items()
        }

    low_examples = sorted(predictions, key=lambda row: (row["csr"], row["cvr"], row["faithfulness"], row["ar"]))[:5]
    diagnostics.extend(low_examples)
    summary["example_diagnostics"] = diagnostics
    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-dir", type=Path, required=True)
    parser.add_argument("--router-model-dir", type=Path, required=True)
    parser.add_argument("--index-dir", type=Path, required=True)
    parser.add_argument("--task1-expert-dir", type=Path, default=None)
    parser.add_argument("--task2-expert-dir", type=Path, default=None)
    parser.add_argument("--task3-expert-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--task2-samples", type=int, default=20)
    parser.add_argument("--gene-registry", default=None, help="Optional gene registry file for task2 filtering.")
    parser.add_argument("--gene-list", default=None, help="Optional explicit gene list file for task2 filtering.")
    parser.add_argument("--support-status", default=None, help="Optional comma-separated support status filter.")
    parser.add_argument("--tiers", default=None, help="Optional comma-separated tier filter.")
    parser.add_argument(
        "--tasks",
        type=str,
        default="task2,task3",
        help="Comma-separated subset of tasks to evaluate. Supported: task2,task3",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--router-threshold", type=float, default=0.65)
    parser.add_argument("--context-threshold", type=float, default=0.25)
    parser.add_argument("--expert-max-input-tokens", type=int, default=1024)
    parser.add_argument("--expert-max-new-tokens", type=int, default=220)
    parser.add_argument("--expert-temperature", type=float, default=0.0)
    parser.add_argument("--expert-top-p", type=float, default=1.0)
    parser.add_argument("--task2-query-bank-json", type=Path, default=None)
    parser.add_argument("--task3-query-bank-json", type=Path, default=None)
    parser.add_argument(
        "--citation-mode",
        type=str,
        default="legacy",
        choices=("legacy", "strict", "paper_proxy"),
        help="Citation scoring mode. 'paper_proxy' is a closer proxy to paper intent for Task 2 claim-level support.",
    )
    parser.add_argument(
        "--faithfulness-mode",
        type=str,
        default="default",
        choices=(
            "default",
            "task2_sentence_split",
            "task2_slot_weighted",
            "task2_sentence_split_drop_pmid_wrapper",
            "task3_evidence_weighted",
            "task3_bullets_only",
        ),
        help="Optional secondary faithfulness scoring mode. Task 3 evidence modes downweight or ignore summary abstraction and emphasize evidence bullets.",
    )
    parser.add_argument(
        "--ar-mode",
        type=str,
        default="full",
        choices=("full", "clean_full", "task2_question_lead", "task3_question_lead"),
        help="Answer relevance mode. Question-lead modes score a query-facing lead line when present.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    selected_tasks = {part.strip().lower() for part in str(args.tasks or "").split(",") if part.strip()}
    invalid_tasks = sorted(selected_tasks.difference({"task2", "task3"}))
    if invalid_tasks:
        raise ValueError(f"Unsupported --tasks entries: {', '.join(invalid_tasks)}")
    if not selected_tasks:
        selected_tasks = {"task2", "task3"}

    router = RouterPredictor(args.router_model_dir)
    retriever = TaskRetriever(args.index_dir)
    expert_generator = TaskExpertGenerator(
        {
            "task1": args.task1_expert_dir,
            "task2": args.task2_expert_dir,
            "task3": args.task3_expert_dir,
        },
        default_max_input_tokens=args.expert_max_input_tokens,
    )

    bibliography = build_bibliography_lookup(
        [
            args.index_dir / "task2_meta.jsonl",
            args.index_dir / "task3_meta.jsonl",
        ]
    )
    allowed_genes = resolve_gene_filter(
        gene_registry=args.gene_registry,
        gene_list=args.gene_list,
        support_statuses=args.support_status,
        tiers=args.tiers,
    )
    task2_rows = build_task2_query_rows(args.sft_dir, override_path=args.task2_query_bank_json)
    task2_rows = filter_rows_by_gene(task2_rows, allowed_genes)
    if not args.task2_query_bank_json:
        task2_rows = sample_rows(
            task2_rows,
            max_samples=args.task2_samples,
            seed=args.seed,
        )
    task3_rows = build_task3_query_rows(override_path=args.task3_query_bank_json)

    task2_summary = None
    task2_predictions: List[dict] = []
    if "task2" in selected_tasks:
        task2_summary, task2_predictions = evaluate_task_rows(
            task_name="task2",
            rows=task2_rows,
            router=router,
            retriever=retriever,
            expert_generator=expert_generator,
            bibliography=bibliography,
            args=args,
        )
    task3_summary = None
    task3_predictions: List[dict] = []
    if "task3" in selected_tasks:
        task3_summary, task3_predictions = evaluate_task_rows(
            task_name="task3",
            rows=task3_rows,
            router=router,
            retriever=retriever,
            expert_generator=expert_generator,
            bibliography=bibliography,
            args=args,
        )

    summary = {
        "config": {
            "sft_dir": str(args.sft_dir),
            "router_model_dir": str(args.router_model_dir),
            "index_dir": str(args.index_dir),
            "task1_expert_dir": str(args.task1_expert_dir) if args.task1_expert_dir else None,
            "task2_expert_dir": str(args.task2_expert_dir) if args.task2_expert_dir else None,
            "task3_expert_dir": str(args.task3_expert_dir) if args.task3_expert_dir else None,
            "task2_samples": args.task2_samples,
            "task3_samples": len(task3_rows) if "task3" in selected_tasks else 0,
            "tasks": sorted(selected_tasks),
            "gene_filter_count": len(allowed_genes) if allowed_genes is not None else None,
            "gene_registry": args.gene_registry,
            "gene_list": args.gene_list,
            "support_status": args.support_status,
            "tiers": args.tiers,
            "topk": args.topk,
            "router_threshold": args.router_threshold,
            "context_threshold": args.context_threshold,
            "expert_max_input_tokens": args.expert_max_input_tokens,
            "expert_max_new_tokens": args.expert_max_new_tokens,
            "expert_temperature": args.expert_temperature,
            "expert_top_p": args.expert_top_p,
            "citation_mode": args.citation_mode,
            "faithfulness_mode": args.faithfulness_mode,
            "ar_mode": args.ar_mode,
        },
        "task2": task2_summary,
        "task3": task3_summary,
        "paper_alignment_note": (
            "These are approximate local implementations of AD-GPT-style metrics. "
            "CSR/CVR depend on explicit detectable citations, AR uses local embedding similarity, "
            "and Faithfulness is grounded against retrieved contexts rather than human adjudication. "
            f"Citation mode for this run: {args.citation_mode}. "
            f"Faithfulness mode for this run: {args.faithfulness_mode}. "
            f"AR mode for this run: {args.ar_mode}."
        ),
    }

    write_json(args.out_dir / "paper_style_summary.json", summary)
    if "task2" in selected_tasks:
        write_jsonl(args.out_dir / "task2_predictions.jsonl", task2_predictions)
    if "task3" in selected_tasks:
        write_jsonl(args.out_dir / "task3_predictions.jsonl", task3_predictions)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
