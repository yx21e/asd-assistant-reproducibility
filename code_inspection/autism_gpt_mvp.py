#!/usr/bin/env python3
"""
Autism GPT MVP pipeline aligned to revised AD-GPT style (3-task architecture).

Subcommands:
  1) prepare-data  -> build task-specific corpora + synthetic router data
  2) build-index   -> build 3 task-specific FAISS indices
  3) train-router  -> fine-tune a lightweight BERT-family router
  4) run-query     -> route + retrieve + verifier + abstain
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

from gene_registry import normalize_gene_symbol, resolve_gene_filter
from task3_topic_rules import (
    count_anchor_hits,
    has_autism_signal,
    has_review_signal,
    infer_topic_from_question,
    infer_task3_topic,
    is_probable_non_autism_record,
    topic_anchor_terms,
    wants_recent_evidence,
    wants_review_style,
)


TASK_LABELS = {
    "task1": 0,  # gene information retrieval
    "task2": 1,  # association study reasoning
    "task3": 2,  # broader autism knowledge
    "reject": 3,
}
ID_TO_TASK = {v: k for k, v in TASK_LABELS.items()}

AUTISM_TERMS = (
    "autism spectrum disorder",
    "autistic disorder",
    "autism",
    "autistic",
    "asd",
)

TASK3_BROAD_TOPICS = (
    "general_autism",
    "clinical_features",
    "diagnosis_screening",
    "risk_pathogenesis",
    "epidemiology_population",
    "biomarkers",
    "methods_study_design",
)

TASK3_TOPIC_QUERY_LABELS = {
    "general_autism": "broad autism overview",
    "clinical_features": "clinical features and symptom profile",
    "diagnosis_screening": "screening and diagnostic practice",
    "risk_pathogenesis": "risk factors and pathogenesis",
    "epidemiology_population": "epidemiology and population patterns",
    "biomarkers": "biomarkers and neurobiology",
    "methods_study_design": "study design and research methods",
}

TASK3_TOPIC_GRAPH = {
    "general_autism": tuple(topic for topic in TASK3_BROAD_TOPICS if topic != "general_autism"),
    "clinical_features": ("diagnosis_screening", "risk_pathogenesis", "general_autism"),
    "diagnosis_screening": ("clinical_features", "methods_study_design", "general_autism"),
    "risk_pathogenesis": ("biomarkers", "clinical_features", "general_autism"),
    "epidemiology_population": ("general_autism", "diagnosis_screening"),
    "biomarkers": ("risk_pathogenesis", "methods_study_design", "general_autism"),
    "methods_study_design": ("diagnosis_screening", "biomarkers", "general_autism"),
}


def _env_flag(name: str) -> bool:
    value = str(os.getenv(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _safe_read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _print_json(obj, ensure_ascii: bool = False) -> None:
    payload = json.dumps(obj, ensure_ascii=ensure_ascii, indent=2)
    sys.stdout.buffer.write(payload.encode("utf-8", errors="replace") + b"\n")


def _repair_mojibake(s: str) -> str:
    text = str(s or "")
    if not text:
        return ""
    if not any(marker in text for marker in ("â", "Ã", "Î", "�")):
        return text
    try:
        repaired = text.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return text
    return repaired or text


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", _repair_mojibake(s)).strip()


def _pick_first_notna(series: pd.Series):
    for x in series:
        if pd.notna(x):
            return x
    return None


def _hash_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()[:12]


def _extract_gene_symbols(text: str) -> List[str]:
    # heuristic symbol extractor for routing support
    toks = re.findall(r"\b[A-Za-z0-9-]{2,15}\b", text or "")
    out = []
    skip_exact = {
        "ASD",
        "AUTISM",
        "GENE",
        "GENES",
        "GRCH37",
        "GRCH38",
        "PMID",
        "QTL",
        "SQTL",
        "EQTL",
        "T2T",
    }
    for t in toks:
        upper = t.upper()
        if upper in skip_exact:
            continue
        if re.fullmatch(r"CHR[0-9XYM]+", upper):
            continue
        if re.fullmatch(r"B\d+", upper):
            continue
        upper_ratio = sum(c.isupper() for c in t) / max(1, len(t))
        if upper_ratio >= 0.6 or any(c.isdigit() for c in t):
            out.append(upper)
    # keep order unique
    seen = set()
    uniq = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def _query_focus_gene(text: str) -> Optional[str]:
    raw = str(text or "")
    bad_candidates = {"THE", "THIS", "THAT", "THESE", "THOSE", "GIVEN", "YOUR", "HUMAN"}
    patterns = [
        r"\bfor\s+([A-Za-z0-9-]{2,15})\b",
        r"\bgene\s+([A-Za-z0-9-]{2,15})\b",
        r"\bis\s+([A-Za-z0-9-]{2,15})\s+associated\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match:
            candidate = match.group(1).upper()
            if candidate in bad_candidates:
                continue
            genes = _extract_gene_symbols(candidate)
            if genes:
                return genes[0]
    genes = _extract_gene_symbols(raw)
    for gene in genes:
        if gene not in bad_candidates:
            return gene
    return None


@dataclass
class AutismAssets:
    gene_context: pd.DataFrame
    gene_function: pd.DataFrame
    combined: pd.DataFrame
    coords: Optional[pd.DataFrame]
    gene_list: List[str]
    pubmed: Optional[pd.DataFrame]
    pubmed_source: Optional[str] = None


def _parse_literature_entry(text: str) -> Optional[dict]:
    text = _norm_space(text)
    if not text:
        return None

    pmid = None
    prefix_match = re.search(r"\{PMID:\s*([0-9]+)\}\s*", text, flags=re.IGNORECASE)
    if prefix_match:
        pmid = prefix_match.group(1)
        text = _norm_space(text.replace(prefix_match.group(0), "", 1))

    suffix_match = re.search(r"\(PMID:\s*([0-9]+)\)\s*$", text, flags=re.IGNORECASE)
    if suffix_match:
        pmid = pmid or suffix_match.group(1)
        text = _norm_space(text[: suffix_match.start()])

    if not text:
        return None

    return {
        "pmid": pmid or f"na_{_hash_text(text)}",
        "title": "",
        "abstract": text,
        "keywords": "",
        "year": None,
        "month": None,
    }


def _load_pubmed_parquet_dir(pubmed_dir: Path, pubmed_max_rows: int) -> Optional[pd.DataFrame]:
    files = sorted(pubmed_dir.rglob("*.parquet"))
    if not files:
        return None

    parts = []
    count = 0
    for fp in files:
        df = pd.read_parquet(fp)
        parts.append(df)
        count += len(df)
        if count >= pubmed_max_rows:
            break

    pubmed_df = pd.concat(parts, ignore_index=True)
    if len(pubmed_df) > pubmed_max_rows:
        pubmed_df = pubmed_df.iloc[:pubmed_max_rows].copy()
    return pubmed_df


def _load_gene_context_json(json_path: Path, gene_filter: set[str], pubmed_max_rows: int) -> Optional[pd.DataFrame]:
    raw = _safe_read_json(json_path)
    if not isinstance(raw, list):
        return None

    rows = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        gene = _norm_space(item.get("gene_name", "")).upper()
        if gene_filter and gene and gene not in gene_filter:
            continue

        contexts = item.get("context", [])
        if isinstance(contexts, str):
            contexts = [contexts]
        if not isinstance(contexts, list):
            continue

        for ctx in contexts:
            parsed = _parse_literature_entry(str(ctx))
            if parsed is None:
                continue
            parsed["keywords"] = gene
            rows.append(parsed)
            if len(rows) >= pubmed_max_rows:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows) if rows else None


def _load_gene_txt_dir(txt_dir: Path, gene_filter: set[str], pubmed_max_rows: int) -> Optional[pd.DataFrame]:
    rows = []
    for fp in sorted(txt_dir.glob("*.txt")):
        stem_lower = fp.stem.lower()
        if stem_lower.endswith("_articles"):
            continue
        if stem_lower.startswith("autism_genes_analysis"):
            continue
        gene = fp.stem.strip().upper()
        if gene_filter and gene not in gene_filter:
            continue

        content = fp.read_text(encoding="utf-8", errors="replace")
        for block in re.split(r"(?:\r?\n\s*){2,}", content):
            parsed = _parse_literature_entry(block)
            if parsed is None:
                continue
            parsed["keywords"] = gene
            rows.append(parsed)
            if len(rows) >= pubmed_max_rows:
                return pd.DataFrame(rows)

    return pd.DataFrame(rows) if rows else None


def _load_literature_records(
    root: Path,
    pubmed_dir: Optional[Path],
    gene_list: Sequence[str],
    pubmed_max_rows: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    gene_filter = {g.strip().upper() for g in gene_list if str(g).strip()}
    alt_extract_dir = root.parent / "Autism" / "AD_GPT_DATA" / "AD_GPT_DATA" / "abstract_extract"

    candidates: List[Path] = []
    if pubmed_dir is not None:
        candidates.append(pubmed_dir.resolve())
    default_year_month = root / "year_month"
    if default_year_month.exists():
        candidates.append(default_year_month.resolve())
    if alt_extract_dir.exists():
        candidates.append(alt_extract_dir.resolve())

    seen = set()
    unique_candidates = []
    for cand in candidates:
        key = str(cand)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(cand)

    for cand in unique_candidates:
        if cand.is_file():
            if cand.suffix.lower() == ".parquet":
                df = pd.read_parquet(cand)
                if len(df) > pubmed_max_rows:
                    df = df.iloc[:pubmed_max_rows].copy()
                return df, str(cand)
            if cand.suffix.lower() == ".json":
                df = _load_gene_context_json(cand, gene_filter=gene_filter, pubmed_max_rows=pubmed_max_rows)
                if df is not None and not df.empty:
                    return df, str(cand)
            continue

        if not cand.is_dir():
            continue

        df = _load_pubmed_parquet_dir(cand, pubmed_max_rows=pubmed_max_rows)
        if df is not None and not df.empty:
            return df, str(cand)

        for json_name in ("autism_genes_analysis_all.json", "autism_genes_analysis.json"):
            json_path = cand / json_name
            if json_path.exists():
                df = _load_gene_context_json(json_path, gene_filter=gene_filter, pubmed_max_rows=pubmed_max_rows)
                if df is not None and not df.empty:
                    return df, str(json_path)

        df = _load_gene_txt_dir(cand, gene_filter=gene_filter, pubmed_max_rows=pubmed_max_rows)
        if df is not None and not df.empty:
            return df, str(cand)

    return None, None


def load_assets(root: Path, pubmed_dir: Optional[Path] = None, pubmed_max_rows: int = 200_000) -> AutismAssets:
    root = root.resolve()
    required = {
        "gene_context": root / "gene_context_prose.csv",
        "gene_function": root / "gene_with_function.json",
        "combined": root / "combined.csv",
        "gene_list": root / "Autism_gene_list.txt",
    }
    for k, p in required.items():
        if not p.exists():
            raise FileNotFoundError(f"Missing required file for {k}: {p}")

    coords_path = root / "combined_updated_coords.csv"
    coords_df = pd.read_csv(coords_path) if coords_path.exists() else None

    gc = pd.read_csv(required["gene_context"])
    gf_raw = _safe_read_json(required["gene_function"])
    gf = pd.json_normalize(gf_raw)
    # normalize columns
    rename_map = {}
    if "Gene symbol" in gf.columns:
        rename_map["Gene symbol"] = "gene_name"
    if "MIM Number" in gf.columns:
        rename_map["MIM Number"] = "mim_number"
    gf = gf.rename(columns=rename_map)

    combined = pd.read_csv(required["combined"])
    gene_list = [line.strip() for line in required["gene_list"].read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]

    pubmed_df, pubmed_source = _load_literature_records(
        root=root,
        pubmed_dir=pubmed_dir,
        gene_list=gene_list,
        pubmed_max_rows=pubmed_max_rows,
    )

    return AutismAssets(
        gene_context=gc,
        gene_function=gf,
        combined=combined,
        coords=coords_df,
        gene_list=gene_list,
        pubmed=pubmed_df,
        pubmed_source=pubmed_source,
    )


def aggregate_gene_stats(combined: pd.DataFrame) -> pd.DataFrame:
    needed = {"Gene Symbol", "Region", "Source", "Variant ID", "P-value Nominal"}
    missing = needed - set(combined.columns)
    if missing:
        raise ValueError(f"combined.csv missing columns: {sorted(missing)}")

    rows = []
    for gene, g in combined.groupby("Gene Symbol", dropna=True):
        g = g.copy()
        g["Region"] = g["Region"].astype(str)
        g["Source"] = g["Source"].astype(str)
        eq = g[g["Source"].str.lower().eq("eqtl")]
        sq = g[g["Source"].str.lower().eq("sqtl")]
        eq_regions = sorted(set(eq["Region"].tolist()))
        sq_regions = sorted(set(sq["Region"].tolist()))
        eq_min_p = float(eq["P-value Nominal"].min()) if not eq.empty else None
        sq_min_p = float(sq["P-value Nominal"].min()) if not sq.empty else None
        rows.append(
            {
                "gene_name": str(gene),
                "eqtl_regions": eq_regions,
                "sqtl_regions": sq_regions,
                "eqtl_variants": sorted(set(eq["Variant ID"].astype(str).tolist())),
                "sqtl_variants": sorted(set(sq["Variant ID"].astype(str).tolist())),
                "eqtl_min_p": eq_min_p,
                "sqtl_min_p": sq_min_p,
            }
        )
    return pd.DataFrame(rows)


def build_task1_docs(assets: AutismAssets, stats_df: pd.DataFrame) -> List[dict]:
    gc = assets.gene_context.copy()
    if "gene_name" not in gc.columns:
        # fallback for older files
        if "Gene Symbol" in gc.columns:
            gc = gc.rename(columns={"Gene Symbol": "gene_name"})
        else:
            raise ValueError("gene_context_prose.csv missing gene_name column.")

    gf = assets.gene_function.copy()
    if "gene_name" not in gf.columns:
        raise ValueError("gene_with_function.json normalized frame missing gene_name.")

    merged = gc.merge(gf, on="gene_name", how="outer")
    merged = merged.merge(stats_df, on="gene_name", how="left")

    if assets.coords is not None and "Gene Symbol" in assets.coords.columns:
        coord = assets.coords.groupby("Gene Symbol", dropna=True).agg(
            {
                "GRCh37_start": _pick_first_notna,
                "GRCh37_end": _pick_first_notna,
                "GRCh38_start": _pick_first_notna,
                "GRCh38_end": _pick_first_notna,
                "T2T_start": _pick_first_notna,
                "T2T_end": _pick_first_notna,
            }
        ).reset_index().rename(columns={"Gene Symbol": "gene_name"})
        merged = merged.merge(coord, on="gene_name", how="left")

    docs = []
    for _, r in merged.iterrows():
        gene = str(r.get("gene_name", "")).strip()
        if not gene:
            continue

        context = _norm_space(str(r.get("context", "")))
        omim = _norm_space(str(r.get("Gene_Function_OMIM", "") if pd.notna(r.get("Gene_Function_OMIM")) else ""))
        eq_regions = r.get("eqtl_regions", []) if isinstance(r.get("eqtl_regions", []), list) else []
        sq_regions = r.get("sqtl_regions", []) if isinstance(r.get("sqtl_regions", []), list) else []
        eq_min_p = r.get("eqtl_min_p", None)
        sq_min_p = r.get("sqtl_min_p", None)

        coord_bits = []
        for asm in ("GRCh37", "GRCh38", "T2T"):
            s = r.get(f"{asm}_start", None)
            e = r.get(f"{asm}_end", None)
            if pd.notna(s) and pd.notna(e):
                coord_bits.append(f"{asm}:{int(float(s))}-{int(float(e))}")
        coord_text = "; ".join(coord_bits) if coord_bits else "Not specified"

        text = (
            f"Gene: {gene}. "
            f"Coordinates: {coord_text}. "
            f"eQTL regions: {', '.join(eq_regions) if eq_regions else 'None'}. "
            f"sQTL regions: {', '.join(sq_regions) if sq_regions else 'None'}. "
            f"Best eQTL p-value: {eq_min_p if eq_min_p is not None else 'N/A'}. "
            f"Best sQTL p-value: {sq_min_p if sq_min_p is not None else 'N/A'}. "
            f"Gene context: {context if context else 'N/A'}. "
            f"OMIM function: {omim if omim else 'N/A'}."
        )

        docs.append(
            {
                "doc_id": f"task1_{gene}",
                "task": "task1",
                "gene_name": gene,
                "text": text,
                "meta": {
                    "gene_name": gene,
                    "eqtl_regions": eq_regions,
                    "sqtl_regions": sq_regions,
                },
            }
        )
    return docs


def _meta_text_value(value) -> str:
    if isinstance(value, list):
        return _norm_space("; ".join(str(x) for x in value if str(x).strip()))
    return _norm_space(str(value or ""))


def _gene_mentioned_in_text(text: str, gene: str) -> bool:
    raw = str(text or "")
    symbol = str(gene or "").strip()
    if not raw or not symbol:
        return False
    if len(symbol) <= 4:
        pattern = rf"(?<![A-Za-z0-9-]){re.escape(symbol)}(?![A-Za-z0-9-])"
        return re.search(pattern, raw) is not None
    pattern = rf"(?<![A-Za-z0-9-]){re.escape(symbol)}(?![A-Za-z0-9-])"
    return re.search(pattern, raw, flags=re.IGNORECASE) is not None


def _compile_gene_scan_pattern(gene_symbols: Sequence[str]) -> Optional[re.Pattern]:
    genes = [re.escape(str(gene)) for gene in sorted(set(gene_symbols), key=lambda item: (-len(item), item))]
    if not genes:
        return None
    pattern = r"(?<![A-Za-z0-9-])(" + "|".join(genes) + r")(?![A-Za-z0-9-])"
    return re.compile(pattern, flags=re.IGNORECASE)


def _scan_gene_mentions(text: str, gene_pattern: Optional[re.Pattern]) -> List[str]:
    raw = str(text or "")
    if not raw or gene_pattern is None:
        return []
    seen = set()
    out = []
    for match in gene_pattern.finditer(raw):
        gene = normalize_gene_symbol(match.group(1))
        if gene and gene not in seen:
            seen.add(gene)
            out.append(gene)
    return out


def _task2_literature_bonus(
    title: str,
    abstract: str,
    keywords: str,
    author_keywords: str,
    mesh_terms: str,
    publication_type: str,
) -> Tuple[float, str]:
    title_text = _norm_space(title)
    abstract_text = _norm_space(abstract)
    keyword_text = _norm_space(" ".join(x for x in (keywords, author_keywords, mesh_terms) if x))
    combined = _norm_space(" ".join([title_text, abstract_text, keyword_text, publication_type]))

    autism_title = bool(re.search(r"\bautism\b|\bautistic\b|\bautism spectrum disorder\b", title_text, flags=re.IGNORECASE))
    autism_body = has_autism_signal(
        title=title_text,
        abstract=abstract_text,
        keywords=keyword_text,
        publication_type=publication_type,
    )
    association_signal = bool(
        re.search(
            r"\bassociation\b|\bassociated\b|\brisk\b|\bvariant\b|\bmutation\b|\bcase-control\b|\bcohort\b|\bde novo\b|\bhaploinsufficiency\b|\bknockdown\b|\bgenetic\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    phenotype_signal = bool(
        re.search(
            r"\bdevelopmental delay\b|\bintellectual disability\b|\bneurodevelopmental disorder\b|\bsociability deficits\b|\brepetitive behaviors\b|\bbehavioral\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    review_signal = has_review_signal(
        publication_type=publication_type,
        title=title_text,
        keywords=keyword_text,
    )
    keyword_autism = has_autism_signal(
        title="",
        abstract="",
        keywords=keyword_text,
        publication_type="",
    )
    keyword_association = bool(
        re.search(
            r"\bassociation\b|\bassociated\b|\brisk factors?\b|\bgenetic predisposition\b|\bwhole[- ]exome sequencing\b|\bexome sequencing\b|\bde novo\b|\bvariant\b|\bmutation\b|\bcopy number variation\b|\bcohort\b",
            keyword_text,
            flags=re.IGNORECASE,
        )
    )

    bonus = 0.0
    tier = "generic"
    if autism_title and association_signal:
        bonus += 2.6
        tier = "direct_autism_association"
    elif autism_title or (autism_body and association_signal):
        bonus += 1.9
        tier = "autism_association"
    elif autism_body and phenotype_signal:
        bonus += 1.2
        tier = "autism_phenotype"
    elif autism_body:
        bonus += 0.8
        tier = "autism_generic"

    if keyword_autism and keyword_association:
        bonus += 0.28
    elif keyword_autism:
        bonus += 0.12

    if review_signal and tier in {"generic", "autism_generic"}:
        bonus -= 0.15
    return bonus, tier


def _task2_literature_match_score(
    gene: str,
    title: str,
    abstract: str,
    keywords: str,
    author_keywords: str,
    mesh_terms: str,
    publication_type: str,
    year,
) -> float:
    if not any(
        _gene_mentioned_in_text(text, gene)
        for text in (title, abstract, keywords, author_keywords, mesh_terms)
    ):
        return 0.0

    score = 0.0
    if _gene_mentioned_in_text(title, gene):
        score += 3.0
    if _gene_mentioned_in_text(keywords, gene):
        score += 1.5
    if _gene_mentioned_in_text(author_keywords, gene):
        score += 1.25
    if _gene_mentioned_in_text(mesh_terms, gene):
        score += 1.0
    if _gene_mentioned_in_text(abstract, gene):
        score += 1.0
    direct_bonus, _ = _task2_literature_bonus(
        title=title,
        abstract=abstract,
        keywords=keywords,
        author_keywords=author_keywords,
        mesh_terms=mesh_terms,
        publication_type=publication_type,
    )
    score += direct_bonus
    if has_review_signal(publication_type=publication_type, title=title, keywords=keywords):
        score += 0.2
    try:
        year_val = int(year)
    except (TypeError, ValueError):
        year_val = None
    if year_val is not None:
        score += max(0.0, min((year_val - 2000) / 30.0, 0.5))
    return score


def build_task2_docs(assets: AutismAssets, stats_df: pd.DataFrame, max_lit_docs_per_gene: int = 4) -> List[dict]:
    # association reasoning docs (gene-region-variant evidence)
    gf = assets.gene_function.copy()
    omim_map = {}
    if "gene_name" in gf.columns and "Gene_Function_OMIM" in gf.columns:
        for _, r in gf.iterrows():
            g = str(r["gene_name"]).strip()
            if g:
                txt = r.get("Gene_Function_OMIM", None)
                if pd.notna(txt):
                    omim_map[g] = _norm_space(str(txt))

    summary_docs = []
    for _, r in stats_df.iterrows():
        gene = str(r["gene_name"])
        eq_regions = r["eqtl_regions"] if isinstance(r["eqtl_regions"], list) else []
        sq_regions = r["sqtl_regions"] if isinstance(r["sqtl_regions"], list) else []
        eq_vars = r["eqtl_variants"] if isinstance(r["eqtl_variants"], list) else []
        sq_vars = r["sqtl_variants"] if isinstance(r["sqtl_variants"], list) else []
        text = (
            f"Association evidence for autism candidate gene {gene}. "
            f"eQTL evidence regions: {', '.join(eq_regions) if eq_regions else 'None'}. "
            f"sQTL evidence regions: {', '.join(sq_regions) if sq_regions else 'None'}. "
            f"Top eQTL variants: {', '.join(eq_vars[:10]) if eq_vars else 'None'}. "
            f"Top sQTL variants: {', '.join(sq_vars[:10]) if sq_vars else 'None'}. "
            f"Statistical minima: eQTL p={r.get('eqtl_min_p', 'N/A')}, sQTL p={r.get('sqtl_min_p', 'N/A')}. "
            f"OMIM note: {omim_map.get(gene, 'N/A')}."
        )
        summary_docs.append(
            {
                "doc_id": f"task2_summary_{gene}",
                "task": "task2",
                "gene_name": gene,
                "text": text,
                "meta": {"gene_name": gene, "doc_type": "summary"},
            }
        )

    literature_docs: List[dict] = []
    if assets.pubmed is not None and not assets.pubmed.empty and max_lit_docs_per_gene > 0:
        gene_set = {
            normalize_gene_symbol(gene)
            for gene in stats_df["gene_name"].astype(str).tolist()
            if normalize_gene_symbol(gene)
        }
        gene_pattern = _compile_gene_scan_pattern(sorted(gene_set))
        by_gene: Dict[str, List[dict]] = {gene: [] for gene in gene_set}

        for _, r in assets.pubmed.iterrows():
            title = _norm_space(str(r.get("title", "")))
            abstract = _norm_space(str(r.get("abstract", "")))
            keywords = _meta_text_value(r.get("keywords", ""))
            author_keywords = _meta_text_value(r.get("author_keywords", ""))
            mesh_terms = _meta_text_value(r.get("mesh_terms", ""))
            publication_type = _meta_text_value(r.get("publication_type", ""))
            if not has_autism_signal(
                title=title,
                abstract=abstract,
                keywords=" ".join(x for x in (keywords, author_keywords, mesh_terms) if x),
                publication_type=publication_type,
            ):
                continue
            if is_probable_non_autism_record(
                title=title,
                abstract=abstract,
                keywords=" ".join(x for x in (keywords, author_keywords, mesh_terms) if x),
                publication_type=publication_type,
            ):
                continue

            raw_for_extract = " ".join([title, abstract, keywords, author_keywords, mesh_terms])
            candidate_genes = set(_scan_gene_mentions(raw_for_extract, gene_pattern))
            candidate_genes.update(
                symbol
                for symbol in _extract_gene_symbols(raw_for_extract)
                if symbol in gene_set
            )
            if not candidate_genes:
                continue

            pmid = _norm_space(str(r.get("pmid", "")))
            journal = _norm_space(str(r.get("journal", "")))
            year = r.get("year", None)
            month = r.get("month", None)
            text = _pubmed_text(r)
            if not text:
                continue
            direct_bonus, evidence_tier = _task2_literature_bonus(
                title=title,
                abstract=abstract,
                keywords=keywords,
                author_keywords=author_keywords,
                mesh_terms=mesh_terms,
                publication_type=publication_type,
            )

            for gene in sorted(candidate_genes):
                score = _task2_literature_match_score(
                    gene=gene,
                    title=title,
                    abstract=abstract,
                    keywords=keywords,
                    author_keywords=author_keywords,
                    mesh_terms=mesh_terms,
                    publication_type=publication_type,
                    year=year,
                )
                if score <= 0:
                    continue
                by_gene.setdefault(gene, []).append(
                    {
                        "pmid": pmid,
                        "gene_name": gene,
                        "title": title,
                        "journal": journal,
                        "year": year,
                        "month": month,
                        "text": text,
                        "keywords": keywords,
                        "author_keywords": author_keywords,
                        "mesh_terms": mesh_terms,
                        "publication_type": publication_type,
                        "evidence_tier": evidence_tier,
                        "autism_direct_bonus": direct_bonus,
                        "match_score": score,
                    }
                )

        for gene in sorted(by_gene):
            seen_pmids = set()
            ranked = sorted(
                by_gene[gene],
                key=lambda row: (
                    row.get("match_score", 0.0),
                    row.get("year") if row.get("year") is not None else -1,
                    row.get("pmid", ""),
                ),
                reverse=True,
            )
            kept = 0
            for row in ranked:
                pmid = row.get("pmid") or f"na_{_hash_text(row.get('text', ''))}"
                if pmid in seen_pmids:
                    continue
                seen_pmids.add(pmid)
                literature_docs.append(
                    {
                        "doc_id": f"task2_lit_{gene}_{pmid}",
                        "task": "task2",
                        "gene_name": gene,
                        "text": row["text"],
                        "meta": {
                            "gene_name": gene,
                            "doc_type": "literature",
                            "pmid": pmid,
                            "title": row.get("title"),
                            "journal": row.get("journal"),
                            "year": row.get("year"),
                            "month": row.get("month"),
                            "keywords": row.get("keywords"),
                            "author_keywords": row.get("author_keywords"),
                            "mesh_terms": row.get("mesh_terms"),
                            "publication_type": row.get("publication_type"),
                            "evidence_tier": row.get("evidence_tier"),
                            "autism_direct_bonus": row.get("autism_direct_bonus"),
                            "match_score": row.get("match_score"),
                        },
                    }
                )
                kept += 1
                if kept >= max_lit_docs_per_gene:
                    break

    return summary_docs + literature_docs


def _filter_gene_docs(docs: Sequence[dict], allowed_genes: Optional[set[str]]) -> List[dict]:
    if allowed_genes is None:
        return list(docs)
    return [doc for doc in docs if normalize_gene_symbol(doc.get("gene_name")) in allowed_genes]


def _pubmed_text(row: pd.Series) -> str:
    title = _norm_space(str(row.get("title", "")))
    abstract = _norm_space(str(row.get("abstract", "")))
    keywords = row.get("keywords", "")
    if isinstance(keywords, list):
        kw = ", ".join([str(x) for x in keywords[:20]])
    else:
        kw = _norm_space(str(keywords))
    author_keywords = _norm_space(str(row.get("author_keywords", "")))
    mesh_terms = _norm_space(str(row.get("mesh_terms", "")))
    supplement = _norm_space(str(row.get("supplement_excerpt") or row.get("supplement_text", "")))
    pub_types = _norm_space(str(row.get("publication_type", "")))
    text = f"Title: {title}. Abstract: {abstract}. Keywords: {kw}."
    if author_keywords:
        text += f" Author keywords: {author_keywords}."
    if mesh_terms:
        text += f" MeSH terms: {mesh_terms}."
    if pub_types:
        text += f" Publication types: {pub_types}."
    if supplement:
        text += f" Supplementary: {supplement}."
    return _norm_space(text)


def build_task3_docs(assets: AutismAssets, fallback_docs: List[dict], max_docs: int = 120_000) -> List[dict]:
    # task3 is broad autism knowledge from literature
    if assets.pubmed is None or assets.pubmed.empty:
        # fallback: derive from task2 docs so pipeline can still run end-to-end
        out = []
        for i, d in enumerate(fallback_docs[:3000]):
            out.append(
                {
                    "doc_id": f"task3_fb_{i}",
                    "task": "task3",
                    "text": d["text"],
                    "meta": {"fallback": True, "source_doc": d.get("doc_id")},
                }
            )
        return out

    # mild deduplication by title+abstract hash
    seen = set()
    out = []
    for _, r in assets.pubmed.iterrows():
        title = r.get("title", "")
        abstract = r.get("abstract", "")
        keywords = r.get("keywords", "")
        publication_type = r.get("publication_type", "")
        if not has_autism_signal(
            title=title,
            abstract=abstract,
            keywords=keywords,
            publication_type=publication_type,
        ):
            continue
        if is_probable_non_autism_record(
            title=title,
            abstract=abstract,
            keywords=keywords,
            publication_type=publication_type,
        ):
            continue
        text = _pubmed_text(r)
        if not text:
            continue
        hid = _hash_text(text)
        if hid in seen:
            continue
        seen.add(hid)
        pmid = str(r.get("pmid", "")).strip() or f"na_{hid}"
        existing_topic = _norm_space(str(r.get("topic", "")))
        topic = existing_topic if existing_topic and existing_topic != "general_autism" else infer_task3_topic(
            title=title,
            abstract=abstract,
            keywords=keywords,
            publication_type=publication_type,
        )
        out.append(
            {
                "doc_id": f"task3_{pmid}",
                "task": "task3",
                "text": text,
                "meta": {
                    "pmid": pmid,
                    "year": r.get("year", None),
                    "month": r.get("month", None),
                    "title": r.get("title", None),
                    "journal": r.get("journal", None),
                    "topic": topic or "general_autism",
                    "keywords": keywords or None,
                    "author_keywords": r.get("author_keywords", None),
                    "mesh_terms": r.get("mesh_terms", None),
                    "publication_type": publication_type or None,
                },
            }
        )
        if len(out) >= max_docs:
            break
    return out


def build_router_samples(gene_symbols: Sequence[str], seed: int = 42) -> Tuple[List[dict], List[dict]]:
    rng = random.Random(seed)
    genes = list(gene_symbols)
    if not genes:
        genes = ["CHD8", "CNTNAP2", "SCN2A", "ADNP", "SHANK3"]

    t1_templates = [
        "What chromosome is gene {g} on?",
        "Give me GRCh38 coordinates for {g}.",
        "List brain regions where {g} has eQTL evidence.",
        "What is the OMIM gene function summary for {g}?",
    ]
    t2_templates = [
        "Based on evidence, is {g} associated with autism? Summarize statistical support.",
        "Evaluate association study findings for {g} in autism.",
        "Explain mechanism and brain-region context for {g} autism association.",
    ]
    t3_templates = [
        "What are early clinical signs of autism spectrum disorder?",
        "Summarize major autism biomarkers discussed in recent literature.",
        "What are common study designs in autism risk research?",
        "How do sex differences appear in autism epidemiology?",
    ]
    reject_templates = [
        "How do I cook pasta?",
        "Write a poem about summer rain.",
        "Explain NBA salary cap in detail.",
    ]

    def gen_rows(task_key: str, templates: Sequence[str], n: int) -> List[dict]:
        rows = []
        for _ in range(n):
            tpl = rng.choice(templates)
            if "{g}" in tpl:
                q = tpl.format(g=rng.choice(genes))
            else:
                q = tpl
            rows.append({"text": q, "label": TASK_LABELS[task_key], "task": task_key})
        return rows

    train = []
    train += gen_rows("task1", t1_templates, 800)
    train += gen_rows("task2", t2_templates, 800)
    train += gen_rows("task3", t3_templates, 800)
    train += gen_rows("reject", reject_templates, 350)
    rng.shuffle(train)

    eval_rows = []
    eval_rows += gen_rows("task1", t1_templates, 120)
    eval_rows += gen_rows("task2", t2_templates, 120)
    eval_rows += gen_rows("task3", t3_templates, 120)
    eval_rows += gen_rows("reject", reject_templates, 60)
    rng.shuffle(eval_rows)
    return train, eval_rows


def cmd_prepare_data(args):
    root = Path(args.root)
    pubmed_dir = Path(args.pubmed_dir) if args.pubmed_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    assets = load_assets(root, pubmed_dir=pubmed_dir, pubmed_max_rows=args.pubmed_max_rows)
    stats_df = aggregate_gene_stats(assets.combined)

    allowed_genes = resolve_gene_filter(
        gene_registry=args.gene_registry,
        gene_list=args.gene_list,
        support_statuses=args.support_status,
        tiers=args.tiers,
    )

    task1_docs_all = build_task1_docs(assets, stats_df)
    task2_docs_all = build_task2_docs(
        assets,
        stats_df,
        max_lit_docs_per_gene=args.task2_max_lit_docs_per_gene,
    )
    task1_docs = _filter_gene_docs(task1_docs_all, allowed_genes)
    task2_docs = _filter_gene_docs(task2_docs_all, allowed_genes)
    task2_summary_docs = [
        doc for doc in task2_docs
        if str(doc.get("meta", {}).get("doc_type") or "summary") == "summary"
    ]
    task3_docs = build_task3_docs(
        assets,
        fallback_docs=task2_summary_docs,
        max_docs=args.task3_max_docs,
    )

    _write_jsonl(out_dir / "task1_docs.jsonl", task1_docs)
    _write_jsonl(out_dir / "task2_docs.jsonl", task2_docs)
    _write_jsonl(out_dir / "task3_docs.jsonl", task3_docs)

    train_rows, eval_rows = build_router_samples([d.get("gene_name", "") for d in task1_docs if d.get("gene_name")])
    _write_jsonl(out_dir / "router_train.jsonl", train_rows)
    _write_jsonl(out_dir / "router_eval.jsonl", eval_rows)

    report = {
        "root": str(root),
        "pubmed_loaded_rows": int(len(assets.pubmed)) if assets.pubmed is not None else 0,
        "pubmed_source": assets.pubmed_source,
        "task3_source": "literature" if assets.pubmed is not None and not assets.pubmed.empty else "task2_fallback",
        "gene_filter_count": len(allowed_genes) if allowed_genes is not None else None,
        "gene_registry": args.gene_registry,
        "gene_list": args.gene_list,
        "support_status": args.support_status,
        "tiers": args.tiers,
        "task1_docs_unfiltered": len(task1_docs_all),
        "task2_docs_unfiltered": len(task2_docs_all),
        "task2_summary_docs_unfiltered": sum(
            1
            for doc in task2_docs_all
            if str(doc.get("meta", {}).get("doc_type") or "summary") == "summary"
        ),
        "task2_literature_docs_unfiltered": sum(
            1
            for doc in task2_docs_all
            if str(doc.get("meta", {}).get("doc_type") or "") == "literature"
        ),
        "task1_docs": len(task1_docs),
        "task2_docs": len(task2_docs),
        "task2_summary_docs": len(task2_summary_docs),
        "task2_literature_docs": sum(
            1
            for doc in task2_docs
            if str(doc.get("meta", {}).get("doc_type") or "") == "literature"
        ),
        "task3_docs": len(task3_docs),
        "task3_topic_counts": dict(sorted(Counter(d.get("meta", {}).get("topic", "general_autism") for d in task3_docs).items())),
        "router_train_rows": len(train_rows),
        "router_eval_rows": len(eval_rows),
    }
    _write_json(out_dir / "prepare_report.json", report)
    _print_json(report, ensure_ascii=False)


def _load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model


def _encode_texts(model, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
    embs = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return embs.astype("float32")


def cmd_build_index(args):
    import faiss

    in_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_embedder(args.embed_model)
    summary = {}
    for task in ("task1", "task2", "task3"):
        rows = _read_jsonl(in_dir / f"{task}_docs.jsonl")
        texts = [r["text"] for r in rows]
        metas = [{"doc_id": r.get("doc_id"), "task": r.get("task"), "meta": r.get("meta", {})} for r in rows]
        embs = _encode_texts(model, texts, batch_size=args.embed_batch_size)
        if embs.size == 0:
            dim = model.get_sentence_embedding_dimension()
            index = faiss.IndexFlatIP(dim)
        else:
            dim = embs.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(embs)

        faiss.write_index(index, str(out_dir / f"{task}.faiss"))
        _write_jsonl(out_dir / f"{task}_meta.jsonl", [{"text": t, **m} for t, m in zip(texts, metas)])
        summary[task] = {"docs": len(rows), "dim": int(dim)}

    _write_json(
        out_dir / "index_config.json",
        {
            "embed_model": args.embed_model,
            "tasks": summary,
        },
    )
    _print_json(summary, ensure_ascii=False)


def _read_router_rows(path: Path) -> pd.DataFrame:
    rows = _read_jsonl(path)
    df = pd.DataFrame(rows)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"{path} must contain text and label columns.")
    return df


def cmd_train_router(args):
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    import evaluate

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = _read_router_rows(data_dir / "router_train.jsonl")
    eval_df = _read_router_rows(data_dir / "router_eval.jsonl")

    tokenizer = AutoTokenizer.from_pretrained(args.router_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.router_model,
        num_labels=len(TASK_LABELS),
        id2label=ID_TO_TASK,
        label2id=TASK_LABELS,
    )

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    train_ds = Dataset.from_pandas(train_df[["text", "label"]], preserve_index=False).map(tok_fn, batched=True)
    eval_ds = Dataset.from_pandas(eval_df[["text", "label"]], preserve_index=False).map(tok_fn, batched=True)

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return acc_metric.compute(predictions=preds, references=labels)

    targs_kwargs = {
        "output_dir": str(out_dir / "checkpoints"),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "accuracy",
        "greater_is_better": True,
        "report_to": "none",
        "seed": args.seed,
    }
    ta_params = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in ta_params:
        targs_kwargs["evaluation_strategy"] = "epoch"
    else:
        targs_kwargs["eval_strategy"] = "epoch"
    targs = TrainingArguments(**targs_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": targs,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "data_collator": collator,
        "compute_metrics": compute_metrics,
    }
    trainer_params = inspect.signature(Trainer.__init__).parameters
    if "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer
    trainer = Trainer(**trainer_kwargs)
    trainer.train()
    metrics = trainer.evaluate()

    model_dir = out_dir / "router_model"
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    _write_json(out_dir / "router_metrics.json", metrics)
    _print_json(metrics, ensure_ascii=False)


class RouterPredictor:
    def __init__(self, model_dir: Path):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        import torch

        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        best_id = int(np.argmax(probs))
        scores = {ID_TO_TASK[i]: float(probs[i]) for i in range(len(probs))}
        return ID_TO_TASK[best_id], float(probs[best_id]), scores


class TaskRetriever:
    def __init__(self, index_dir: Path):
        import faiss

        self.index_dir = index_dir
        self.task1_direct_lookup = _env_flag("AUTISM_GPT_TASK1_DIRECT_LOOKUP")
        self.task2_pheno_rerank = _env_flag("AUTISM_GPT_TASK2_PHENO_RERANK")
        self.task2_keyword_deep = _env_flag("AUTISM_GPT_TASK2_KEYWORD_DEEP1")
        self.task2_runtime = {
            "evidence_priority": _env_flag("AUTISM_GPT_TASK2_EVIDENCE_PRIORITY1"),
        }
        self.task3_keyword_deep = _env_flag("AUTISM_GPT_TASK3_KEYWORD_DEEP1")
        self.task3_runtime = {
            "self_route": _env_flag("AUTISM_GPT_TASK3_SELF_ROUTE"),
            "graph_rag": _env_flag("AUTISM_GPT_TASK3_GRAPH_RAG"),
            "iterative": _env_flag("AUTISM_GPT_TASK3_ITERATIVE"),
            "general_route": _env_flag("AUTISM_GPT_TASK3_GENERAL_ROUTE"),
        }
        self.task3_mode = (_norm_space(os.environ.get("AUTISM_GPT_TASK3_MODE", "paper")).lower() or "paper")
        cfg = _safe_read_json(index_dir / "index_config.json")
        self.embed_model_name = cfg["embed_model"]
        self.embed_model = _load_embedder(self.embed_model_name)
        self.faiss = faiss
        self.indices = {}
        self.metas = {}
        self.task3_doc_lookup = {}
        self.task3_topic_rows = {}
        for task in ("task1", "task2", "task3"):
            self.indices[task] = faiss.read_index(str(index_dir / f"{task}.faiss"))
            self.metas[task] = _read_jsonl(index_dir / f"{task}_meta.jsonl")
        for row in self.metas["task3"]:
            meta = dict(row.get("meta") or {})
            topic = str(meta.get("topic") or infer_task3_topic(
                title=meta.get("title", ""),
                abstract=row.get("text", ""),
                keywords=meta.get("keywords", ""),
                publication_type=meta.get("publication_type", ""),
            ) or "general_autism")
            title = str(meta.get("title") or "")
            keywords = str(meta.get("keywords") or "")
            pub_type = str(meta.get("publication_type") or "")
            search_text = _norm_space(" ".join([title, row.get("text", ""), keywords, pub_type]))
            payload = {
                "doc_id": row.get("doc_id"),
                "task": "task3",
                "text": row.get("text", ""),
                "meta": dict(meta, topic=topic),
                "_topic": topic,
                "_search_text": search_text,
                "_title_text": _norm_space(title).lower(),
                "_review": has_review_signal(publication_type=pub_type, title=title, keywords=keywords),
                "_broad": bool(
                    re.search(
                        r"\breview\b|\boverview\b|\bresearch landscape\b|\bbibliometric\b|\bcurrent research\b|\bresearch directions?\b|\bpromise and pitfalls\b",
                        search_text,
                        flags=re.IGNORECASE,
                    )
                ),
                "_year": meta.get("year"),
            }
            self.task3_doc_lookup[str(payload["doc_id"] or "")] = payload
            self.task3_topic_rows.setdefault(topic, []).append(payload)
        alias_to_symbols: Dict[str, set] = {}
        for task in ("task1", "task2", "task3"):
            for row in self.metas[task]:
                meta = row.get("meta") or {}
                gene = normalize_gene_symbol(
                    row.get("gene_name")
                    or meta.get("gene_name")
                    or meta.get("resolved_symbol")
                )
                resolved = normalize_gene_symbol(meta.get("resolved_symbol") or gene)
                if not resolved:
                    continue
                for token in {gene, resolved}:
                    if token:
                        alias_to_symbols.setdefault(token, set()).add(resolved)
                for alias in meta.get("aliases") or []:
                    alias_norm = normalize_gene_symbol(alias)
                    if alias_norm:
                        alias_to_symbols.setdefault(alias_norm, set()).add(resolved)
        self.gene_alias_map = {
            alias: next(iter(symbols))
            for alias, symbols in alias_to_symbols.items()
            if len(symbols) == 1
        }
        self.gene_alias_ambiguous = {
            alias
            for alias, symbols in alias_to_symbols.items()
            if len(symbols) > 1
        }

    def resolve_gene_name(self, gene_name: Optional[str]) -> str:
        gene = normalize_gene_symbol(gene_name)
        if not gene:
            return ""
        if gene in self.gene_alias_ambiguous:
            return gene
        return str(self.gene_alias_map.get(gene) or gene)

    def _merge_ranked_hits(self, rows: Sequence[dict]) -> List[dict]:
        merged: Dict[str, dict] = {}
        for row in rows:
            doc_key = str(row.get("doc_id") or "")
            if not doc_key:
                continue
            score = float(row.get("score") or 0.0)
            row_meta = dict(row.get("meta") or {})
            multiquery_hits = int(row_meta.get("multiquery_hits") or 1)
            existing = merged.get(doc_key)
            if existing is None:
                payload = {
                    "score": score,
                    "doc_id": row.get("doc_id"),
                    "task": row.get("task"),
                    "text": row.get("text", ""),
                    "meta": dict(row_meta, multiquery_hits=multiquery_hits),
                }
                merged[doc_key] = payload
            else:
                existing["score"] = max(float(existing.get("score") or 0.0), score)
                existing_meta = dict(existing.get("meta") or {})
                existing_meta["multiquery_hits"] = int(existing_meta.get("multiquery_hits") or 1) + multiquery_hits
                for key, value in row_meta.items():
                    existing_meta.setdefault(key, value)
                existing["meta"] = existing_meta
        return list(merged.values())

    def _task3_vector_search(self, query_variants: Sequence[str], topk: int = 5, secondary_topk: int = 4) -> List[dict]:
        idx = self.indices["task3"]
        if idx.ntotal == 0:
            return []
        q_embs = _encode_texts(self.embed_model, list(query_variants), batch_size=min(len(query_variants), 4))
        rows = []
        for query_idx, q in enumerate(q_embs):
            variant_topk = topk if query_idx == 0 else max(3, min(secondary_topk, topk))
            scores, ids = idx.search(q.reshape(1, -1), variant_topk)
            for score, doc_id in zip(scores[0], ids[0]):
                if int(doc_id) < 0:
                    continue
                meta = self.metas["task3"][int(doc_id)]
                rows.append(
                    {
                        "score": float(score),
                        "doc_id": meta.get("doc_id"),
                        "task": "task3",
                        "text": meta.get("text", ""),
                        "meta": dict(meta.get("meta", {}), multiquery_hits=1),
                    }
                )
        return self._merge_ranked_hits(rows)

    def _task3_topic_candidate_score(
        self,
        query: str,
        row: dict,
        score_topic: str,
        prefer_review: bool = False,
    ) -> float:
        query_terms = _task3_query_terms(query)
        anchors = topic_anchor_terms(score_topic)
        search_text = str(row.get("_search_text") or "")
        text_lower = search_text.lower()
        title_lower = str(row.get("_title_text") or "")
        meta_signal_text = _task3_meta_signal_text(row.get("meta") or {}).lower()
        score = 0.0
        score += min(sum(term in text_lower for term in query_terms[:8]), 5) * 0.16
        score += min(sum(term in title_lower for term in query_terms[:8]), 3) * 0.22
        score += min(count_anchor_hits(search_text, anchors), 5) * 0.12
        if self.task3_keyword_deep and meta_signal_text:
            score += min(sum(term in meta_signal_text for term in query_terms[:8]), 4) * 0.12
            score += min(count_anchor_hits(meta_signal_text, anchors), 4) * 0.16
        if prefer_review and row.get("_review"):
            score += 0.6
        if row.get("_broad") and score_topic == "general_autism":
            score += 0.35
        try:
            year_val = int(row.get("_year"))
        except (TypeError, ValueError):
            year_val = None
        if wants_recent_evidence(query) and year_val is not None:
            score += max(0.0, min((year_val - 2010) / 25.0, 0.45))
        return score

    def _task3_pick_topic_candidates(
        self,
        query: str,
        topic: str,
        limit: int = 1,
        prefer_review: bool = False,
    ) -> List[dict]:
        rows = self.task3_topic_rows.get(topic, [])
        scored = []
        for row in rows:
            score = self._task3_topic_candidate_score(
                query=query,
                row=row,
                score_topic=topic,
                prefer_review=prefer_review,
            )
            if score <= 0:
                continue
            payload = {
                "score": score,
                "doc_id": row.get("doc_id"),
                "task": "task3",
                "text": row.get("text", ""),
                "meta": dict(row.get("meta") or {}, multiquery_hits=0),
            }
            scored.append(payload)
        scored.sort(
            key=lambda item: (
                -(item.get("score") or 0.0),
                -int(has_review_signal(
                    publication_type=(item.get("meta") or {}).get("publication_type", ""),
                    title=(item.get("meta") or {}).get("title", ""),
                    keywords=(item.get("meta") or {}).get("keywords", ""),
                )),
                str(item.get("doc_id") or ""),
            )
        )
        return scored[:limit]

    def _task3_iterative_followup_queries(
        self,
        query: str,
        expected_topic: str,
        base_hits: Sequence[dict],
    ) -> List[str]:
        top_topics = [str((hit.get("meta") or {}).get("topic") or "") for hit in list(base_hits)[:4]]
        raw_queries: List[str] = []
        if expected_topic == "general_autism":
            if not any(
                has_review_signal(
                    publication_type=(hit.get("meta") or {}).get("publication_type", ""),
                    title=(hit.get("meta") or {}).get("title", ""),
                    keywords=(hit.get("meta") or {}).get("keywords", ""),
                )
                for hit in list(base_hits)[:4]
            ):
                raw_queries.append(f"{query} autism review overview")
            missing_topics = [
                topic
                for topic in TASK3_BROAD_TOPICS
                if topic not in {"general_autism"} and topic not in top_topics
            ][:3]
            for topic in missing_topics:
                raw_queries.append(f"{query} autism {_task3_topic_query(topic)} review")
        else:
            if expected_topic not in top_topics:
                raw_queries.append(f"{query} autism {_task3_topic_query(expected_topic)} review")
            if wants_review_style(query):
                raw_queries.append(f"{query} review")
        seen = set()
        out = []
        base = _norm_space(query).lower()
        for item in raw_queries:
            normalized = _norm_space(item)
            key = normalized.lower()
            if not normalized or key == base or key in seen:
                continue
            seen.add(key)
            out.append(normalized)
        return out

    def _task3_graph_expand_hits(
        self,
        query: str,
        expected_topic: str,
        base_hits: Sequence[dict],
        limit: int = 8,
    ) -> List[dict]:
        seed_topics = [str((hit.get("meta") or {}).get("topic") or "") for hit in list(base_hits)[:4] if hit]
        candidate_topics = set(seed_topics)
        if expected_topic == "general_autism":
            candidate_topics.update(TASK3_BROAD_TOPICS)
        else:
            candidate_topics.add(expected_topic)
            candidate_topics.update(TASK3_TOPIC_GRAPH.get(expected_topic, ()))
        for topic in seed_topics:
            candidate_topics.update(TASK3_TOPIC_GRAPH.get(topic, ()))

        rows = list(base_hits)
        prefer_review = wants_review_style(query)
        for topic in sorted(candidate_topics):
            topic_limit = 2 if topic == expected_topic else 1
            rows.extend(
                self._task3_pick_topic_candidates(
                    query=query,
                    topic=topic,
                    limit=topic_limit,
                    prefer_review=prefer_review,
                )
            )
        merged = self._merge_ranked_hits(rows)
        merged.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return merged[:max(limit, 5)]

    def _task3_general_route_hits(self, query: str, base_hits: Sequence[dict], limit: int = 8) -> List[dict]:
        topic_best = []
        prefer_review = True
        for topic in TASK3_BROAD_TOPICS:
            topic_limit = 2 if topic == "general_autism" else 1
            topic_best.extend(
                self._task3_pick_topic_candidates(
                    query=query,
                    topic=topic,
                    limit=topic_limit,
                    prefer_review=prefer_review,
                )
            )
        merged = self._merge_ranked_hits(list(base_hits) + topic_best)
        merged.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return merged[:max(limit, 5)]

    def _task3_general_route_reviewmix_hits(self, query: str, base_hits: Sequence[dict], limit: int = 8) -> List[dict]:
        rows = list(base_hits)[:2]
        seeded_ids = {str((row.get("meta") or {}).get("doc_id") or row.get("doc_id") or "") for row in rows}

        def _extend(rows_in: Sequence[dict], boost_tag: str) -> None:
            for row in rows_in:
                doc_id = str((row.get("meta") or {}).get("doc_id") or row.get("doc_id") or "")
                if doc_id and doc_id in seeded_ids:
                    continue
                payload = dict(row)
                payload_meta = dict(payload.get("meta") or {})
                payload_meta["general_route_mix2"] = boost_tag
                payload["meta"] = payload_meta
                rows.append(payload)
                if doc_id:
                    seeded_ids.add(doc_id)

        _extend(
            self._task3_pick_topic_candidates(
                query=query,
                topic="general_autism",
                limit=3,
                prefer_review=True,
            ),
            "general_review",
        )
        for topic in ("methods_study_design", "risk_pathogenesis", "epidemiology_population", "diagnosis_screening"):
            _extend(
                self._task3_pick_topic_candidates(
                    query=query,
                    topic=topic,
                    limit=1,
                    prefer_review=True,
                ),
                f"broad_{topic}",
            )
        merged = self._merge_ranked_hits(rows)
        merged.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        return merged[:max(limit, 6)]

    def _task3_enhance_hits(self, query: str, hits: List[dict], topk: int) -> List[dict]:
        if not hits:
            return []
        runtime = dict(self.task3_runtime)
        expected_topic = infer_topic_from_question(query) or "general_autism"
        broad_query = _task3_is_broad_query(query, expected_topic)
        use_reviewmix = _env_flag("AUTISM_GPT_TASK3_GENERAL_ROUTE_MIX2")
        if self.task3_mode == "broad_user" and broad_query:
            runtime["self_route"] = True
            runtime["graph_rag"] = True
            runtime["iterative"] = True
            runtime["general_route"] = True
            use_reviewmix = True
        if not any(runtime.values()):
            return hits
        base_hits = list(hits)
        alt_hits = list(base_hits)

        if runtime["general_route"] and broad_query and expected_topic == "general_autism":
            if use_reviewmix:
                alt_hits = self._task3_general_route_reviewmix_hits(
                    query=query,
                    base_hits=alt_hits,
                    limit=max(topk + 3, 8),
                )
            else:
                alt_hits = self._task3_general_route_hits(query=query, base_hits=alt_hits, limit=max(topk + 3, 8))

        if runtime["graph_rag"]:
            alt_hits = self._task3_graph_expand_hits(
                query=query,
                expected_topic=expected_topic,
                base_hits=alt_hits,
                limit=max(topk + 3, 8),
            )

        if runtime["iterative"]:
            followup_queries = self._task3_iterative_followup_queries(
                query=query,
                expected_topic=expected_topic,
                base_hits=alt_hits,
            )
            if followup_queries:
                followup_hits = self._task3_vector_search(
                    query_variants=followup_queries,
                    topk=max(topk, 5),
                    secondary_topk=max(4, topk),
                )
                alt_hits = self._merge_ranked_hits(list(alt_hits) + list(followup_hits))

        if runtime["self_route"]:
            base_quality = _task3_route_quality(query=query, hits=base_hits, expected_topic=expected_topic)
            alt_quality = _task3_route_quality(query=query, hits=alt_hits, expected_topic=expected_topic)
            return alt_hits if alt_quality >= base_quality + 0.1 else base_hits
        return alt_hits

    def _task3_rerank(self, query: str, hits: List[dict]) -> List[dict]:
        if not hits:
            return []

        expected_topic = infer_topic_from_question(query) or "general_autism"
        anchors = topic_anchor_terms(expected_topic)
        prefer_review = wants_review_style(query)
        prefer_recent = wants_recent_evidence(query)
        query_terms = _task3_query_terms(query)
        reranked = []

        for rank, hit in enumerate(hits):
            meta = hit.get("meta") or {}
            parsed = _parse_task3_doc_text(hit.get("text", ""))
            topic = str(meta.get("topic") or infer_task3_topic(
                title=meta.get("title") or parsed.get("title", ""),
                abstract=parsed.get("abstract", ""),
                keywords=meta.get("keywords") or parsed.get("keywords", ""),
                publication_type=meta.get("publication_type") or parsed.get("publication_type", ""),
            ) or "general_autism")
            title = str(meta.get("title") or parsed.get("title", ""))
            keywords = str(meta.get("keywords") or parsed.get("keywords", ""))
            pub_type = str(meta.get("publication_type") or parsed.get("publication_type", ""))
            meta_signal_text = _task3_meta_signal_text(meta, parsed).lower()
            year = meta.get("year")
            multiquery_hits = int(meta.get("multiquery_hits") or 1)
            text = _norm_space(" ".join([title, parsed.get("abstract", ""), keywords, pub_type]))
            text_lower = text.lower()
            title_lower = _norm_space(title).lower()
            rerank_bonus = 0.0
            if expected_topic == topic:
                rerank_bonus += 2.0
            elif expected_topic == "general_autism":
                if topic != "general_autism":
                    rerank_bonus -= 0.12
            elif topic == "general_autism":
                rerank_bonus -= 0.08
            else:
                rerank_bonus -= 0.28

            rerank_bonus += min(max(multiquery_hits - 1, 0), 3) * 0.12
            rerank_bonus += min(count_anchor_hits(text, anchors), 6) * 0.18
            if query_terms:
                term_hits = sum(term in text_lower for term in query_terms[:8])
                title_term_hits = sum(term in title_lower for term in query_terms[:8])
                rerank_bonus += min(term_hits, 4) * 0.12
                rerank_bonus += min(title_term_hits, 2) * 0.18
                if self.task3_keyword_deep and meta_signal_text:
                    rerank_bonus += min(sum(term in meta_signal_text for term in query_terms[:8]), 4) * 0.11
            if self.task3_keyword_deep and meta_signal_text:
                rerank_bonus += min(count_anchor_hits(meta_signal_text, anchors), 4) * 0.14

            if prefer_review and has_review_signal(publication_type=pub_type, title=title, keywords=keywords):
                rerank_bonus += 0.65
            mix_tag = str(meta.get("general_route_mix2") or "")
            if expected_topic == "general_autism" and mix_tag:
                if mix_tag == "general_review":
                    rerank_bonus += 0.26
                else:
                    rerank_bonus += 0.08

            if prefer_recent:
                try:
                    year_val = int(year)
                except (TypeError, ValueError):
                    year_val = None
                if year_val is not None:
                    rerank_bonus += max(0.0, min((year_val - 2010) / 20.0, 0.6))

            if expected_topic == "methods_study_design":
                if re.search(r"\btrial\b|\bcohort\b|\bvalidation\b|\bmachine learning\b|\bsystematic review\b", text, flags=re.IGNORECASE):
                    rerank_bonus += 0.35
            elif expected_topic == "diagnosis_screening":
                if re.search(r"\bADOS\b|\bADI-R\b|\bM-CHAT\b|\bscreen|\bdiagnos|\bassess", text, flags=re.IGNORECASE):
                    rerank_bonus += 0.35
            elif expected_topic == "biomarkers":
                if re.search(r"\bbiomarker\b|\bEEG\b|\bMRI\b|\bblood\b|\bmetabol", text, flags=re.IGNORECASE):
                    rerank_bonus += 0.35
            elif expected_topic == "clinical_features":
                if re.search(r"\bclinical\b|\bfeature|\bdevelopment|\bsocial\b|\blanguage\b|\bsensory\b", text, flags=re.IGNORECASE):
                    rerank_bonus += 0.35
            elif expected_topic == "epidemiology_population":
                if re.search(r"\bprevalence\b|\bcohort\b|\bpopulation\b|\bsex\b|\bgender\b|\bdisparit|\bunderdiagnos|\bdemograph", text, flags=re.IGNORECASE):
                    rerank_bonus += 0.35
            elif expected_topic == "general_autism":
                if topic == "general_autism":
                    rerank_bonus += 0.45
                elif multiquery_hits >= 2:
                    rerank_bonus += 0.18
                if self.task3_keyword_deep and meta_signal_text:
                    if re.search(
                        r"\breview\b|\boverview\b|\bbibliometric\b|\bresearch directions?\b|\bconsensus\b|\bguideline\b",
                        meta_signal_text,
                        flags=re.IGNORECASE,
                    ):
                        rerank_bonus += 0.22
                if re.search(
                    r"\breview\b|\boverview\b|\bresearch landscape\b|\bbibliometric\b|\bhot topics\b|\bcurrent research\b|\badvances? in autism\b|\bresearch directions?\b|\bpromise and pitfalls\b",
                    text,
                    flags=re.IGNORECASE,
                ):
                    rerank_bonus += 0.5

            payload = dict(hit)
            payload["rerank_score"] = float(hit.get("score") or 0.0) + rerank_bonus - rank * 0.01
            payload["meta"] = dict(meta, topic=topic)
            reranked.append(payload)

        reranked.sort(key=lambda row: row.get("rerank_score", row.get("score", 0.0)), reverse=True)
        return reranked

    def _task2_rerank(self, query: str, hits: List[dict]) -> List[dict]:
        if not hits:
            return []

        target_gene = self.resolve_gene_name(_query_focus_gene(query))
        query_lower = _norm_space(query).lower()
        asks_literature = any(
            token in query_lower
            for token in ("cite", "citation", "pmid", "reference", "supporting literature", "supporting studies")
        )
        asks_strength = any(
            token in query_lower
            for token in ("associated with autism", "association evidence", "evaluate", "based on evidence")
        )
        reranked = []
        for rank, hit in enumerate(hits):
            meta = hit.get("meta") or {}
            doc_type = str(meta.get("doc_type") or "summary")
            gene_name = normalize_gene_symbol(meta.get("gene_name"))
            text = str(hit.get("text") or "")
            match_score = float(meta.get("match_score") or 0.0)
            direct_bonus = float(meta.get("autism_direct_bonus") or 0.0)
            evidence_tier = _task2_effective_evidence_tier(meta)
            evidence_rank = _task2_effective_tier_rank(meta)
            pubtator_support_rank = _task2_pubtator_support_rank(meta)

            rerank_bonus = 0.0
            if doc_type == "summary":
                rerank_bonus += 0.55
            elif doc_type == "literature":
                focus = _task2_literature_focus_features(target_gene, text, meta)
                rerank_bonus += 0.15 + min(match_score, 4.5) * 0.12 + min(direct_bonus, 2.6) * 0.18
                rerank_bonus += min(pubtator_support_rank, 4) * 0.14
                if self.task2_pheno_rerank:
                    rerank_bonus += _task2_phenotype_rerank_bonus(query=query, text=text, meta=meta)
                if evidence_tier == "direct_autism_association":
                    rerank_bonus += 0.5
                elif evidence_tier == "autism_association":
                    rerank_bonus += 0.3
                elif evidence_tier == "autism_phenotype":
                    rerank_bonus += 0.15
                elif evidence_tier == "autism_generic":
                    rerank_bonus += 0.02
                else:
                    rerank_bonus -= 0.06
                if focus["title_gene_match"]:
                    rerank_bonus += 0.55
                elif focus["body_gene_match"]:
                    rerank_bonus += 0.22
                if focus["validated_target_gene"]:
                    rerank_bonus += 0.4
                elif "pubtator_support_rank" in meta:
                    rerank_bonus -= 1.2
                if focus["autism_title"]:
                    rerank_bonus += 0.12
                elif focus["autism_body"]:
                    rerank_bonus += 0.05
                if focus["review_like"] and asks_literature:
                    rerank_bonus += 0.04
                if self.task2_keyword_deep:
                    keyword_block = _norm_space(
                        " ".join(
                            [
                                str(focus.get("keywords", "")),
                                str(focus.get("author_keywords", "")),
                                str(focus.get("mesh_terms", "")),
                                str(focus.get("supplement", "")),
                            ]
                        )
                    ).lower()
                    if keyword_block:
                        if any(term in keyword_block for term in AUTISM_TERMS):
                            rerank_bonus += 0.12
                        if any(
                            term in keyword_block
                            for term in (
                                "risk factor",
                                "genetic predisposition",
                                "association",
                                "variant",
                                "mutation",
                                "de novo",
                                "whole-exome sequencing",
                                "exome sequencing",
                                "cohort",
                                "case-control",
                            )
                        ):
                            rerank_bonus += 0.12
                    if focus["supplement_gene_match"] and (focus["supplement_autism"] or focus["supplement_association_signal"]):
                        rerank_bonus += 0.18
                if asks_literature:
                    rerank_bonus += 0.2
                if asks_strength:
                    if evidence_rank >= 3:
                        rerank_bonus += 0.14
                    elif evidence_rank == 2:
                        rerank_bonus += 0.05

            if target_gene:
                if gene_name == target_gene:
                    rerank_bonus += 2.8
                    if doc_type == "summary":
                        rerank_bonus += 0.45
                    else:
                        rerank_bonus += 0.7
                        if meta.get("pubtator_target_gene_hit"):
                            rerank_bonus += 0.35
                elif gene_name:
                    if doc_type == "summary":
                        rerank_bonus -= 1.8
                    else:
                        rerank_bonus -= 3.0
                if _gene_mentioned_in_text(text, target_gene):
                    rerank_bonus += 0.6

            if re.search(r"\bPMID:\s*[0-9]+\b", text, flags=re.IGNORECASE):
                rerank_bonus += 0.05

            payload = dict(hit)
            payload["meta"] = dict(meta, doc_type=doc_type)
            payload["rerank_score"] = float(hit.get("score") or 0.0) + rerank_bonus - rank * 0.01
            reranked.append(payload)

        reranked.sort(key=lambda row: row.get("rerank_score", row.get("score", 0.0)), reverse=True)
        return reranked

    def search(self, task: str, query: str, topk: int = 5) -> List[dict]:
        if task not in self.indices:
            return []
        idx = self.indices[task]
        if idx.ntotal == 0:
            return []
        if task == "task1" and self.task1_direct_lookup:
            task1_target_gene = self.resolve_gene_name(_query_focus_gene(query))
            if task1_target_gene:
                direct_doc = self.find_gene_doc("task1", task1_target_gene)
                if direct_doc and direct_doc.get("text"):
                    payload = dict(direct_doc)
                    payload["score"] = max(float(payload.get("score") or 0.0), 1.0)
                    return [payload]
        if task == "task3":
            query_variants = [query]
            query_variants.extend(_task3_decomposition_queries(query))
            out = self._task3_vector_search(
                query_variants=query_variants,
                topk=topk,
                secondary_topk=max(4, topk),
            )
            out = self._task3_enhance_hits(query=query, hits=out, topk=topk)
            return self._task3_rerank(query=query, hits=out)
        query_variants = [query]
        task2_target_gene = self.resolve_gene_name(_query_focus_gene(query)) if task == "task2" else ""
        q_embs = _encode_texts(self.embed_model, query_variants, batch_size=min(len(query_variants), 4))
        merged: Dict[str, dict] = {}
        for query_idx, q in enumerate(q_embs):
            if task == "task2":
                variant_topk = max(topk * 6, 30) if query_idx == 0 else max(6, min(topk * 2, 10))
            else:
                variant_topk = topk if query_idx == 0 else max(3, min(topk, 4))
            scores, ids = idx.search(q.reshape(1, -1), variant_topk)
            for score, doc_id in zip(scores[0], ids[0]):
                if int(doc_id) < 0:
                    continue
                meta = self.metas[task][int(doc_id)]
                doc_key = str(meta.get("doc_id") or int(doc_id))
                row = merged.get(doc_key)
                if row is None:
                    merged[doc_key] = {
                        "score": float(score),
                        "doc_id": meta.get("doc_id"),
                        "task": task,
                        "text": meta.get("text", ""),
                        "meta": dict(meta.get("meta", {}), multiquery_hits=1),
                    }
                else:
                    row["score"] = max(float(row.get("score") or 0.0), float(score))
                    row_meta = dict(row.get("meta") or {})
                    row_meta["multiquery_hits"] = int(row_meta.get("multiquery_hits") or 1) + 1
                    row["meta"] = row_meta
        out = list(merged.values())
        if task == "task2":
            if task2_target_gene:
                for row in self.find_gene_docs(task="task2", gene_name=task2_target_gene, limit=max(topk * 6, 24)):
                    doc_key = str(row.get("doc_id") or "")
                    if doc_key in merged:
                        continue
                    payload = dict(row)
                    payload["score"] = float(payload.get("score") or 0.0)
                    payload["meta"] = dict(payload.get("meta") or {}, multiquery_hits=0)
                    merged[doc_key] = payload
                out = list(merged.values())
            return self._task2_rerank(query=query, hits=out)
        return out

    def find_gene_docs(
        self,
        task: str,
        gene_name: str,
        doc_type: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[dict]:
        target = self.resolve_gene_name(gene_name)
        if not target or task not in self.metas:
            return []
        candidates = []
        for meta in self.metas[task]:
            meta_gene = (
                str(meta.get("gene_name") or meta.get("meta", {}).get("gene_name") or "")
                .strip()
                .upper()
            )
            if meta_gene == target:
                candidates.append(
                    {
                    "score": None,
                    "doc_id": meta.get("doc_id"),
                    "task": task,
                    "text": meta.get("text", ""),
                    "meta": meta.get("meta", {}),
                    }
                )

        def sort_key(row: dict) -> tuple:
            meta = row.get("meta") or {}
            doc_type = str(meta.get("doc_type") or "summary")
            doc_type_rank = 0 if doc_type == "summary" else 1
            try:
                match_score = float(meta.get("match_score") or 0.0)
            except (TypeError, ValueError):
                match_score = 0.0
            try:
                direct_bonus = float(meta.get("autism_direct_bonus") or 0.0)
            except (TypeError, ValueError):
                direct_bonus = 0.0
            return (doc_type_rank, -direct_bonus, -match_score, str(row.get("doc_id") or ""))

        candidates.sort(key=sort_key)
        if doc_type:
            target_doc_type = str(doc_type)
            candidates = [
                row for row in candidates
                if str((row.get("meta") or {}).get("doc_type") or "summary") == target_doc_type
            ]
        if limit is not None and limit > 0:
            return candidates[:limit]
        return candidates

    def find_gene_doc(self, task: str, gene_name: str) -> Optional[dict]:
        rows = self.find_gene_docs(task=task, gene_name=gene_name, limit=1)
        return rows[0] if rows else None


class TaskExpertGenerator:
    def __init__(self, expert_dirs: Dict[str, Optional[Path]], default_max_input_tokens: int = 1024):
        self.expert_dirs = {k: (Path(v) if v else None) for k, v in expert_dirs.items()}
        self.default_max_input_tokens = default_max_input_tokens
        self._cache = {}

    def has_expert(self, task: str) -> bool:
        path = self.expert_dirs.get(task)
        return bool(path and path.exists())

    def _load(self, task: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        if task in self._cache:
            return self._cache[task]

        adapter_dir = self.expert_dirs.get(task)
        if not adapter_dir or not adapter_dir.exists():
            raise FileNotFoundError(f"No expert adapter found for {task}.")

        adapter_cfg = _safe_read_json(adapter_dir / "adapter_config.json")
        base_model_name = adapter_cfg["base_model_name_or_path"]
        model_load_kwargs = {}
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                model_load_kwargs["dtype"] = torch.bfloat16
            else:
                model_load_kwargs["dtype"] = torch.float16

        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(base_model_name, **model_load_kwargs)
        model = PeftModel.from_pretrained(base, str(adapter_dir))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        self._cache[task] = {
            "adapter_dir": str(adapter_dir),
            "base_model_name": base_model_name,
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
        }
        return self._cache[task]

    def build_prompt(self, task: str, user_query: str, contexts: Sequence[str]) -> str:
        return (
            "### System:\n"
            f"{_build_system_prompt(task)}\n\n"
            "### Question:\n"
            f"{user_query}\n\n"
            "### Context:\n"
            f"{chr(10).join(contexts)}\n\n"
            "### Response:\n"
        )

    def generate(
        self,
        task: str,
        user_query: str,
        contexts: Sequence[str],
        max_input_tokens: Optional[int] = None,
        max_new_tokens: int = 220,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> dict:
        import torch

        bundle = self._load(task)
        tokenizer = bundle["tokenizer"]
        model = bundle["model"]
        device = bundle["device"]
        prompt = self.build_prompt(task=task, user_query=user_query, contexts=contexts)
        input_limit = max_input_tokens or self.default_max_input_tokens
        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=input_limit,
        ).to(device)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p
        with torch.no_grad():
            output = model.generate(**enc, **gen_kwargs)
        prompt_len = int(enc["input_ids"].shape[1])
        gen_tokens = output[0][prompt_len:]
        answer = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return {
            "answer": answer,
            "prompt_text": prompt,
            "adapter_dir": bundle["adapter_dir"],
            "base_model_name": bundle["base_model_name"],
            "max_input_tokens": input_limit,
            "max_new_tokens": max_new_tokens,
        }


def _select_generation_contexts(task: str, query: str, hits: List[dict], retriever: TaskRetriever) -> List[str]:
    if not hits:
        return []
    def _task2_meta_prefix(meta: Optional[dict]) -> str:
        meta = meta or {}
        parts = []
        pmid = _norm_space(str(meta.get("pmid", "")))
        if pmid:
            parts.append(f"{{PMID: {pmid}}}")
        support_rank = _task2_pubtator_support_rank(meta)
        if support_rank:
            parts.append(f"{{SupportRank: {support_rank}}}")
        effective_tier = _task2_effective_evidence_tier(meta)
        if effective_tier and effective_tier != "generic":
            parts.append(f"{{EffectiveTier: {effective_tier}}}")
        if meta.get("pubtator_target_gene_hit"):
            parts.append("{TargetGeneHit: 1}")
        if meta.get("pubtator_title_target_gene_hit"):
            parts.append("{TitleGeneHit: 1}")
        if meta.get("pubtator_abstract_target_gene_hit") or meta.get("pubtator_target_gene_hit"):
            parts.append("{BodyGeneHit: 1}")
        if meta.get("pubtator_title_autism_hit"):
            parts.append("{AutismTitleHit: 1}")
        if meta.get("pubtator_abstract_autism_hit") or meta.get("pubtator_autism_hit"):
            parts.append("{AutismBodyHit: 1}")
        return "".join(parts) + (" " if parts else "")

    def _task2_context_text(hit: dict) -> str:
        meta = hit.get("meta") or {}
        text = str(hit.get("text", ""))
        prefix = _task2_meta_prefix(meta)
        return f"{prefix}{text}" if prefix else text

    def _task3_context_text(hit: dict) -> str:
        meta = hit.get("meta") or {}
        text = str(hit.get("text", ""))
        prefix = _task2_meta_prefix(meta)
        return f"{prefix}{text}" if prefix else text

    if task == "task1":
        return [hits[0]["text"]]
    if task == "task2":
        contexts = []
        gene = _query_focus_gene(query) or hits[0].get("meta", {}).get("gene_name")
        if gene:
            gene = retriever.resolve_gene_name(gene)
        target_gene = normalize_gene_symbol(gene)

        summary_hit = None
        literature_hits = []
        for hit in hits:
            meta = hit.get("meta") or {}
            hit_gene = normalize_gene_symbol(meta.get("gene_name"))
            if target_gene and hit_gene and hit_gene != target_gene:
                continue
            doc_type = str(meta.get("doc_type") or "summary")
            if doc_type == "summary" and summary_hit is None:
                summary_hit = hit
            elif doc_type == "literature":
                literature_hits.append(hit)

        if gene:
            if summary_hit is None:
                summary_hit = retriever.find_gene_doc("task2", gene)
            if summary_hit and summary_hit.get("text"):
                contexts.append(summary_hit["text"])
            task1_doc = retriever.find_gene_doc("task1", gene)
            if task1_doc and task1_doc.get("text"):
                contexts.append(task1_doc["text"])
            direct_lit_hits = retriever.find_gene_docs(
                "task2",
                gene_name=gene,
                doc_type="literature",
                limit=8,
            )
            seen_ids = {row.get("doc_id") for row in literature_hits}
            for hit in direct_lit_hits:
                if hit.get("doc_id") not in seen_ids:
                    literature_hits.append(hit)
                    seen_ids.add(hit.get("doc_id"))
            if len(literature_hits) < 2 and target_gene:
                backup_hits = retriever.search("task3", f"{gene} autism", topk=8)
                gene_pat = re.compile(rf"(?<![A-Za-z0-9-]){re.escape(str(gene))}(?![A-Za-z0-9-])", flags=re.IGNORECASE)
                for hit in backup_hits:
                    lit_text = _task3_context_text(hit)
                    if lit_text and gene_pat.search(lit_text):
                        literature_hits.append(hit)
                    if len(literature_hits) >= 2:
                        break
            if retriever.task2_runtime.get("evidence_priority"):
                literature_hits = _task2_apply_generation_evidence_priority(
                    query=query,
                    gene=gene,
                    literature_hits=literature_hits,
                )
            def _lit_sort_key(hit: dict) -> tuple:
                meta = hit.get("meta") or {}
                text = _task2_context_text(hit) if hit.get("task") == "task2" else _task3_context_text(hit)
                focus = _task2_literature_focus_features(gene, text, meta)
                tier_rank = _task2_effective_tier_rank(meta)
                pubtator_rank = _task2_pubtator_support_rank(meta)
                try:
                    direct_bonus = float(meta.get("autism_direct_bonus") or 0.0)
                except (TypeError, ValueError):
                    direct_bonus = 0.0
                try:
                    match_score = float(meta.get("match_score") or 0.0)
                except (TypeError, ValueError):
                    match_score = 0.0
                try:
                    year_val = int(meta.get("year"))
                except (TypeError, ValueError):
                    year_val = -1
                return (
                    -(float(hit.get("_evidence_priority_score") or 0.0)),
                    -int(bool(meta.get("selected_support_entry"))),
                    -pubtator_rank,
                    -tier_rank,
                    -int(focus["validated_target_gene"]),
                    -int(focus["title_gene_match"]),
                    -int(focus["body_gene_match"]),
                    -int(focus["autism_title"]),
                    -int(focus["autism_body"]),
                    -direct_bonus,
                    -match_score,
                    -year_val,
                    str(hit.get("doc_id") or ""),
                )

            literature_hits = sorted(literature_hits, key=_lit_sort_key)
            for hit in literature_hits[:2]:
                lit_text = _task2_context_text(hit) if hit.get("task") == "task2" else _task3_context_text(hit)
                if lit_text:
                    contexts.append(lit_text)
        if not contexts:
            contexts.append(hits[0]["text"])
        return contexts[:4]
    return [_task3_context_text(h) for h in hits[:3]]


def _refine_query(query: str, predicted_task: str) -> str:
    # minimal rewrite for low-confidence router verdicts
    gene = _query_focus_gene(query)
    if predicted_task == "task1" and gene:
        return f"Gene information retrieval for {gene}: {query}"
    if predicted_task == "task2" and gene:
        return f"Association study reasoning for autism and gene {gene}: {query}"
    if predicted_task == "task3":
        return f"General autism knowledge query: {query}"
    return query


def _retrieval_judge_enabled(task: str) -> bool:
    if task not in {"task2", "task3"}:
        return False
    return _env_flag("AUTISM_GPT_RETRIEVAL_JUDGE1")


def _retrieval_judge_provider() -> str:
    return _norm_space(os.getenv("AUTISM_GPT_RETRIEVAL_JUDGE_PROVIDER", "deepseek")).lower() or "deepseek"


def _retrieval_judge_api_key(provider: str) -> str:
    if provider == "openai":
        return str(os.getenv("OPENAI_API_KEY", "")).strip()
    if provider == "deepseek":
        return str(os.getenv("DEEPSEEK_API_KEY", "")).strip()
    return ""


def _retrieval_judge_models(provider: str) -> List[str]:
    override = _norm_space(os.getenv("AUTISM_GPT_RETRIEVAL_JUDGE_MODEL", ""))
    if override:
        return [override]
    if provider == "openai":
        return ["gpt-5.4-mini", "gpt-4.1"]
    if provider == "deepseek":
        return ["deepseek-chat"]
    return []


def _post_json(url: str, headers: dict, payload: dict) -> requests.Response:
    return requests.post(url, headers=headers, json=payload, timeout=120)


def _extract_json_block(text: str) -> Optional[dict]:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None


def _task2_hit_preview(query: str, hit: dict) -> dict:
    meta = hit.get("meta") or {}
    text = _norm_space(str(hit.get("text") or ""))
    gene = _query_focus_gene(query)
    focus = _task2_literature_focus_features(gene, text, meta)
    return {
        "score": round(float(hit.get("score") or 0.0), 4),
        "title": str(meta.get("title") or ""),
        "pmid": str(meta.get("pmid") or ""),
        "doc_type": str(meta.get("doc_type") or ""),
        "effective_tier": _task2_effective_evidence_tier(meta),
        "pubtator_support_rank": _task2_pubtator_support_rank(meta),
        "validated_target_gene": int(bool(focus.get("validated_target_gene"))),
        "title_gene_match": int(bool(focus.get("title_gene_match"))),
        "body_gene_match": int(bool(focus.get("body_gene_match"))),
        "autism_title": int(bool(focus.get("autism_title"))),
        "autism_body": int(bool(focus.get("autism_body"))),
        "snippet": text[:320],
    }


def _task3_hit_preview(hit: dict) -> dict:
    meta = hit.get("meta") or {}
    text = _norm_space(str(hit.get("text") or ""))
    title = str(meta.get("title") or "")
    keywords = str(meta.get("keywords") or "")
    pub_type = str(meta.get("publication_type") or "")
    topic = str(meta.get("topic") or infer_task3_topic(
        title=title,
        abstract=text,
        keywords=keywords,
        publication_type=pub_type,
    ) or "general_autism")
    return {
        "score": round(float(hit.get("score") or 0.0), 4),
        "title": title,
        "pmid": str(meta.get("pmid") or ""),
        "topic": topic,
        "review_like": int(bool(has_review_signal(publication_type=pub_type, title=title, keywords=keywords))),
        "publication_type": pub_type,
        "snippet": text[:320],
    }


def _should_run_retrieval_judge(task: str, query: str, hits: List[dict], context_threshold: float) -> bool:
    if not _retrieval_judge_enabled(task):
        return False
    if not hits:
        return True
    if not _context_ok(hits, context_threshold):
        return True
    top_hits = hits[:3]
    if task == "task2":
        gene = _query_focus_gene(query)
        for hit in top_hits:
            meta = hit.get("meta") or {}
            if str(meta.get("doc_type") or "") != "literature":
                continue
            text = _norm_space(str(hit.get("text") or ""))
            focus = _task2_literature_focus_features(gene, text, meta)
            if (
                _task2_effective_tier_rank(meta) >= 3
                and _task2_pubtator_support_rank(meta) >= 2
                and (focus.get("validated_target_gene") or focus.get("title_gene_match") or focus.get("body_gene_match"))
                and (focus.get("autism_title") or focus.get("autism_body"))
            ):
                return False
        return True

    expected_topic = infer_topic_from_question(query) or "general_autism"
    if expected_topic != "general_autism" and not wants_review_style(query) and _context_ok(hits, context_threshold):
        return False
    top_topics = []
    broad_ok = False
    for hit in top_hits:
        meta = hit.get("meta") or {}
        title = str(meta.get("title") or "")
        keywords = str(meta.get("keywords") or "")
        pub_type = str(meta.get("publication_type") or "")
        topic = str(meta.get("topic") or infer_task3_topic(
            title=title,
            abstract=str(hit.get("text") or ""),
            keywords=keywords,
            publication_type=pub_type,
        ) or "general_autism")
        top_topics.append(topic)
        if expected_topic == "general_autism":
            if topic == "general_autism" or has_review_signal(publication_type=pub_type, title=title, keywords=keywords):
                broad_ok = True
    if expected_topic == "general_autism":
        return not broad_ok
    return expected_topic not in top_topics


def _call_retrieval_judge(task: str, query: str, hits: List[dict], stage: str) -> dict:
    provider = _retrieval_judge_provider()
    api_key = _retrieval_judge_api_key(provider)
    if not api_key:
        return {"decision": "GOOD_ENOUGH", "reason": f"judge_disabled_missing_{provider}_key", "provider": provider}
    models = _retrieval_judge_models(provider)
    if not models:
        return {"decision": "GOOD_ENOUGH", "reason": "judge_disabled_missing_model", "provider": provider}

    gene = _query_focus_gene(query)
    expected_topic = infer_topic_from_question(query) or "general_autism"
    previews = [
        _task2_hit_preview(query, hit) if task == "task2" else _task3_hit_preview(hit)
        for hit in hits[:5]
    ]

    system = (
        "You are a biomedical retrieval judge for an autism literature QA system. "
        "Decide whether the current retrieved papers are sufficient for a careful answer. "
        "Return strict JSON only with keys: decision, refined_query, reason. "
        "decision must be one of GOOD_ENOUGH, REFINE_QUERY, ABSTAIN."
    )
    if task == "task2":
        user_payload = {
            "task": task,
            "stage": stage,
            "query": query,
            "focus_gene": gene,
            "instruction": (
                "Prefer GOOD_ENOUGH only if the current hits contain sufficiently direct autism-relevant evidence "
                "for the target gene. Prefer REFINE_QUERY if the hits are partially relevant but too generic, indirect, "
                "or weak. Prefer ABSTAIN if a second retrieval is unlikely to recover direct support."
            ),
            "top_hits": previews,
        }
    else:
        user_payload = {
            "task": task,
            "stage": stage,
            "query": query,
            "expected_topic": expected_topic,
            "instruction": (
                "For broad autism questions, review and overview papers are acceptable support. "
                "Prefer REFINE_QUERY if the hits are autism-related but not well focused for the question. "
                "Prefer ABSTAIN only if the retrieved set is too weak to support a concise literature-backed answer."
            ),
            "top_hits": previews,
        }

    last_error = None
    for model in models:
        try:
            if provider == "openai":
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    "response_format": {"type": "json_object"},
                }
                resp = _post_json("https://api.openai.com/v1/chat/completions", headers=headers, payload=payload)
                if not resp.ok:
                    last_error = f"{model}: {resp.status_code} {resp.text[:300]}"
                    time.sleep(1)
                    continue
                data = resp.json()
                content = str(data["choices"][0]["message"]["content"] or "")
            elif provider == "deepseek":
                headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                payload = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
                    ],
                    "response_format": {"type": "json_object"},
                }
                resp = _post_json("https://api.deepseek.com/chat/completions", headers=headers, payload=payload)
                if not resp.ok:
                    last_error = f"{model}: {resp.status_code} {resp.text[:300]}"
                    time.sleep(1)
                    continue
                data = resp.json()
                content = str(data["choices"][0]["message"]["content"] or "")
            else:
                return {"decision": "GOOD_ENOUGH", "reason": f"unsupported_provider_{provider}", "provider": provider}

            parsed = _extract_json_block(content) or {}
            decision = str(parsed.get("decision") or "").strip().upper()
            if decision not in {"GOOD_ENOUGH", "REFINE_QUERY", "ABSTAIN"}:
                decision = "GOOD_ENOUGH"
            refined_query = _norm_space(str(parsed.get("refined_query") or ""))
            reason = _norm_space(str(parsed.get("reason") or ""))
            return {
                "provider": provider,
                "model": model,
                "decision": decision,
                "refined_query": refined_query,
                "reason": reason,
            }
        except Exception as exc:  # pragma: no cover
            last_error = f"{model}: {exc}"
            time.sleep(1)
    return {"decision": "GOOD_ENOUGH", "reason": f"judge_error:{last_error}", "provider": provider}


def _retrieval_abstain_answer(task: str, query: str) -> str:
    gene = _query_focus_gene(query)
    if task == "task2":
        focus = gene or "this gene"
        return (
            "<reasoning>\n"
            f"The selected retrieval did not provide sufficiently direct autism-specific evidence for {focus}.\n"
            "</reasoning>\n"
            "<answer>\n"
            f"The selected corpus does not provide sufficiently direct autism-specific evidence to support a stronger conclusion for {focus}. A more focused literature search may be needed.\n"
            "</answer>\n"
            "<reference>\n"
            "No sufficiently direct PMID-backed autism association paper was identified in the selected context.\n"
            "</reference>"
        )
    return (
        "Summary: The selected retrieval does not provide a sufficiently focused literature set to support a concise "
        "autism overview for this question.\n"
        "Evidence highlights:\n"
        "- No sufficiently focused PMID-backed synthesis was identified in the selected context."
    )


def _maybe_apply_retrieval_judge(
    task: str,
    query: str,
    query_used: str,
    hits: List[dict],
    retriever: TaskRetriever,
    topk: int,
    context_threshold: float,
) -> dict:
    trace: List[dict] = []
    if not _should_run_retrieval_judge(task, query, hits, context_threshold):
        return {"query_used": query_used, "hits": hits, "judge_trace": trace, "abstain_answer": None}

    first = _call_retrieval_judge(task=task, query=query, hits=hits, stage="first_pass")
    trace.append(dict(first, query_used=query_used))
    decision = str(first.get("decision") or "GOOD_ENOUGH").upper()
    if decision == "GOOD_ENOUGH":
        return {"query_used": query_used, "hits": hits, "judge_trace": trace, "abstain_answer": None}
    if decision == "ABSTAIN":
        return {
            "query_used": query_used,
            "hits": hits,
            "judge_trace": trace,
            "abstain_answer": _retrieval_abstain_answer(task, query),
        }

    refined_query = _norm_space(str(first.get("refined_query") or "")) or _refine_query(query_used, task)
    if not refined_query or refined_query == query_used:
        refined_query = _refine_query(query_used, task)
    hits_refined = retriever.search(task, refined_query, topk=topk)
    hits_refined = _promote_explicit_gene_hit(task=task, query=refined_query, hits=hits_refined, retriever=retriever)
    if not _context_ok(hits_refined, context_threshold):
        query_retry = _norm_space(re.sub(r"[^\w\s-]", " ", refined_query))
        hits_retry = retriever.search(task, query_retry, topk=topk)
        hits_retry = _promote_explicit_gene_hit(task=task, query=query_retry, hits=hits_retry, retriever=retriever)
        if _context_ok(hits_retry, context_threshold):
            refined_query = query_retry
            hits_refined = hits_retry

    second = _call_retrieval_judge(task=task, query=query, hits=hits_refined, stage="second_pass")
    trace.append(dict(second, query_used=refined_query))
    second_decision = str(second.get("decision") or "GOOD_ENOUGH").upper()
    if second_decision == "GOOD_ENOUGH":
        return {"query_used": refined_query, "hits": hits_refined, "judge_trace": trace, "abstain_answer": None}
    if second_decision == "REFINE_QUERY":
        return {
            "query_used": refined_query,
            "hits": hits_refined,
            "judge_trace": trace,
            "abstain_answer": _retrieval_abstain_answer(task, query),
        }
    return {
        "query_used": refined_query,
        "hits": hits_refined,
        "judge_trace": trace,
        "abstain_answer": _retrieval_abstain_answer(task, query),
    }


def _maybe_override_task1(query: str, predicted_task: str, confidence: float) -> Tuple[str, float]:
    q = _norm_space(query).lower()
    if predicted_task == "task1":
        return predicted_task, confidence
    gene = _query_focus_gene(query)
    if not gene:
        return predicted_task, confidence

    task1_terms = (
        "grch",
        "coordinate",
        "coordinates",
        "ensembl",
        "gene context",
        "gene function",
        "function summary",
        "omim",
        "qtl evidence",
        "eqtl",
        "sqtl",
        "p-value",
        "p value",
        "best reported",
        "brain-region qtl",
        "reported qtl",
        "reported brain qtl",
    )
    task2_terms = (
        "autism association",
        "association study",
        "associated with autism",
        "association evidence",
        "risk for autism",
        "linked to autism",
        "linked with autism",
        "autism-related association",
        "mechanism",
        "mechanistic",
        "pathway",
        "literature support",
    )
    if any(term in q for term in task1_terms) and not any(term in q for term in task2_terms):
        return "task1", max(confidence, 0.85)
    return predicted_task, confidence


def _maybe_override_task3(query: str, predicted_task: str, confidence: float) -> Tuple[str, float]:
    q = _norm_space(query).lower()
    if predicted_task == "task3":
        return predicted_task, confidence
    if _query_focus_gene(query):
        return predicted_task, confidence
    association_terms = (
        "gene",
        "variant",
        "eqtl",
        "sqtl",
        "association",
        "candidate gene",
        "genetic information",
        "genomic coordinate",
    )
    broad_task3_signal = (
        infer_topic_from_question(query) is not None
        or wants_review_style(query)
        or "research directions" in q
        or "high-level" in q
        or "literature-backed" in q
    )
    if broad_task3_signal and not any(term in q for term in association_terms):
        return "task3", max(confidence, 0.85)
    return predicted_task, confidence


def _context_ok(hits: List[dict], min_score: float) -> bool:
    if not hits:
        return False
    return max(h["score"] for h in hits) >= min_score


def _promote_explicit_gene_hit(task: str, query: str, hits: List[dict], retriever: TaskRetriever) -> List[dict]:
    if task not in {"task1", "task2"}:
        return hits
    gene = _query_focus_gene(query)
    if not gene:
        return hits
    gene_doc = retriever.find_gene_doc(task, gene)
    if not gene_doc or not gene_doc.get("text"):
        return hits

    top_score = max([float(h.get("score") or 0.0) for h in hits] + [1.0])
    gene_doc["score"] = top_score
    out = [gene_doc]
    seen = {gene_doc.get("doc_id")}
    for hit in hits:
        doc_id = hit.get("doc_id")
        if doc_id in seen:
            continue
        out.append(hit)
        seen.add(doc_id)
    return out


def _build_system_prompt(task: str) -> str:
    if task == "task1":
        return (
            "You are a biologist focused on autism genomics. You answer retrieval-style questions about genes, "
            "identifiers, genomic coordinates, and brain-region eQTL or sQTL strictly from the provided context. "
            "Do not use external knowledge. Do not infer mechanisms beyond the retrieved evidence.\n\n"
            "Always output exactly in this format:\n"
            "<reasoning>\n"
            "(concise steps, or N/A)\n"
            "</reasoning>\n"
            "<answer>\n"
            "(final answer only; prefer exact tokens from context)\n"
            "</answer>\n"
            "<reference>\n"
            "(supporting evidence copied or normalized from context)\n"
            "</reference>\n\n"
            "If information is missing, answer exactly: Not specified in the provided context. "
            "Preserve brain-region names exactly, including underscores."
        )
    if task == "task2":
        return (
            "You are an autism association-study analyst. Use only the provided context. "
            "Always output exactly in the format <reasoning>...</reasoning><answer>...</answer><reference>...</reference>. "
            "In reasoning, organize the evidence around statistics, mechanism, region, and variant evidence. "
            "In the reference block, include supporting statistics and literature citations with PMID when available."
        )
    return (
        "You are an autism research assistant. Answer only from the provided evidence. "
        "Write a short synthesis first, then 2-3 brief evidence bullets. Prefer themes over raw title dumping. "
        "Mention study titles only when they support a point. Do not use outside knowledge."
    )


def _split_sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", _norm_space(text))
    return [p for p in parts if p]


def _first_sentences(text: str, n: int = 2) -> str:
    sents = _split_sentences(text)
    return " ".join(sents[:n]) if sents else _norm_space(text)


def _support_query_terms(query: str) -> List[str]:
    skip = {
        "about",
        "across",
        "answer",
        "autism",
        "autistic",
        "backed",
        "based",
        "broad",
        "cite",
        "cited",
        "current",
        "describe",
        "disorder",
        "evidence",
        "features",
        "from",
        "gene",
        "genes",
        "give",
        "high",
        "how",
        "important",
        "is",
        "knowledge",
        "level",
        "literature",
        "paper",
        "papers",
        "pmid",
        "recent",
        "reported",
        "research",
        "review",
        "studies",
        "study",
        "summarize",
        "summary",
        "support",
        "supporting",
        "spectrum",
        "using",
        "what",
        "which",
    }
    terms: List[str] = []
    for token in re.findall(r"[a-z][a-z0-9-]{2,}", _norm_space(query).lower()):
        if token in skip or token in terms:
            continue
        terms.append(token)
    return terms


def _sentence_chunks(text: str, max_sentences: int = 2) -> List[str]:
    sents = _split_sentences(text)
    if not sents:
        normalized = _norm_space(text)
        return [normalized] if normalized else []
    out: List[str] = []
    max_sentences = max(1, min(max_sentences, len(sents)))
    for span in range(1, max_sentences + 1):
        for start in range(0, len(sents) - span + 1):
            out.append(" ".join(sents[start : start + span]))
    return out


def _best_support_snippet(
    text: str,
    query: str,
    gene: Optional[str] = None,
    topic: Optional[str] = None,
    max_sentences: int = 2,
) -> str:
    candidates = _sentence_chunks(text, max_sentences=max_sentences)
    if not candidates:
        return _norm_space(text)
    anchors = topic_anchor_terms(topic or "general_autism")
    query_terms = _support_query_terms(query)
    gene = _norm_space(gene or "")
    scored: List[Tuple[float, int, str]] = []
    for idx, candidate in enumerate(candidates):
        candidate_text = _norm_space(candidate)
        lower = candidate_text.lower()
        overlap = sum(1 for term in query_terms if term in lower)
        anchor_hits = count_anchor_hits(lower, anchors) if anchors else 0
        autism_hits = sum(1 for term in AUTISM_TERMS if term in lower)
        gene_hit = bool(gene and re.search(rf"\b{re.escape(gene.lower())}\b", lower))
        score = overlap * 1.6 + anchor_hits * 1.2 + min(autism_hits, 1) * 1.3
        if gene_hit:
            score += 3.5
        if re.search(r"\b(p\s*[<=>]|odds ratio|confidence interval|prevalence|incidence|cohort|trial|review|assessment|diagnos|screen|ados|m-?chat|biomarker|eeg|mri)\b", lower):
            score += 0.6
        if overlap == 0 and anchor_hits == 0 and not autism_hits and not gene_hit:
            score -= 1.5
        if len(candidate_text.split()) < 8:
            score -= 0.5
        if re.match(r"^(autism spectrum disorder|autism|[a-z0-9-]+ is a\b)", lower):
            score -= 0.3
        score -= idx * 0.03
        scored.append((score, overlap + anchor_hits + int(gene_hit), candidate_text))
    scored.sort(key=lambda item: (item[0], item[1], len(item[2])), reverse=True)
    return scored[0][2]


def _parse_task1_doc_text(text: str) -> dict:
    text = _norm_space(text)
    out = {}
    patterns = {
        "gene": r"Gene:\s*(.*?)\.",
        "coordinates": r"Coordinates:\s*(.*?)\.\s*eQTL regions:",
        "eqtl_regions": r"eQTL regions:\s*(.*?)\.\s*sQTL regions:",
        "sqtl_regions": r"sQTL regions:\s*(.*?)\.\s*Best eQTL p-value:",
        "eqtl_p": r"Best eQTL p-value:\s*(.*?)\.\s*Best sQTL p-value:",
        "sqtl_p": r"Best sQTL p-value:\s*(.*?)\.\s*Gene context:",
        "gene_context": r"Gene context:\s*(.*?)\.\s*OMIM function:",
        "omim": r"OMIM function:\s*(.*?)\.$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = _norm_space(match.group(1)) if match else ""
    return out


def _parse_task2_doc_text(text: str) -> dict:
    text = _norm_space(text)
    out = {}
    patterns = {
        "gene": r"Association evidence for autism candidate gene\s+(.*?)\.",
        "eqtl_regions": r"eQTL evidence regions:\s*(.*?)\.\s*sQTL evidence regions:",
        "sqtl_regions": r"sQTL evidence regions:\s*(.*?)\.\s*Top eQTL variants:",
        "eqtl_vars": r"Top eQTL variants:\s*(.*?)\.\s*Top sQTL variants:",
        "sqtl_vars": r"Top sQTL variants:\s*(.*?)\.\s*Statistical minima:",
        "stats": r"Statistical minima:\s*(.*?)\.\s*OMIM note:",
        "omim": r"OMIM note:\s*(.*?)\.$",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = _norm_space(match.group(1)) if match else ""
    return out


def _parse_task3_doc_text(text: str) -> dict:
    text = _norm_space(text)
    prefix_tags = {}
    while True:
        prefix_match = re.match(r"\{([A-Za-z0-9_]+):\s*([^{}]+)\}\s*", text)
        if not prefix_match:
            break
        prefix_tags[prefix_match.group(1)] = _norm_space(prefix_match.group(2))
        text = _norm_space(text[prefix_match.end() :])
    pmid = str(prefix_tags.get("PMID") or "")
    suffix_match = re.search(r"\(PMID:\s*([0-9]+)\)\s*$", text, flags=re.IGNORECASE)
    if suffix_match:
        pmid = pmid or suffix_match.group(1)
        text = _norm_space(text[: suffix_match.start()])
    def _tag_int(name: str) -> int:
        raw = prefix_tags.get(name)
        try:
            return max(0, int(str(raw or "0").strip()))
        except (TypeError, ValueError):
            return 0

    def _tag_flag(name: str) -> bool:
        return str(prefix_tags.get(name) or "").strip() in {"1", "true", "True", "yes", "Yes"}

    title_match = re.search(r"Title:\s*(.*?)\.\s*Abstract:", text)
    abstract_match = re.search(r"Abstract:\s*(.*?)\.\s*Keywords:", text)
    keywords_match = re.search(
        r"Keywords:\s*(.*?)\.(?:\s*Author keywords:|\s*MeSH terms:|\s*Publication types:|\s*Supplementary:|\s*$)",
        text,
    )
    author_keywords_match = re.search(
        r"Author keywords:\s*(.*?)\.(?:\s*MeSH terms:|\s*Publication types:|\s*Supplementary:|\s*$)",
        text,
    )
    mesh_terms_match = re.search(
        r"MeSH terms:\s*(.*?)\.(?:\s*Publication types:|\s*Supplementary:|\s*$)",
        text,
    )
    pub_types_match = re.search(r"Publication types:\s*(.*?)\.(?:\s*Supplementary:|\s*$)", text)
    supplement_match = re.search(r"Supplementary:\s*(.*?)\.\s*$", text)
    if not title_match or not abstract_match or not keywords_match:
        return {
            "title": "",
            "abstract": text,
            "keywords": "",
            "author_keywords": "",
            "mesh_terms": "",
            "supplement": "",
            "publication_type": "",
            "pmid": pmid,
            "support_rank_meta": _tag_int("SupportRank"),
            "effective_tier_meta": _norm_space(str(prefix_tags.get("EffectiveTier") or "")),
            "target_gene_hit_meta": _tag_flag("TargetGeneHit"),
            "title_gene_hit_meta": _tag_flag("TitleGeneHit"),
            "body_gene_hit_meta": _tag_flag("BodyGeneHit"),
            "autism_title_hit_meta": _tag_flag("AutismTitleHit"),
            "autism_body_hit_meta": _tag_flag("AutismBodyHit"),
        }
    return {
        "title": _norm_space(title_match.group(1)),
        "abstract": _norm_space(abstract_match.group(1)),
        "keywords": _norm_space(keywords_match.group(1)),
        "author_keywords": _norm_space(author_keywords_match.group(1)) if author_keywords_match else "",
        "mesh_terms": _norm_space(mesh_terms_match.group(1)) if mesh_terms_match else "",
        "supplement": _norm_space(supplement_match.group(1)) if supplement_match else "",
        "publication_type": _norm_space(pub_types_match.group(1)) if pub_types_match else "",
        "pmid": pmid,
        "support_rank_meta": _tag_int("SupportRank"),
        "effective_tier_meta": _norm_space(str(prefix_tags.get("EffectiveTier") or "")),
        "target_gene_hit_meta": _tag_flag("TargetGeneHit"),
        "title_gene_hit_meta": _tag_flag("TitleGeneHit"),
        "body_gene_hit_meta": _tag_flag("BodyGeneHit"),
        "autism_title_hit_meta": _tag_flag("AutismTitleHit"),
        "autism_body_hit_meta": _tag_flag("AutismBodyHit"),
    }


_TASK2_EVIDENCE_TIER_RANK = {
    "direct_autism_association": 4,
    "autism_association": 3,
    "autism_phenotype": 2,
    "autism_generic": 1,
    "generic_gene_literature": 1,
    "generic_literature": 1,
    "generic": 0,
}


def _task2_pubtator_support_rank(meta: Optional[dict]) -> int:
    if not meta:
        return 0
    try:
        return max(0, int(meta.get("pubtator_support_rank") or 0))
    except (TypeError, ValueError):
        return 0


def _task2_effective_evidence_tier(meta: Optional[dict]) -> str:
    if not meta:
        return "generic"
    pub_rank = _task2_pubtator_support_rank(meta)
    if pub_rank >= 4:
        return "direct_autism_association"
    if pub_rank == 3:
        return "autism_association"
    if pub_rank == 2:
        return "autism_phenotype"
    if pub_rank == 1:
        return "generic_gene_literature"
    pub_tier = _norm_space(meta.get("evidence_tier_pubtator", ""))
    if pub_tier:
        return pub_tier
    raw_tier = _norm_space(meta.get("evidence_tier", ""))
    return raw_tier or "generic"


def _task2_effective_tier_rank(meta: Optional[dict]) -> int:
    return _TASK2_EVIDENCE_TIER_RANK.get(_task2_effective_evidence_tier(meta), 0)


def _task2_literature_focus_features(gene: Optional[str], text: str, meta: Optional[dict] = None) -> dict:
    parsed = _parse_task3_doc_text(text)
    title = _norm_space(parsed.get("title", ""))
    abstract = _norm_space(parsed.get("abstract", ""))
    keywords = _norm_space(parsed.get("keywords", ""))
    author_keywords = _norm_space(parsed.get("author_keywords", "") or str((meta or {}).get("author_keywords", "")))
    mesh_terms = _norm_space(parsed.get("mesh_terms", "") or str((meta or {}).get("mesh_terms", "")))
    supplement = _norm_space(parsed.get("supplement", "") or str((meta or {}).get("supplement_excerpt") or (meta or {}).get("supplement_text", "")))
    publication_type = _norm_space(parsed.get("publication_type", ""))
    search_space = " ".join(part for part in (abstract, keywords, author_keywords, mesh_terms, supplement) if part)
    gene_norm = normalize_gene_symbol(gene)
    meta = meta or {}
    title_gene_match = bool(gene_norm and _gene_mentioned_in_text(title, gene_norm))
    body_gene_match = bool(gene_norm and _gene_mentioned_in_text(search_space, gene_norm))
    supplement_gene_match = bool(gene_norm and _gene_mentioned_in_text(supplement, gene_norm))
    autism_title = bool(
        re.search(r"\bautism\b|\bautistic\b|\bautism spectrum disorder\b", title, flags=re.IGNORECASE)
    )
    autism_body = has_autism_signal(
        title=title,
        abstract=abstract,
        keywords=" ".join(part for part in (keywords, author_keywords, mesh_terms, supplement) if part),
        publication_type=publication_type,
    )
    supplement_autism = has_autism_signal(
        title="",
        abstract="",
        keywords=supplement,
        publication_type="",
    )
    supplement_association_signal = bool(
        re.search(
            r"\bassociation\b|\bassociated\b|\brisk factors?\b|\bvariant\b|\bmutation\b|\bde novo\b|\bcohort\b|\bcase-control\b|\bwhole[- ]exome\b|\bsequencing\b",
            supplement,
            flags=re.IGNORECASE,
        )
    )
    pub_title_gene_match = bool(meta.get("pubtator_title_target_gene_hit"))
    pub_body_gene_match = bool(meta.get("pubtator_abstract_target_gene_hit") or meta.get("pubtator_target_gene_hit"))
    pub_autism_title = bool(meta.get("pubtator_title_autism_hit"))
    pub_autism_body = bool(meta.get("pubtator_abstract_autism_hit") or meta.get("pubtator_autism_hit"))
    title_gene_match = bool(title_gene_match or pub_title_gene_match)
    body_gene_match = bool(body_gene_match or pub_body_gene_match or supplement_gene_match)
    autism_title = bool(autism_title or pub_autism_title)
    autism_body = bool(autism_body or pub_autism_body or supplement_autism)
    review_like = has_review_signal(
        publication_type=publication_type,
        title=title,
        keywords=" ".join(part for part in (keywords, author_keywords, mesh_terms) if part),
    )
    return {
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "author_keywords": author_keywords,
        "mesh_terms": mesh_terms,
        "supplement": supplement,
        "publication_type": publication_type,
        "title_gene_match": title_gene_match,
        "body_gene_match": body_gene_match,
        "supplement_gene_match": supplement_gene_match,
        "autism_title": autism_title,
        "autism_body": autism_body,
        "supplement_autism": supplement_autism,
        "supplement_association_signal": supplement_association_signal,
        "review_like": review_like,
        "validated_target_gene": bool(meta.get("pubtator_target_gene_hit")),
        "pubtator_support_rank": _task2_pubtator_support_rank(meta),
        "effective_evidence_tier": _task2_effective_evidence_tier(meta),
    }


def _task2_phenotype_rerank_bonus(query: str, text: str, meta: Optional[dict] = None) -> float:
    q = _norm_space(query).lower()
    if not any(
        token in q
        for token in (
            "autism",
            "asd",
            "autistic",
            "association evidence",
            "associated with autism",
            "based on evidence",
            "supporting studies",
            "pmid",
            "reference",
        )
    ):
        return 0.0
    parsed = _parse_task3_doc_text(text)
    title = _norm_space(parsed.get("title", ""))
    abstract = _norm_space(parsed.get("abstract", ""))
    keywords = _norm_space(parsed.get("keywords", ""))
    publication_type = _norm_space(parsed.get("publication_type", ""))
    combined = " ".join(part for part in (title, abstract, keywords, publication_type) if part)
    meta = meta or {}
    score = 0.0
    if has_autism_signal(
        title=title,
        abstract=abstract,
        keywords=keywords,
        publication_type=publication_type,
    ) or bool(meta.get("pubtator_autism_hit")):
        score += 0.22
    for pattern, bonus in (
        (r"\bdevelopmental delay\b|\bglobal developmental delay\b|\bintellectual disability\b|\bneurodevelopment(?:al)?\b", 0.12),
        (r"\bsocial\b|\blanguage\b|\bspeech\b|\bsensory\b|\bbehavior(?:al)?\b|\bhyperactivity\b|\battention\b|\badhd\b", 0.08),
        (r"\bseizure\b|\bepilep\w*\b|\bhypotonia\b|\bmotor\b|\bataxia\b", 0.06),
    ):
        if re.search(pattern, combined, flags=re.IGNORECASE):
            score += bonus
    if has_review_signal(publication_type=publication_type, title=title, keywords=keywords):
        score += 0.03
    if is_probable_non_autism_record(
        title=title,
        abstract=abstract,
        keywords=keywords,
        publication_type=publication_type,
    ):
        score -= 0.3
    return round(score, 4)


def _task2_entry_support_score(entry: dict, gene: Optional[str]) -> float:
    gene_norm = normalize_gene_symbol(gene)
    title = _norm_space(str(entry.get("title", "")))
    abstract = _norm_space(str(entry.get("abstract", "")))
    keywords = _norm_space(str(entry.get("keywords", "")))
    publication_type = _norm_space(str(entry.get("publication_type", "")))
    search_space = " ".join(part for part in (abstract, keywords) if part)
    support_rank = 0
    try:
        support_rank = max(
            0,
            int(
                entry.get("support_rank")
                or entry.get("support_rank_meta")
                or 0
            ),
        )
    except (TypeError, ValueError):
        support_rank = 0
    evidence_tier = _norm_space(
        str(
            entry.get("effective_evidence_tier")
            or entry.get("effective_tier_meta")
            or entry.get("evidence_tier")
            or "generic"
        )
    ) or "generic"
    tier_rank = _TASK2_EVIDENCE_TIER_RANK.get(evidence_tier, 0)
    title_gene_match = bool(entry.get("title_gene_hit_meta")) or bool(gene_norm and _gene_mentioned_in_text(title, gene_norm))
    body_gene_match = bool(entry.get("body_gene_hit_meta")) or bool(gene_norm and _gene_mentioned_in_text(search_space, gene_norm))
    target_gene_hit = bool(entry.get("target_gene_hit_meta")) or title_gene_match or body_gene_match
    autism_title = bool(entry.get("autism_title_hit_meta")) or bool(
        re.search(r"\bautism\b|\bautistic\b|\bautism spectrum disorder\b|\bASD\b", title, flags=re.IGNORECASE)
    )
    autism_body = bool(entry.get("autism_body_hit_meta")) or has_autism_signal(
        title=title,
        abstract=abstract,
        keywords=keywords,
        publication_type=publication_type,
    )
    review_like = has_review_signal(
        publication_type=publication_type,
        title=title,
        keywords=keywords,
    )
    score = tier_rank * 0.95 + min(support_rank, 4) * 0.55
    if target_gene_hit:
        score += 0.55
    if title_gene_match:
        score += 0.35
    elif body_gene_match:
        score += 0.16
    if autism_title:
        score += 0.22
    elif autism_body:
        score += 0.1
    if review_like:
        score += 0.04
    weak_case_like = bool(
        re.search(
            r"\bcase report\b|\bcase series\b|\bsyndrome\b|\btranslocation\b|\bmultiple congenital anomalies\b",
            " ".join(part for part in (title, abstract, publication_type) if part),
            flags=re.IGNORECASE,
        )
    )
    if weak_case_like and tier_rank <= 3:
        score -= 0.18
    return round(score, 4)


def _task2_literature_support_profile(entries: Sequence[dict], gene: Optional[str] = None) -> dict:
    direct_autism_entries: List[dict] = []
    autism_association_entries: List[dict] = []
    autism_phenotype_entries: List[dict] = []
    autism_generic_entries: List[dict] = []
    autism_entries: List[dict] = []
    broader_ndd_entries: List[dict] = []
    generic_entries: List[dict] = []
    for entry in entries:
        title = _norm_space(str(entry.get("title", "")))
        abstract = _norm_space(str(entry.get("abstract", "")))
        keywords = _norm_space(str(entry.get("keywords", "")))
        publication_type = _norm_space(str(entry.get("publication_type", "")))
        _, evidence_tier = _task2_literature_bonus(
            title=title,
            abstract=abstract,
            keywords=keywords,
            author_keywords="",
            mesh_terms="",
            publication_type=publication_type,
        )
        payload = dict(entry)
        payload["evidence_tier"] = evidence_tier
        payload["effective_evidence_tier"] = _norm_space(str(payload.get("effective_tier_meta") or evidence_tier)) or evidence_tier
        payload["support_rank"] = max(
            0,
            int(payload.get("support_rank_meta") or 0),
        )
        payload["entry_support_score"] = _task2_entry_support_score(payload, gene=gene)
        evidence_tier = payload["evidence_tier"]
        if has_autism_signal(
            title=title,
            abstract=abstract,
            keywords=keywords,
            publication_type=publication_type,
        ) and not is_probable_non_autism_record(
            title=title,
            abstract=abstract,
            keywords=keywords,
            publication_type=publication_type,
        ):
            autism_entries.append(payload)
            if evidence_tier == "direct_autism_association":
                direct_autism_entries.append(payload)
            elif evidence_tier == "autism_association":
                autism_association_entries.append(payload)
            elif evidence_tier == "autism_phenotype":
                autism_phenotype_entries.append(payload)
            else:
                autism_generic_entries.append(payload)
            continue

        text = " ".join([title, abstract, keywords, publication_type])
        if re.search(
            r"\bneurodevelopment(?:al)?\b|\bdevelopmental delay\b|\bintellectual disability\b|\bepilep\w*\b|\bsyndrome\b",
            text,
            flags=re.IGNORECASE,
        ):
            broader_ndd_entries.append(payload)
        else:
            generic_entries.append(payload)

    for bucket in (
        direct_autism_entries,
        autism_association_entries,
        autism_phenotype_entries,
        autism_generic_entries,
        autism_entries,
        broader_ndd_entries,
        generic_entries,
    ):
        bucket.sort(
            key=lambda row: (
                -float(row.get("entry_support_score") or 0.0),
                -int(row.get("support_rank") or 0),
                str(row.get("pmid") or ""),
                str(row.get("title") or ""),
            )
        )

    if direct_autism_entries:
        mode = "autism_direct"
    elif autism_association_entries:
        mode = "autism_association"
    elif autism_phenotype_entries or autism_generic_entries:
        mode = "autism_limited"
    elif broader_ndd_entries:
        mode = "broader_neurodevelopmental"
    elif generic_entries:
        mode = "generic_literature"
    elif autism_entries:
        mode = "autism_explicit"
    elif broader_ndd_entries:
        mode = "broader_neurodevelopmental"
    else:
        mode = "no_literature"
    return {
        "mode": mode,
        "direct_autism_entries": direct_autism_entries,
        "autism_association_entries": autism_association_entries,
        "autism_phenotype_entries": autism_phenotype_entries,
        "autism_generic_entries": autism_generic_entries,
        "autism_entries": autism_entries,
        "broader_ndd_entries": broader_ndd_entries,
        "generic_entries": generic_entries,
        "strongest_support_score": max(
            (
                float(row.get("entry_support_score") or 0.0)
                for bucket in (
                    direct_autism_entries,
                    autism_association_entries,
                    autism_phenotype_entries,
                    autism_generic_entries,
                    autism_entries,
                    broader_ndd_entries,
                    generic_entries,
                )
                for row in bucket
            ),
            default=0.0,
        ),
    }


def _task2_select_support_entries(support_profile: dict, query: str, gene: Optional[str]) -> List[dict]:
    mode = str((support_profile or {}).get("mode") or "no_literature")
    query_lower = _norm_space(query).lower()
    asks_literature = any(
        token in query_lower
        for token in ("cite", "citation", "pmid", "reference", "supporting literature", "supporting studies")
    )
    asks_strength = any(
        token in query_lower
        for token in ("associated with autism", "association evidence", "evaluate", "based on evidence")
    )
    focused_context = _env_flag("AUTISM_GPT_TASK2_FOCUSED_CONTEXT2")
    if mode == "autism_direct":
        candidates = list((support_profile or {}).get("direct_autism_entries") or [])
        margin = 0.9
    elif mode == "autism_association":
        candidates = list((support_profile or {}).get("autism_association_entries") or [])
        margin = 0.55
    elif mode == "autism_limited":
        candidates = list((support_profile or {}).get("autism_phenotype_entries") or []) + list((support_profile or {}).get("autism_generic_entries") or [])
        margin = 0.45
    elif mode == "broader_neurodevelopmental":
        candidates = list((support_profile or {}).get("broader_ndd_entries") or [])
        margin = 0.35
    elif mode == "generic_literature":
        candidates = list((support_profile or {}).get("generic_entries") or [])
        margin = 0.25
    else:
        candidates = []
        margin = 0.0
    if not candidates:
        return []
    strongest = max(float(row.get("entry_support_score") or 0.0) for row in candidates)
    threshold = strongest - margin
    qualified = []
    for row in candidates:
        score = float(row.get("entry_support_score") or 0.0)
        support_rank = int(row.get("support_rank") or row.get("support_rank_meta") or 0)
        title_gene = bool(row.get("title_gene_hit_meta"))
        body_gene = bool(row.get("body_gene_hit_meta"))
        target_gene_hit = bool(row.get("target_gene_hit_meta")) or title_gene or body_gene
        if score < threshold:
            continue
        if mode == "autism_direct" and support_rank < 3 and not target_gene_hit:
            continue
        if mode == "autism_association" and support_rank < 2 and not target_gene_hit:
            continue
        payload = dict(row)
        payload["_context_focus_score"] = score
        if focused_context and not asks_literature:
            profile = _task2_entry_indirect_profile(payload)
            focus_bonus = 0.0
            if asks_strength:
                effective_tier = profile.get("effective_tier") or ""
                if effective_tier == "direct_autism_association":
                    focus_bonus += 0.42
                elif effective_tier == "autism_association":
                    focus_bonus += 0.2
                elif effective_tier == "autism_phenotype":
                    focus_bonus -= 0.08
                elif effective_tier == "autism_generic":
                    focus_bonus -= 0.18
                elif mode == "broader_neurodevelopmental":
                    focus_bonus -= 0.24
            if profile["case_like"]:
                focus_bonus -= 0.28
            if profile["phenotype_like"] and mode != "autism_direct":
                focus_bonus -= 0.16
            if profile["broader_ndd"] and mode != "autism_direct":
                focus_bonus -= 0.14
            if profile["review_like"] and mode != "autism_direct":
                focus_bonus -= 0.06
            payload["_context_focus_score"] = round(score + focus_bonus, 4)
        qualified.append(payload)
    if not qualified:
        fallback = dict(candidates[0])
        fallback["_context_focus_score"] = float(fallback.get("entry_support_score") or 0.0)
        qualified = [fallback]
    qualified.sort(
        key=lambda row: (
            -(float(row.get("_context_focus_score") or row.get("entry_support_score") or 0.0)),
            -(float(row.get("entry_support_score") or 0.0)),
            str(row.get("pmid") or ""),
        )
    )
    if not asks_literature and len(qualified) > 1:
        if focused_context and asks_strength:
            top_score = float(qualified[0].get("_context_focus_score") or 0.0)
            second_score = float(qualified[1].get("_context_focus_score") or 0.0)
            if mode != "autism_direct" or top_score >= second_score + 0.12:
                qualified = [qualified[0]]
            else:
                qualified = [row for row in qualified if float(row.get("_context_focus_score") or 0.0) >= top_score - 0.12]
        else:
            top_score = float(qualified[0].get("entry_support_score") or 0.0)
            qualified = [row for row in qualified if float(row.get("entry_support_score") or 0.0) >= top_score - 0.3]
    return qualified[:2]


def _task2_generation_entry_from_hit(hit: dict) -> dict:
    meta = dict(hit.get("meta") or {})
    parsed = _parse_task3_doc_text(str(hit.get("text", "")))
    title = str(meta.get("title") or parsed.get("title", ""))
    abstract = str(parsed.get("abstract", ""))
    keywords = str(meta.get("keywords") or parsed.get("keywords", ""))
    publication_type = str(meta.get("publication_type") or parsed.get("publication_type", ""))
    pmid = str(meta.get("pmid") or parsed.get("pmid", ""))
    journal = str(meta.get("journal") or parsed.get("journal", ""))
    year = meta.get("year") or parsed.get("year", "")
    return {
        "doc_id": str(meta.get("doc_id") or hit.get("doc_id") or ""),
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "publication_type": publication_type,
        "pmid": pmid,
        "journal": journal,
        "year": year,
        "support_rank_meta": meta.get("support_rank") or meta.get("pubtator_support_rank") or 0,
        "effective_tier_meta": _task2_effective_evidence_tier(meta),
        "title_gene_hit_meta": bool(meta.get("pubtator_title_target_gene_hit")),
        "body_gene_hit_meta": bool(meta.get("pubtator_abstract_target_gene_hit") or meta.get("pubtator_target_gene_hit")),
        "target_gene_hit_meta": bool(meta.get("pubtator_target_gene_hit")),
    }


def _task2_generation_entry_key(entry: dict) -> str:
    pmid = _norm_space(str(entry.get("pmid", "")))
    if pmid:
        return f"pmid:{pmid}"
    doc_id = _norm_space(str(entry.get("doc_id", "")))
    if doc_id:
        return f"doc:{doc_id}"
    title = normalize_title(str(entry.get("title", "")))
    if title:
        return f"title:{title}"
    return ""


def _task2_apply_generation_evidence_priority(
    *,
    query: str,
    gene: Optional[str],
    literature_hits: Sequence[dict],
) -> List[dict]:
    if not gene or not literature_hits:
        return list(literature_hits)

    asks_literature = any(
        token in _norm_space(query).lower()
        for token in ("cite", "citation", "pmid", "reference", "supporting literature", "supporting studies")
    )
    rows = []
    entries = []
    for hit in literature_hits:
        entry = _task2_generation_entry_from_hit(hit)
        rows.append((hit, entry))
        entries.append(entry)

    support_profile = _task2_literature_support_profile(entries, gene=gene)
    selected_entries = _task2_select_support_entries(support_profile=support_profile, query=query, gene=gene)
    selected_keys = {_task2_generation_entry_key(entry) for entry in selected_entries if _task2_generation_entry_key(entry)}

    prioritized = []
    for rank, (hit, entry) in enumerate(rows):
        key = _task2_generation_entry_key(entry)
        profile = _task2_entry_indirect_profile(entry)
        score = float(entry.get("entry_support_score") or 0.0)
        bucket = str(entry.get("effective_evidence_tier") or "")
        priority = score
        if key in selected_keys:
            priority += 3.0
        if bucket == "direct_autism_association":
            priority += 1.1
        elif bucket == "autism_association":
            priority += 0.5
        elif bucket == "autism_phenotype":
            priority += 0.08
        elif bucket == "autism_generic":
            priority -= 0.08
        elif bucket == "generic":
            priority -= 0.18
        if profile["case_like"] and bucket != "direct_autism_association":
            priority -= 0.24
        if profile["phenotype_like"] and bucket not in {"direct_autism_association", "autism_association"}:
            priority -= 0.18
        if profile["broader_ndd"] and bucket not in {"direct_autism_association", "autism_association"}:
            priority -= 0.14
        if profile["review_like"] and asks_literature:
            priority += 0.06

        payload = dict(hit)
        payload_meta = dict(payload.get("meta") or {})
        payload_meta["selected_support_entry"] = bool(key and key in selected_keys)
        payload_meta["evidence_priority_bucket"] = bucket or "generic"
        payload["meta"] = payload_meta
        payload["_evidence_priority_score"] = round(priority - rank * 0.001, 4)
        prioritized.append(payload)

    prioritized.sort(
        key=lambda row: (
            -(float(row.get("_evidence_priority_score") or 0.0)),
            -(float(row.get("rerank_score") or row.get("score") or 0.0)),
            str(row.get("doc_id") or ""),
        )
    )
    return prioritized


def _task2_support_verdict(gene: str, support_profile: dict) -> dict:
    mode = str((support_profile or {}).get("mode") or "no_literature")
    selected_entries = list((support_profile or {}).get("selected_entries") or [])
    direct_count = len(selected_entries) if mode == "autism_direct" and selected_entries else len((support_profile or {}).get("direct_autism_entries") or [])
    autism_assoc_count = len(selected_entries) if mode == "autism_association" and selected_entries else len((support_profile or {}).get("autism_association_entries") or [])
    autism_limited_count = len(selected_entries) if mode == "autism_limited" and selected_entries else len((support_profile or {}).get("autism_phenotype_entries") or []) + len((support_profile or {}).get("autism_generic_entries") or [])
    broader_count = len(selected_entries) if mode == "broader_neurodevelopmental" and selected_entries else len((support_profile or {}).get("broader_ndd_entries") or [])

    if mode == "autism_direct":
        return {
            "mode": mode,
            "evidence_level": "direct_autism_explicit",
            "support_label": "direct gene-explicit autism literature",
            "evidence_line": f"Evidence level for {gene}: direct gene-explicit autism PMID-backed literature.",
            "answer_lead": (
                f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus "
                f"{direct_count} direct gene-explicit autism study/studies."
            ),
            "association_statement": (
                f"The selected context provides direct gene-explicit autism support for {gene}."
            ),
            "interpretation_line": f"Interpretation: direct autism-explicit support is present for {gene}.",
            "reasoning_line": (
                f"Direct gene-explicit autism literature was retrieved and prioritized in the references "
                f"({direct_count} direct association-focused study/studies in the selected context)."
            ),
            "literature_line": "Direct autism-explicit PMID-backed references are listed in the references.",
        }
    if mode == "autism_association":
        return {
            "mode": mode,
            "evidence_level": "autism_explicit",
            "support_label": "gene-explicit autism association literature",
            "evidence_line": f"Evidence level for {gene}: autism-explicit PMID-backed literature is present, but it is less direct than the strongest gene-specific cases.",
            "answer_lead": (
                f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus "
                f"{autism_assoc_count} gene-explicit autism association-oriented study/studies."
            ),
            "association_statement": (
                f"The selected context supports autism-explicit relevance for {gene}, although the literature is less direct than the strongest association-focused cases."
            ),
            "interpretation_line": f"Interpretation: autism-explicit support is present for {gene}, but the retrieved studies are more indirect or association-oriented than the strongest direct cases.",
            "reasoning_line": (
                f"Gene-explicit autism literature was retrieved and prioritized in the references "
                f"({autism_assoc_count} association-oriented autism study/studies in the selected context)."
            ),
            "literature_line": "Autism-explicit PMID-backed references are listed in the references.",
        }
    if mode == "autism_limited":
        return {
            "mode": mode,
            "evidence_level": "autism_limited",
            "support_label": "limited phenotype-oriented autism literature",
            "evidence_line": f"Evidence level for {gene}: limited autism-oriented literature, mostly phenotype- or case-oriented.",
            "answer_lead": (
                f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus "
                f"{autism_limited_count} limited phenotype-oriented autism study/studies."
            ),
            "association_statement": (
                f"The selected context suggests autism relevance for {gene}, but the gene-explicit autism literature is limited and more phenotype-oriented than direct association-focused."
            ),
            "interpretation_line": f"Interpretation: autism relevance is suggested for {gene}, but the retrieved evidence is limited and not strong direct association evidence.",
            "reasoning_line": (
                f"Gene-explicit autism literature was retrieved, but it is limited or phenotype-oriented "
                f"({autism_limited_count} limited autism study/studies in the selected context)."
            ),
            "literature_line": "The cited PMIDs provide limited autism-explicit support and should be interpreted cautiously.",
        }
    if mode == "broader_neurodevelopmental":
        return {
            "mode": mode,
            "evidence_level": "broader_neurodevelopmental",
            "support_label": "broader neurodevelopmental literature only",
            "evidence_line": f"Evidence level for {gene}: broader neurodevelopmental literature only; no direct autism-explicit PMID was retrieved.",
            "answer_lead": (
                f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus broader neurodevelopmental literature only."
            ),
            "association_statement": (
                f"The selected context supports broader neurodevelopmental relevance for {gene}, but it does not retrieve direct autism-explicit literature."
            ),
            "interpretation_line": f"Interpretation: the retrieved evidence supports broader neurodevelopmental relevance for {gene}, but autism-specific support remains indirect.",
            "reasoning_line": (
                f"Only broader gene-explicit neurodevelopmental literature was retrieved in the selected context "
                f"({broader_count} study/studies), so the autism-specific conclusion should remain conservative."
            ),
            "literature_line": "The cited PMIDs are broader neurodevelopmental references rather than direct autism-explicit confirmation.",
        }
    if mode == "generic_literature":
        return {
            "mode": mode,
            "evidence_level": "generic_gene_literature",
            "support_label": "generic gene-explicit literature only",
            "evidence_line": f"Evidence level for {gene}: generic gene literature only; autism-specific confirmation was not retrieved.",
            "answer_lead": (
                f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus generic gene-explicit literature only."
            ),
            "association_statement": (
                f"The selected context does not provide clear autism-specific confirmation for {gene}; the retrieved literature is generic rather than autism-explicit."
            ),
            "interpretation_line": f"Interpretation: autism-specific support for {gene} is not established in the retrieved literature.",
            "reasoning_line": "Only generic gene-explicit literature was retrieved; direct autism-explicit support was not established in the selected context.",
            "literature_line": "The cited PMIDs are generic gene-explicit references and should not be treated as autism-specific confirmation.",
        }
    return {
        "mode": mode,
        "evidence_level": "qtl_only",
        "support_label": "structured QTL evidence only",
        "evidence_line": f"Evidence level for {gene}: structured QTL evidence only; no direct gene-explicit PMID was retrieved.",
        "answer_lead": (
            f"Autism association evidence for {gene} in the selected context is limited to structured QTL findings."
        ),
        "association_statement": (
            f"Direct gene-explicit literature was not retrieved for {gene} in the current corpus, so the autism-specific association should be treated as tentative."
        ),
        "interpretation_line": f"Interpretation: the current corpus supports a tentative QTL-based signal for {gene}, not a direct literature-backed autism conclusion.",
        "reasoning_line": "Direct gene-explicit literature was not retrieved in the current corpus.",
        "literature_line": "No direct PMID-backed gene literature was retrieved for this gene in the current corpus.",
    }


_TASK2_MECHANISM_PATTERNS: Sequence[Tuple[str, str]] = (
    ("splicing", r"\bsplic"),
    ("synaptic function", r"\bsynap|\bneurotransmi"),
    ("chromatin regulation", r"\bchromatin\b|\bhistone\b|\bmethyl"),
    ("transcriptional regulation", r"\btranscript|\btranscription"),
    ("neuronal connectivity", r"\bconnectiv|\bnetwork"),
    ("neurodevelopment", r"\bneurodevelopment|\bneuronal differentiation|\bdevelopment"),
    ("ion-channel or excitability signal", r"\bchannel\b|\bexcitab|\bseizure"),
    ("immune or inflammatory signal", r"\bimmune\b|\binflamm"),
)


def _task2_mechanism_clues(entries: Sequence[dict], limit: int = 3) -> List[str]:
    combined = " ".join(
        _norm_space(
            " ".join(
                [
                    str(entry.get("title", "")),
                    str(entry.get("abstract", "")),
                    str(entry.get("keywords", "")),
                ]
            )
        )
        for entry in entries
    )
    if not combined:
        return []
    clues: List[str] = []
    for label, pattern in _TASK2_MECHANISM_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            clues.append(label)
        if len(clues) >= limit:
            break
    return clues


def _task2_study_mentions(entries: Sequence[dict], limit: int = 2) -> List[str]:
    mentions: List[str] = []
    seen: set[str] = set()
    for entry in entries:
        title = _norm_space(str(entry.get("title", "")))
        if not title or title in seen:
            continue
        seen.add(title)
        pmid = _norm_space(str(entry.get("pmid", "")))
        prefix = f"PMID {pmid}: " if pmid else ""
        mention = f"{prefix}{title}"
        mentions.append(mention)
        if len(mentions) >= limit:
            break
    return mentions


def _task2_best_support_excerpt(
    entries: Sequence[dict],
    query: str,
    gene: Optional[str],
    support_mode: Optional[str] = None,
    max_sentences: int = 1,
) -> str:
    best_excerpt = ""
    best_score = float("-inf")
    for idx, entry in enumerate(entries):
        abstract = _norm_space(str(entry.get("abstract", "")))
        if not abstract:
            continue
        excerpt = _best_support_snippet(
            text=abstract,
            query=query,
            gene=gene,
            max_sentences=max_sentences,
        )
        if not excerpt:
            continue
        lower = excerpt.lower()
        query_terms = _support_query_terms(query)
        overlap = sum(1 for term in query_terms if term in lower)
        gene_hit = bool(gene and re.search(rf"\b{re.escape(str(gene).lower())}\b", lower))
        autism_hit = any(term in lower for term in AUTISM_TERMS)
        support_score = float(entry.get("entry_support_score") or 0.0)
        score = (
            support_score * 1.8
            + overlap * 1.5
            + (1.6 if gene_hit else 0.0)
            + (1.0 if autism_hit else 0.0)
            - idx * 0.03
        )
        if support_mode in {"autism_direct", "autism_association", "autism_limited"}:
            if re.search(
                r"\bnot associated with autism\b|\bmight not be associated with autism\b|\bno significant association\b|\bnot significantly associated\b|\bdid not show association\b",
                lower,
            ):
                score -= 2.4
            if re.search(
                r"\bautism spectrum disorder\b|\bassociated with asd\b|\bassociated with autism\b|\bin asd patients\b|\bin autistic patients\b|\bin autism patients\b",
                lower,
            ):
                score += 0.8
        if score > best_score:
            best_score = score
            best_excerpt = excerpt
    return best_excerpt


def _task2_fallback_support_sentence(
    entries: Sequence[dict],
    gene: Optional[str],
    support_mode: Optional[str] = None,
) -> str:
    gene = _norm_space(gene or "")
    best_sentence = ""
    best_score = float("-inf")
    for idx, entry in enumerate(entries):
        abstract = _norm_space(str(entry.get("abstract", "")))
        if not abstract:
            continue
        support_score = float(entry.get("entry_support_score") or 0.0)
        for sent_idx, candidate in enumerate(_sentence_chunks(abstract, max_sentences=1)):
            candidate_text = _norm_space(candidate)
            if not candidate_text:
                continue
            lower = candidate_text.lower()
            gene_hit = bool(gene and re.search(rf"\b{re.escape(gene.lower())}\b", lower))
            autism_hit = any(term in lower for term in AUTISM_TERMS)
            evidence_hit = bool(
                re.search(
                    r"\b(variant|mutation|de novo|patient|proband|case|cases|cohort|identified|reported|diagnosed|asd|autism spectrum disorder)\b",
                    lower,
                )
            )
            generic_review = bool(
                re.search(
                    r"\b(review|overview|future directions|we review|this review|the review)\b",
                    lower,
                )
            )
            negative_signal = bool(
                re.search(
                    r"\bnot associated with autism\b|\bmight not be associated with autism\b|\bno significant association\b|\bnot significantly associated\b|\bdid not show association\b",
                    lower,
                )
            )
            score = support_score * 1.6
            if gene_hit:
                score += 2.8
            if autism_hit:
                score += 1.6
            if evidence_hit:
                score += 1.1
            if generic_review:
                score -= 0.8
            if len(candidate_text.split()) < 8:
                score -= 0.45
            if support_mode in {"autism_direct", "autism_association", "autism_limited"} and negative_signal:
                score -= 2.2
            score -= idx * 0.04
            score -= sent_idx * 0.02
            if score > best_score:
                best_score = score
                best_sentence = candidate_text
    return best_sentence


def _task2_mechanism_strength(entries: Sequence[dict], clues: Sequence[str]) -> Dict[str, int]:
    clue_hits: Dict[str, int] = {clue: 0 for clue in clues}
    if not entries or not clues:
        return clue_hits
    for entry in entries:
        text = _norm_space(
            " ".join(
                [
                    str(entry.get("title", "")),
                    str(entry.get("abstract", "")),
                    str(entry.get("keywords", "")),
                ]
            )
        ).lower()
        for clue in clues:
            if clue == "splicing" and re.search(r"\bsplic", text):
                clue_hits[clue] += 1
            elif clue == "synaptic function" and re.search(r"\bsynap|\bneurotransmi", text):
                clue_hits[clue] += 1
            elif clue == "chromatin regulation" and re.search(r"\bchromatin\b|\bhistone\b|\bmethyl", text):
                clue_hits[clue] += 1
            elif clue == "transcriptional regulation" and re.search(r"\btranscript|\btranscription", text):
                clue_hits[clue] += 1
            elif clue == "neuronal connectivity" and re.search(r"\bconnectiv|\bnetwork", text):
                clue_hits[clue] += 1
            elif clue == "neurodevelopment" and re.search(r"\bneurodevelopment|\bneuronal differentiation|\bdevelopment", text):
                clue_hits[clue] += 1
            elif clue == "ion-channel or excitability signal" and re.search(r"\bchannel\b|\bexcitab|\bseizure", text):
                clue_hits[clue] += 1
            elif clue == "immune or inflammatory signal" and re.search(r"\bimmune\b|\binflamm", text):
                clue_hits[clue] += 1
    return clue_hits


def _task2_mechanism_sentence(support_profile: dict, clues: Sequence[str], selected_lit_entries: Sequence[dict]) -> str:
    if not clues:
        return ""
    mode = str((support_profile or {}).get("mode") or "")
    direct_entries = (support_profile or {}).get("direct_autism_entries") or []
    assoc_entries = (support_profile or {}).get("autism_association_entries") or []
    strength = _task2_mechanism_strength(direct_entries or assoc_entries or selected_lit_entries, clues)
    strong_clues = [clue for clue in clues if strength.get(clue, 0) >= 1]
    if not strong_clues:
        return ""
    joined = ", ".join(strong_clues[:3])
    if mode == "autism_direct":
        return f"Mechanism clues in the cited studies include {joined}."
    if mode == "autism_association":
        return f"Possible mechanism clues in the cited studies include {joined}, but this mechanistic signal is less direct than the strongest autism-specific cases."
    return ""


def _task2_entry_indirect_profile(entry: dict) -> dict:
    title = _norm_space(str(entry.get("title", "")))
    abstract = _norm_space(str(entry.get("abstract", "")))
    keywords = _norm_space(str(entry.get("keywords", "")))
    publication_type = _norm_space(str(entry.get("publication_type", "")))
    combined = " ".join(part for part in (title, abstract, keywords, publication_type) if part)
    lower = combined.lower()
    case_like = bool(
        re.search(
            r"\bcase report\b|\bcase series\b|\bsyndrome\b|\btranslocation\b|\bfamily\b|\bcoffin-lowry\b|\bogden syndrome\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    phenotype_like = bool(
        re.search(
            r"\bautistic feature\b|\bautistic behaviour\b|\bfeatures of autism\b|\bsymptoms of autism\b|\bautism-like\b|\bautism spectrum disorder\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    broader_ndd = bool(
        re.search(
            r"\bneurodevelopment(?:al)?\b|\bintellectual disability\b|\bdevelopmental delay\b|\bepilep\w*\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    review_like = has_review_signal(
        publication_type=publication_type,
        title=title,
        keywords=keywords,
    )
    return {
        "case_like": case_like,
        "phenotype_like": phenotype_like,
        "broader_ndd": broader_ndd,
        "review_like": review_like,
        "support_rank": int(entry.get("support_rank") or entry.get("support_rank_meta") or 0),
        "effective_tier": _norm_space(str(entry.get("effective_evidence_tier") or entry.get("effective_tier_meta") or entry.get("evidence_tier") or "")),
        "combined_lower": lower,
    }


def _task2_build_evidence_profile(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
    selected_lit_entries: Sequence[dict],
) -> dict:
    profiles = [_task2_entry_indirect_profile(entry) for entry in selected_lit_entries]
    total = len(profiles)
    direct_count = sum(1 for row in profiles if row["effective_tier"] == "direct_autism_association")
    association_count = sum(1 for row in profiles if row["effective_tier"] == "autism_association")
    phenotype_count = sum(
        1
        for row in profiles
        if row["effective_tier"] in {"autism_phenotype", "autism_generic"}
    )
    broader_count = sum(1 for row in profiles if row["broader_ndd"])
    case_like_count = sum(1 for row in profiles if row["case_like"])
    phenotype_like_count = sum(1 for row in profiles if row["phenotype_like"])
    review_like_count = sum(1 for row in profiles if row["review_like"])
    strong_support_count = sum(1 for row in profiles if int(row.get("support_rank") or 0) >= 3)
    indirect_like_count = sum(
        1
        for row in profiles
        if row["case_like"] or row["phenotype_like"] or row["broader_ndd"]
    )
    mode = str((verdict or {}).get("mode") or (support_profile or {}).get("mode") or "no_literature")

    if direct_count > 0:
        profile_mode = "direct_autism"
    elif association_count > 0 and indirect_like_count >= max(1, math.ceil(total / 2)):
        profile_mode = "indirect_autism_association"
    elif association_count > 0:
        profile_mode = "autism_association"
    elif phenotype_count > 0:
        profile_mode = "autism_limited"
    elif broader_count > 0:
        profile_mode = "broader_neurodevelopmental"
    elif total > 0:
        profile_mode = "generic_literature"
    else:
        profile_mode = mode

    profile_parts = []
    if direct_count:
        profile_parts.append(f"{direct_count} direct")
    if association_count:
        profile_parts.append(f"{association_count} association-oriented")
    if phenotype_count:
        profile_parts.append(f"{phenotype_count} phenotype-oriented")
    if broader_count:
        profile_parts.append(f"{broader_count} broader-NDD")
    if not profile_parts:
        profile_parts.append("no PMID-backed literature selected")

    profile_line = f"Evidence profile for {gene}: " + ", ".join(profile_parts) + "."
    if profile_mode == "indirect_autism_association":
        profile_line += " The PMID-backed signal is mainly phenotype-, syndrome-, or case-oriented rather than direct association-focused."
    elif profile_mode == "broader_neurodevelopmental":
        profile_line += " The PMID-backed signal is broader neurodevelopmental rather than direct autism-explicit support."

    return {
        "mode": profile_mode,
        "total": total,
        "direct_count": direct_count,
        "association_count": association_count,
        "phenotype_count": phenotype_count,
        "broader_count": broader_count,
        "case_like_count": case_like_count,
        "phenotype_like_count": phenotype_like_count,
        "review_like_count": review_like_count,
        "strong_support_count": strong_support_count,
        "indirect_like_count": indirect_like_count,
        "profile_line": profile_line,
    }


def _task2_apply_evidence_profile_assembly(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
    selected_lit_entries: Sequence[dict],
) -> dict:
    if not _env_flag("AUTISM_GPT_TASK2_EVIDENCE_PROFILE1"):
        return {
            "verdict": verdict,
            "evidence_profile": _task2_build_evidence_profile(
                gene=gene,
                verdict=verdict,
                support_profile=support_profile,
                selected_lit_entries=selected_lit_entries,
            ),
            "assembly_mode": "disabled",
            "use_profile_reasoning": False,
            "suppress_mechanism": False,
        }

    updated = dict(verdict)
    profile = _task2_build_evidence_profile(
        gene=gene,
        verdict=verdict,
        support_profile=support_profile,
        selected_lit_entries=selected_lit_entries,
    )
    mode = profile["mode"]
    total = profile["total"]
    use_profile_reasoning = True
    suppress_mechanism = False

    if mode == "direct_autism":
        updated["answer_lead"] = (
            f"Evidence for {gene} consists of structured QTL findings plus {profile['direct_count']} direct autism-explicit PMID-backed study/studies."
        )
        updated["association_statement"] = (
            f"The selected context provides direct gene-explicit autism support for {gene}."
        )
        updated["reasoning_line"] = profile["profile_line"]
        assembly_mode = "direct_autism"
    elif mode == "indirect_autism_association":
        updated["support_label"] = "gene-explicit autism association literature"
        updated["evidence_level"] = "autism_explicit"
        updated["answer_lead"] = (
            f"Evidence for {gene} consists of structured QTL findings plus {total} gene-explicit autism-oriented PMID-backed study/studies."
        )
        updated["association_statement"] = (
            f"The selected context keeps {gene} autism-relevant, but the PMID-backed support is mainly phenotype-, syndrome-, or case-oriented rather than the strongest direct association-focused evidence."
        )
        updated["interpretation_line"] = (
            f"Interpretation: autism-relevant support is present for {gene}, but the retrieved PMIDs are mainly indirect rather than direct association-focused confirmation."
        )
        updated["reasoning_line"] = profile["profile_line"]
        updated["literature_line"] = "Autism-explicit PMID-backed references are listed in the references."
        suppress_mechanism = True
        assembly_mode = "indirect_autism_association"
    elif mode == "autism_association":
        updated["answer_lead"] = (
            f"Evidence for {gene} consists of structured QTL findings plus {profile['association_count']} gene-explicit autism association-oriented PMID-backed study/studies."
        )
        updated["association_statement"] = (
            f"The selected context supports autism-explicit relevance for {gene}, although it is less direct than the strongest gene-specific association cases."
        )
        updated["reasoning_line"] = profile["profile_line"]
        assembly_mode = "autism_association"
    elif mode == "autism_limited":
        updated["answer_lead"] = (
            f"Evidence for {gene} consists of structured QTL findings plus limited phenotype-oriented autism PMID-backed literature."
        )
        updated["association_statement"] = (
            f"The selected context suggests autism relevance for {gene}, but the PMID-backed evidence is limited and more phenotype-oriented than direct association-focused."
        )
        updated["reasoning_line"] = profile["profile_line"]
        suppress_mechanism = True
        assembly_mode = "autism_limited"
    elif mode == "broader_neurodevelopmental":
        updated["answer_lead"] = (
            f"Evidence for {gene} consists of structured QTL findings plus broader neurodevelopmental PMID-backed literature."
        )
        updated["association_statement"] = (
            f"The selected context supports broader neurodevelopmental relevance for {gene}, but it does not retrieve direct autism-explicit literature."
        )
        updated["reasoning_line"] = profile["profile_line"]
        suppress_mechanism = True
        assembly_mode = "broader_ndd"
    else:
        updated["reasoning_line"] = profile["profile_line"]
        suppress_mechanism = True
        assembly_mode = mode or "generic"

    return {
        "verdict": updated,
        "evidence_profile": profile,
        "assembly_mode": assembly_mode,
        "use_profile_reasoning": use_profile_reasoning,
        "suppress_mechanism": suppress_mechanism,
    }


def _task2_apply_claim_postcheck(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
    selected_lit_entries: Sequence[dict],
    asks_literature: bool,
    asks_mechanism: bool,
) -> dict:
    if not _env_flag("AUTISM_GPT_TASK2_CLAIM_POSTCHECK"):
        return {
            "verdict": verdict,
            "suppress_mechanism": False,
            "postcheck_mode": "disabled",
        }

    mode = str((verdict or {}).get("mode") or "")
    profiles = [_task2_entry_indirect_profile(entry) for entry in selected_lit_entries]
    if not profiles:
        return {
            "verdict": verdict,
            "suppress_mechanism": mode != "autism_direct",
            "postcheck_mode": "no_entries",
        }

    case_like_count = sum(1 for row in profiles if row["case_like"] or row["phenotype_like"])
    broader_ndd_count = sum(1 for row in profiles if row["broader_ndd"])
    review_like_count = sum(1 for row in profiles if row["review_like"])
    strongest_support_rank = max(int(row.get("support_rank") or 0) for row in profiles)
    total = len(profiles)
    mostly_indirect = case_like_count >= max(1, math.ceil(total / 2)) or (
        strongest_support_rank <= 2 and review_like_count == 0
    )
    weak_scope_mode = mode in {"autism_limited", "broader_neurodevelopmental", "generic_literature", "qtl_only"}

    updated = dict(verdict)
    suppress_mechanism = False
    postcheck_mode = "unchanged"

    if mode == "autism_association" and mostly_indirect:
        postcheck_mode = "association_scoped"
        updated["support_label"] = "gene-explicit autism-oriented literature"
        updated["evidence_level"] = "autism_oriented_indirect"
        updated["evidence_line"] = (
            f"Evidence level for {gene}: gene-explicit autism-oriented literature is present, but the selected PMIDs are mainly case-, syndrome-, or phenotype-oriented rather than direct association-focused."
        )
        updated["answer_lead"] = (
            f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus "
            f"{total} gene-explicit autism-oriented report(s)."
        )
        updated["association_statement"] = (
            f"The cited literature links {gene} to autism-related or autism-overlapping phenotypes, but it is more case- or syndrome-oriented than direct association-focused evidence."
        )
        updated["interpretation_line"] = (
            f"Interpretation: the retrieved PMIDs keep {gene} autism-relevant, but they are better read as indirect or phenotype-oriented support than as direct association-focused confirmation."
        )
        updated["reasoning_line"] = (
            f"Selected PMID-backed literature for {gene} is gene-explicit and autism-oriented, but it is mainly case-, syndrome-, or phenotype-oriented rather than direct association-focused."
        )
        updated["literature_line"] = (
            "The cited PMIDs mainly provide autism-oriented or phenotype-overlap evidence rather than direct association-focused confirmation."
        )
        suppress_mechanism = True
    elif weak_scope_mode and _env_flag("AUTISM_GPT_TASK2_SCOPED_ANSWER"):
        postcheck_mode = "weak_scope"
        updated["answer_lead"] = (
            f"Autism association evidence for {gene} in the selected context remains limited to structured QTL findings plus narrower supporting literature."
        )
        updated["association_statement"] = (
            f"The cited context keeps {gene} relevant to autism or broader neurodevelopmental phenotypes, but it does not justify a strong direct autism-association claim."
        )
        updated["reasoning_line"] = (
            f"The retrieved context for {gene} is limited in scope, so the answer should stay conservative and avoid direct-confirmation language."
        )
        if asks_literature:
            updated["literature_line"] = (
                "The cited PMIDs should be read as limited-scope supporting context rather than direct autism-specific confirmation."
            )
        suppress_mechanism = not asks_mechanism

    if broader_ndd_count and mode != "autism_direct":
        suppress_mechanism = True

    return {
        "verdict": updated,
        "suppress_mechanism": suppress_mechanism,
        "postcheck_mode": postcheck_mode,
    }


def _task2_apply_claim_verifier(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
    selected_lit_entries: Sequence[dict],
    asks_literature: bool,
) -> dict:
    if not _env_flag("AUTISM_GPT_TASK2_CLAIM_VERIFY"):
        return {
            "verdict": verdict,
            "suppress_mechanism": False,
            "claim_verify_mode": "disabled",
        }
    profiles = [_task2_entry_indirect_profile(entry) for entry in selected_lit_entries]
    if not profiles:
        return {
            "verdict": verdict,
            "suppress_mechanism": False,
            "claim_verify_mode": "no_entries",
        }
    indirect_like = sum(
        1
        for row in profiles
        if row["case_like"] or row["phenotype_like"] or row["broader_ndd"]
    )
    strongest_score = float(support_profile.get("strongest_support_score") or 0.0)
    mode = str((verdict or {}).get("mode") or "")
    if strongest_score >= 3.3 or mode not in {"autism_association", "autism_limited"}:
        return {
            "verdict": verdict,
            "suppress_mechanism": False,
            "claim_verify_mode": "unchanged",
        }
    if indirect_like < max(1, len(profiles) // 2):
        return {
            "verdict": verdict,
            "suppress_mechanism": False,
            "claim_verify_mode": "unchanged",
        }
    updated = dict(verdict)
    updated["support_label"] = "verified gene-explicit autism-oriented literature"
    updated["answer_lead"] = (
        f"Autism association evidence for {gene} in the selected context includes structured QTL findings plus gene-explicit autism-oriented literature."
    )
    updated["association_statement"] = (
        f"The retrieved PMID-backed evidence keeps {gene} relevant to autism, but the strongest support is phenotype-, case-, or syndrome-oriented rather than direct association-focused evidence."
    )
    updated["reasoning_line"] = (
        f"Claim-level verification indicates that the strongest retrieved PMIDs for {gene} are indirect autism-oriented support rather than direct association-focused evidence."
    )
    if asks_literature:
        updated["literature_line"] = (
            "The cited PMIDs are gene-explicit and autism-oriented, but their strongest support is indirect rather than direct association-focused confirmation."
        )
    return {
        "verdict": updated,
        "suppress_mechanism": True,
        "claim_verify_mode": "indirect_verified",
    }


def _task2_apply_neutral_claim_mode(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
    asks_literature: bool,
) -> dict:
    if not _env_flag("AUTISM_GPT_TASK2_NEUTRAL_CLAIM1"):
        return verdict

    updated = dict(verdict)
    mode = str((updated or {}).get("mode") or "")

    if mode == "autism_direct":
        updated["evidence_line"] = f"Evidence level for {gene}: gene-explicit autism PMID-backed literature is present in the selected context."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context includes structured QTL findings and gene-explicit autism PMID-backed literature."
        updated["association_statement"] = f"The selected context contains gene-explicit autism evidence for {gene}."
        updated["interpretation_line"] = f"Interpretation: gene-explicit autism evidence is present for {gene} in the selected context."
        updated["reasoning_line"] = f"Gene-explicit autism PMID-backed literature was retrieved for {gene} and used in the selected context."
        updated["literature_line"] = "PMID-backed autism literature is listed in the references."
    elif mode == "autism_association":
        updated["evidence_line"] = f"Evidence level for {gene}: gene-explicit autism-oriented PMID-backed literature is present in the selected context."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context includes structured QTL findings and gene-explicit autism-oriented PMID-backed literature."
        updated["association_statement"] = f"The selected context contains gene-explicit autism-oriented evidence for {gene}."
        updated["interpretation_line"] = f"Interpretation: gene-explicit autism-oriented evidence is present for {gene} in the selected context."
        updated["reasoning_line"] = f"Gene-explicit autism-oriented PMID-backed literature was retrieved for {gene} and used in the selected context."
        updated["literature_line"] = "PMID-backed autism-oriented literature is listed in the references."
    elif mode == "autism_limited":
        updated["evidence_line"] = f"Evidence level for {gene}: gene-explicit autism-oriented literature is present, but the selected corpus is narrow in scope."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context includes structured QTL findings and gene-explicit autism-oriented PMID-backed literature."
        updated["association_statement"] = f"The selected context contains autism-oriented evidence for {gene}, although the retrieved literature is narrow in scope."
        updated["interpretation_line"] = f"Interpretation: autism-oriented evidence is present for {gene}, but the selected literature is narrow in scope."
        updated["reasoning_line"] = f"Gene-explicit autism-oriented literature was retrieved for {gene}, but the selected corpus is narrow in scope."
        updated["literature_line"] = "The cited PMIDs provide autism-oriented supporting context and should be read in the scope of the selected corpus."
    elif mode == "broader_neurodevelopmental":
        updated["evidence_line"] = f"Evidence level for {gene}: broader neurodevelopmental PMID-backed literature is present; autism-specific literature was limited in the selected corpus."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context includes structured QTL findings and broader neurodevelopmental PMID-backed literature."
        updated["association_statement"] = f"The selected context contains broader neurodevelopmental evidence for {gene}; autism-specific literature was limited in the selected corpus."
        updated["interpretation_line"] = f"Interpretation: broader neurodevelopmental evidence is present for {gene}, while autism-specific literature was limited in the selected corpus."
        updated["reasoning_line"] = f"Broader neurodevelopmental PMID-backed literature was retrieved for {gene}, while autism-specific literature was limited in the selected corpus."
        updated["literature_line"] = "The cited PMIDs are broader neurodevelopmental references within the selected corpus."
    elif mode == "generic_literature":
        updated["evidence_line"] = f"Evidence level for {gene}: gene-explicit literature was retrieved, but autism-specific literature was limited in the selected corpus."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context includes structured QTL findings and gene-explicit PMID-backed literature."
        updated["association_statement"] = f"The selected context contains gene-explicit literature for {gene}; autism-specific literature was limited in the selected corpus."
        updated["interpretation_line"] = f"Interpretation: gene-explicit literature was retrieved for {gene}, but autism-specific literature was limited in the selected corpus."
        updated["reasoning_line"] = f"Gene-explicit PMID-backed literature was retrieved for {gene}, but autism-specific literature was limited in the selected corpus."
        updated["literature_line"] = "The cited PMIDs are gene-explicit references from the selected corpus."
    elif mode == "qtl_only":
        updated["evidence_line"] = f"Evidence level for {gene}: structured QTL evidence is present; direct gene-explicit literature was not retrieved in the selected corpus."
        updated["answer_lead"] = f"Evidence for {gene} in the selected context is limited to structured QTL findings."
        updated["association_statement"] = f"The selected context contains structured QTL findings for {gene}; direct gene-explicit literature was not retrieved in the selected corpus."
        updated["interpretation_line"] = f"Interpretation: the selected context contains structured QTL findings for {gene}, without direct gene-explicit literature in the selected corpus."
        updated["reasoning_line"] = f"Structured QTL findings were available for {gene}, while direct gene-explicit literature was not retrieved in the selected corpus."
        updated["literature_line"] = "No direct PMID-backed gene literature was retrieved for this gene in the selected corpus."

    if not asks_literature:
        updated["literature_line"] = updated.get("literature_line") or verdict.get("literature_line", "")

    return updated


def _task2_apply_context_lead_mode(
    *,
    gene: str,
    verdict: dict,
    support_profile: dict,
) -> dict:
    if not _env_flag("AUTISM_GPT_TASK2_CONTEXT_LEAD1"):
        return verdict

    updated = dict(verdict)
    mode = str((updated or {}).get("mode") or "")
    selected_entries = list((support_profile or {}).get("selected_entries") or [])
    direct_count = len(selected_entries) if mode == "autism_direct" and selected_entries else len((support_profile or {}).get("direct_autism_entries") or [])
    autism_assoc_count = len(selected_entries) if mode == "autism_association" and selected_entries else len((support_profile or {}).get("autism_association_entries") or [])
    autism_limited_count = len(selected_entries) if mode == "autism_limited" and selected_entries else len((support_profile or {}).get("autism_phenotype_entries") or []) + len((support_profile or {}).get("autism_generic_entries") or [])
    broader_count = len(selected_entries) if mode == "broader_neurodevelopmental" and selected_entries else len((support_profile or {}).get("broader_ndd_entries") or [])

    updated["reasoning_lead"] = f"Association evidence for autism candidate gene {gene}."

    if mode == "autism_direct":
        updated["support_label"] = "direct autism studies"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} includes structured QTL findings and "
            f"{direct_count} direct autism study/studies."
        )
    elif mode == "autism_association":
        updated["support_label"] = "autism-oriented studies"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} includes structured QTL findings and "
            f"{autism_assoc_count} autism-oriented study/studies."
        )
    elif mode == "autism_limited":
        updated["support_label"] = "limited autism-oriented studies"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} includes structured QTL findings and "
            f"{autism_limited_count} limited autism-oriented study/studies."
        )
    elif mode == "broader_neurodevelopmental":
        updated["support_label"] = "broader neurodevelopmental studies"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} includes structured QTL findings and "
            f"{broader_count} broader neurodevelopmental study/studies."
        )
    elif mode == "generic_literature":
        updated["support_label"] = "generic gene studies"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} includes structured QTL findings and generic gene studies in the selected corpus."
        )
    elif mode == "qtl_only":
        updated["support_label"] = "structured QTL findings only"
        updated["answer_lead"] = (
            f"Association evidence for autism candidate gene {gene} is limited to structured QTL findings."
        )

    return updated


def _format_task2_reference_entry(*, entry: dict, query: str, gene: str, refstyle1: bool = False) -> str:
    title = _norm_space(str(entry.get("title") or "Untitled study"))
    pmid = _norm_space(str(entry.get("pmid") or ""))
    journal = _norm_space(str(entry.get("journal") or ""))
    year = entry.get("year")
    year_text = _norm_space(str(year)) if year not in (None, "") else ""
    snippet = _best_support_snippet(
        text=entry.get("abstract", ""),
        query=query,
        gene=gene,
        max_sentences=2,
    )
    if not refstyle1:
        pmid_text = f" (PMID: {pmid})" if pmid else ""
        return f"- {title}{pmid_text}: {snippet}"

    cite_parts = [title]
    venue_parts = [part for part in (journal, year_text) if part]
    if venue_parts:
        cite_parts.append(f"{', '.join(venue_parts)}.")
    if pmid:
        cite_parts.append(f"PMID: {pmid}.")
    cite_text = " ".join(part for part in cite_parts if part).strip()
    if snippet:
        return f"- {cite_text} Evidence: {snippet}"
    return f"- {cite_text}"


def _task3_query_terms(query: str) -> List[str]:
    skip = {
        "are",
        "can",
        "common",
        "commonly",
        "current",
        "does",
        "do",
        "dominant",
        "dominate",
        "features",
        "findings",
        "give",
        "how",
        "important",
        "identify",
        "including",
        "knowledge",
        "main",
        "many",
        "most",
        "practical",
        "recent",
        "reflected",
        "reflect",
        "suggest",
        "suggests",
        "themes",
        "theme",
        "used",
        "use",
        "using",
        "young",
        "what",
        "which",
        "that",
        "about",
        "autism",
        "autistic",
        "spectrum",
        "disorder",
        "literature",
        "studies",
        "papers",
        "study",
        "research",
        "current",
        "summarize",
        "summary",
        "describe",
        "overview",
        "high",
        "level",
        "backed",
    }
    terms: List[str] = []
    for token in re.findall(r"[a-z][a-z-]{2,}", _norm_space(query).lower()):
        if token in skip or token in terms:
            continue
        terms.append(token)
    return terms


def _task3_meta_signal_text(meta: Optional[dict], parsed: Optional[dict] = None) -> str:
    meta = meta or {}
    parsed = parsed or {}
    return _norm_space(
        " ".join(
            str(part or "")
            for part in (
                meta.get("keywords", ""),
                meta.get("author_keywords", ""),
                meta.get("mesh_terms", ""),
                meta.get("supplement_excerpt", ""),
                meta.get("supplement_text", ""),
                parsed.get("keywords", ""),
                parsed.get("author_keywords", ""),
                parsed.get("mesh_terms", ""),
                parsed.get("supplement", ""),
            )
        )
    )


def _task3_decomposition_queries(query: str, expected_topic: Optional[str] = None) -> List[str]:
    base = _norm_space(query)
    if not base:
        return []
    expected_topic = expected_topic or infer_topic_from_question(query) or "general_autism"
    focus_terms = " ".join(_task3_query_terms(query)[:4])

    raw_queries: List[str] = []
    if expected_topic == "general_autism":
        raw_queries.extend(
            [
                f"{base} review overview",
                "autism review clinical features diagnosis screening",
                "autism review biomarkers neurobiology mechanisms",
                "autism review epidemiology prevalence population sex differences",
                "autism review study design methods systematic review",
            ]
        )
        if focus_terms:
            raw_queries.insert(1, f"{base} {focus_terms} review")
    elif wants_review_style(query):
        raw_queries.append(f"{base} review")

    seen = set()
    out: List[str] = []
    for item in raw_queries:
        normalized = _norm_space(item)
        key = normalized.lower()
        if not normalized or key == base.lower() or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _task3_is_broad_query(query: str, expected_topic: str) -> bool:
    q = _norm_space(query).lower()
    if expected_topic == "general_autism":
        return True
    broad_markers = (
        "high-level",
        "high level",
        "broad themes",
        "overview",
        "research directions",
        "what is known",
    )
    return wants_review_style(query) and any(marker in q for marker in broad_markers)


def _task3_route_quality(query: str, hits: Sequence[dict], expected_topic: str) -> float:
    top_hits = list(hits[:3])
    if not top_hits:
        return 0.0
    query_terms = _task3_query_terms(query)
    review_hits = 0
    topic_match = 0
    term_hits = 0
    anchor_hits = 0
    topics = []
    for hit in top_hits:
        meta = hit.get("meta") or {}
        topic = str(meta.get("topic") or "general_autism")
        topics.append(topic)
        title = str(meta.get("title") or "")
        keywords = str(meta.get("keywords") or "")
        pub_type = str(meta.get("publication_type") or "")
        text = _norm_space(" ".join([title, hit.get("text", ""), keywords, pub_type])).lower()
        if topic == expected_topic:
            topic_match += 1
        if has_review_signal(publication_type=pub_type, title=title, keywords=keywords):
            review_hits += 1
        anchor_hits += min(count_anchor_hits(text, topic_anchor_terms(expected_topic)), 4)
        term_hits += sum(term in text for term in query_terms[:6])

    distinct_topics = len({topic for topic in topics if topic})
    score = (
        topic_match * 1.0
        + review_hits * 0.35
        + distinct_topics * 0.15
        + min(term_hits, 5) * 0.12
        + min(anchor_hits, 6) * 0.08
    )
    if expected_topic == "general_autism":
        non_general_topics = len({topic for topic in topics if topic and topic != "general_autism"})
        if "general_autism" in topics:
            score += 0.25
        score += non_general_topics * 0.18
    return score


def _task3_topic_query(topic: str) -> str:
    return TASK3_TOPIC_QUERY_LABELS.get(topic, topic.replace("_", " "))


def _task3_focus_terms(query: str, expected_topic: str, entries: Sequence[dict], limit: int = 3) -> List[str]:
    combined = " ".join(
        _norm_space(
            " ".join(
                [
                    str(entry.get("title", "")),
                    str(entry.get("abstract", "")),
                    str(entry.get("keywords", "")),
                    str(entry.get("publication_type", "")),
                ]
            )
        ).lower()
        for entry in entries
    )
    if not combined:
        return []

    focus_terms: List[str] = []
    for term in _task3_query_terms(query):
        if term in combined and term not in focus_terms:
            focus_terms.append(term)
        if len(focus_terms) >= limit:
            return focus_terms

    for anchor in topic_anchor_terms(expected_topic):
        if anchor.lower() in combined and anchor not in focus_terms:
            focus_terms.append(anchor)
        if len(focus_terms) >= limit:
            break
    return focus_terms


def _task3_topic_sentence(expected_topic: str, focus_terms: Sequence[str]) -> str:
    joined = ", ".join(focus_terms[:3])
    if expected_topic == "diagnosis_screening":
        return f"Summary: The retrieved studies focus on autism assessment, screening, and diagnostic practice, especially {joined}."
    if expected_topic == "biomarkers":
        return f"Summary: The retrieved studies focus on autism biomarker work, especially {joined}."
    if expected_topic == "epidemiology_population":
        return f"Summary: The retrieved studies focus on autism population patterns, especially {joined}."
    if expected_topic == "methods_study_design":
        return f"Summary: The retrieved studies focus on autism research design and evidence synthesis, especially {joined}."
    if expected_topic == "risk_pathogenesis":
        return f"Summary: The retrieved studies focus on autism risk and pathogenesis, especially {joined}."
    if expected_topic == "clinical_features":
        return f"Summary: The retrieved studies focus on autism clinical presentation, especially {joined}."
    return f"Summary: The retrieved studies span broad autism literature, with recurring emphasis on {joined}."


def _task3_humanize_focus_term(term: str) -> str:
    raw = _norm_space(str(term or "")).lower()
    mapping = {
        "diagnos": "diagnosis",
        "screen": "screening",
        "assess": "assessment",
        "questionnaire": "questionnaires",
        "instrument": "diagnostic instruments",
        "biomarker": "biomarkers",
        "seroton": "serotonin",
        "mri": "MRI",
        "eeg": "EEG",
        "blood": "blood-based markers",
        "prevalence": "prevalence",
        "incidence": "incidence",
        "population": "population patterns",
        "cohort": "cohorts",
        "sex": "sex differences",
        "gender": "gender differences",
        "adult": "adult cohorts",
        "disparit": "disparities",
        "underdiagnos": "under-diagnosis",
        "demograph": "demographic patterns",
        "review": "reviews",
        "meta": "meta-analyses",
        "trial": "trials",
        "method": "methods",
        "study": "study design",
        "risk": "risk",
        "etiology": "etiology",
        "mechanism": "mechanisms",
        "genetic": "genetic factors",
        "prenatal": "prenatal factors",
        "environment": "environmental factors",
        "clinical": "clinical features",
        "development": "development",
        "social": "social communication",
        "behavior": "behavior",
        "language": "language",
        "sensory": "sensory features",
        "autism": "autism",
        "spectrum": "autism spectrum disorder",
        "disorder": "autism spectrum disorder",
    }
    return mapping.get(raw, raw.replace("-", " "))


def _task3_readable_topic(topic: str) -> str:
    mapping = {
        "clinical_features": "clinical features",
        "diagnosis_screening": "diagnosis and screening",
        "biomarkers": "biomarkers",
        "epidemiology_population": "population patterns",
        "methods_study_design": "study design and evidence synthesis",
        "risk_pathogenesis": "risk and pathogenesis",
        "general_autism": "broad autism topics",
    }
    return mapping.get(str(topic or ""), str(topic or "").replace("_", " "))


def _task3_focus_phrase(focus_terms: Sequence[str], limit: int = 3) -> str:
    phrases: List[str] = []
    seen: set[str] = set()
    for term in focus_terms[:limit]:
        phrase = _task3_humanize_focus_term(term)
        key = re.sub(r"[^a-z0-9]+", " ", _norm_space(phrase).lower()).strip()
        if not phrase or key in seen:
            continue
        seen.add(key)
        phrases.append(phrase)
    return ", ".join(phrases)


def _task3_context_summary_sentence(expected_topic: str, focus_terms: Sequence[str]) -> str:
    joined = _task3_focus_phrase(focus_terms)
    if expected_topic == "diagnosis_screening":
        base = "Summary: The selected studies describe autism screening and diagnostic assessment."
    elif expected_topic == "biomarkers":
        base = "Summary: The selected studies describe autism biomarker findings."
    elif expected_topic == "epidemiology_population":
        base = "Summary: The selected studies describe autism prevalence and population patterns."
    elif expected_topic == "methods_study_design":
        base = "Summary: The selected studies describe autism study design and evidence-synthesis methods."
    elif expected_topic == "risk_pathogenesis":
        base = "Summary: The selected studies describe autism risk and pathogenesis."
    elif expected_topic == "clinical_features":
        base = "Summary: The selected studies describe autism clinical features."
    else:
        base = "Summary: The selected studies cover broad autism topics."
    if joined:
        if base.endswith("."):
            base = base[:-1]
        return f"{base}, including {joined}."
    return base


def _task3_simple_summary(expected_topic: str, entries: Sequence[dict]) -> str:
    if expected_topic == "diagnosis_screening":
        return "Summary: The selected studies describe autism screening and diagnostic assessment."
    if expected_topic == "biomarkers":
        return "Summary: The selected studies describe autism biomarker findings."
    if expected_topic == "epidemiology_population":
        return "Summary: The selected studies describe autism prevalence and population patterns."
    if expected_topic == "methods_study_design":
        return "Summary: The selected studies describe autism study design and evidence-synthesis methods."
    if expected_topic == "risk_pathogenesis":
        return "Summary: The selected studies describe autism risk and pathogenesis."
    if expected_topic == "clinical_features":
        return "Summary: The selected studies describe autism clinical features."

    seen_topics: List[str] = []
    for entry in entries:
        inferred = infer_task3_topic(
            title=entry.get("title", ""),
            abstract=entry.get("abstract", ""),
            keywords=entry.get("keywords", ""),
            publication_type=entry.get("publication_type", ""),
        )
        if inferred and inferred != "general_autism" and inferred not in seen_topics:
            seen_topics.append(inferred)
        if len(seen_topics) >= 3:
            break
    if seen_topics:
        readable = ", ".join(_task3_readable_topic(topic) for topic in seen_topics[:3])
        return f"Summary: The selected studies cover broad autism topics across {readable}."
    return "Summary: The selected studies cover broad autism topics."


def _task3_entry_text(entries: Sequence[dict]) -> str:
    return _norm_space(
        " ".join(
            " ".join(
                str(part or "")
                for part in (
                    entry.get("title", ""),
                    entry.get("abstract", ""),
                    entry.get("keywords", ""),
                    entry.get("publication_type", ""),
                )
            )
            for entry in entries
        )
    ).lower()


def _task3_evidence_anchor_terms(expected_topic: str, entries: Sequence[dict], limit: int = 3) -> List[str]:
    combined = _task3_entry_text(entries)
    if not combined:
        return []
    terms: List[str] = []
    for anchor in topic_anchor_terms(expected_topic):
        raw = str(anchor or "").lower()
        if not raw or raw in {"autism", "spectrum", "disorder"}:
            continue
        if raw in combined and raw not in terms:
            terms.append(raw)
        if len(terms) >= limit:
            break
    return terms


def _task3_query_evidence_terms(query: str, entries: Sequence[dict], limit: int = 2) -> List[str]:
    combined = _task3_entry_text(entries)
    if not combined:
        return []
    blocked = {
        "and",
        "are",
        "can",
        "how",
        "the",
        "what",
        "which",
        "with",
        "about",
        "autism",
        "autistic",
        "spectrum",
        "disorder",
        "literature",
        "research",
        "studies",
        "study",
        "papers",
        "paper",
        "current",
        "recent",
        "summary",
        "summarize",
        "overview",
        "themes",
        "theme",
        "broad",
        "high",
        "level",
        "knowledge",
    }
    out: List[str] = []
    for token in re.findall(r"[a-z][a-z-]{3,}", _norm_space(query).lower()):
        if token in blocked or token in out:
            continue
        if token in combined:
            out.append(token)
        if len(out) >= limit:
            break
    return out


def _task3_topic_coverage_summary(query: str, entries: Sequence[dict]) -> str:
    seen_topics: List[str] = []
    for entry in entries:
        inferred = infer_task3_topic(
            title=entry.get("title", ""),
            abstract=entry.get("abstract", ""),
            keywords=entry.get("keywords", ""),
            publication_type=entry.get("publication_type", ""),
        )
        if inferred and inferred != "general_autism" and inferred not in seen_topics:
            seen_topics.append(inferred)
        if len(seen_topics) >= 3:
            break
    readable = ", ".join(_task3_readable_topic(topic) for topic in seen_topics[:3])
    q = _norm_space(query).lower()
    if "research direction" in q:
        lead = "autism research directions"
    elif "knowledge" in q:
        lead = "autism knowledge"
    elif "theme" in q:
        lead = "broad autism themes"
    else:
        lead = "broad autism topics"
    if readable:
        return f"Summary: The selected studies describe {lead} across {readable}."
    return f"Summary: The selected studies describe {lead}."


def _task3_guarded_summary(query: str, expected_topic: str, entries: Sequence[dict]) -> str:
    if expected_topic == "general_autism":
        return _task3_topic_coverage_summary(query=query, entries=entries)

    base = _task3_simple_summary(expected_topic=expected_topic, entries=entries)
    anchor_terms = _task3_evidence_anchor_terms(expected_topic=expected_topic, entries=entries, limit=3)
    query_terms = _task3_query_evidence_terms(query=query, entries=entries, limit=2)
    combined_terms: List[str] = []
    for term in list(query_terms) + list(anchor_terms):
        phrase = _task3_humanize_focus_term(term)
        key = re.sub(r"[^a-z0-9]+", " ", _norm_space(phrase).lower()).strip()
        if not phrase or key in combined_terms:
            continue
        if key and key not in re.sub(r"[^a-z0-9]+", " ", base.lower()):
            combined_terms.append(key)
    if not combined_terms:
        return base
    readable_terms = ", ".join(term.replace(" and ", " / ") for term in combined_terms[:3])
    if base.endswith("."):
        base = base[:-1]
    return f"{base}, including {readable_terms}."


def _maybe_prepend_task3_ar_lead(task: str, query: str, answer: str) -> str:
    if task != "task3" or not _env_flag("AUTISM_GPT_TASK3_ARLEAD1"):
        return answer
    clean_query = _norm_space(query or "").rstrip(".")
    if not clean_query:
        return answer
    raw = str(answer or "")
    first_line = raw.splitlines()[0].strip() if raw.splitlines() else ""
    if first_line.lower().startswith("question addressed:"):
        return raw
    return f"Question addressed: {clean_query}\n{raw}".strip()


def _task3_summary_from_entries(query: str, expected_topic: str, entries: Sequence[dict], style: str = "default") -> str:
    focus_terms = _task3_focus_terms(query, expected_topic, entries, limit=3)
    if focus_terms:
        if style == "context":
            return _task3_context_summary_sentence(expected_topic, focus_terms)
        return _task3_topic_sentence(expected_topic, focus_terms)

    seen_topics: List[str] = []
    for entry in entries:
        inferred = infer_task3_topic(
            title=entry.get("title", ""),
            abstract=entry.get("abstract", ""),
            keywords=entry.get("keywords", ""),
            publication_type=entry.get("publication_type", ""),
        )
        if inferred and inferred not in seen_topics:
            seen_topics.append(inferred)
        if len(seen_topics) >= 3:
            break

    if seen_topics:
        readable = ", ".join(topic.replace("_", " ") for topic in seen_topics)
        if style == "context":
            return f"Summary: The selected studies cover {readable} in autism."
        return f"Summary: The retrieved studies span {readable} in the autism literature."
    if style == "context":
        return "Summary: The selected studies provide an autism literature overview."
    return "Summary: The retrieved studies provide a literature-backed autism overview."


def _task3_bottom_line(query: str, expected_topic: str, entries: Sequence[dict]) -> str:
    focus_terms = _task3_focus_terms(query, expected_topic, entries, limit=3)
    if not focus_terms:
        return ""
    joined = ", ".join(focus_terms)
    if expected_topic == "general_autism":
        return f"Bottom line: The selected papers collectively emphasize {joined} rather than a single narrow autism theme."
    return f"Bottom line: In the selected studies, the clearest repeated emphasis is on {joined}."


def _task1_query_mode(query: str) -> str:
    q = _norm_space(query).lower()
    if any(token in q for token in ("gene function", "function summary", "omim", "gene context")):
        return "function"
    if any(token in q for token in ("coordinate", "grch", "region")):
        return "coordinates"
    if any(token in q for token in ("qtl evidence", "p-value", "p value", "statistical", "best reported")):
        return "qtl"
    return "general"


def _task1_reasoning_text(query: str, fields: dict) -> str:
    q = _norm_space(query).lower()
    coords = fields.get("coordinates", "")
    if "length" in q or "bp" in q:
        asm = "GRCh38" if "grch38" in q else "GRCh37" if "grch37" in q else "T2T" if "t2t" in q else None
        pattern = rf"{asm}:(\d+)-(\d+)" if asm else r"(GRCh37|GRCh38|T2T):(\d+)-(\d+)"
        match = re.search(pattern, coords)
        if match:
            if asm:
                start = int(match.group(1))
                end = int(match.group(2))
                label = asm
            else:
                label = match.group(1)
                start = int(match.group(2))
                end = int(match.group(3))
            length = end - start + 1
            return f"{label} length = {end} - {start} + 1 = {length} bp."
    return "N/A"


def _task1_answer_text(query: str, fields: dict) -> str:
    gene = fields.get("gene") or _query_focus_gene(query) or "The gene"
    coords = fields.get("coordinates") or "Not specified in the provided context."
    eqtl_regions = fields.get("eqtl_regions") or "Not specified in the provided context."
    sqtl_regions = fields.get("sqtl_regions") or "Not specified in the provided context."
    eqtl_p = fields.get("eqtl_p") or "Not specified in the provided context."
    sqtl_p = fields.get("sqtl_p") or "Not specified in the provided context."
    gene_context = fields.get("gene_context") or "Not specified in the provided context."
    omim = fields.get("omim") or "Not specified in the provided context."

    mode = _task1_query_mode(query)
    if mode == "qtl":
        return _norm_space(
            f"For {gene}, the best reported eQTL p-value is {eqtl_p} and the best reported sQTL p-value is {sqtl_p}. "
            f"eQTL regions: {eqtl_regions}. sQTL regions: {sqtl_regions}."
        )
    if mode == "function":
        return _norm_space(
            f"Gene context for {gene}: {gene_context}. OMIM function: {omim}. "
            f"eQTL regions: {eqtl_regions}. sQTL regions: {sqtl_regions}."
        )
    if mode == "coordinates":
        return _norm_space(
            f"{gene} has coordinates {coords}. The reported eQTL regions are {eqtl_regions}. "
            f"The reported sQTL regions are {sqtl_regions}."
        )
    return _norm_space(
        f"{gene} has coordinates {coords}. Gene context: {gene_context}. OMIM function: {omim}. "
        f"Best eQTL p-value: {eqtl_p}. Best sQTL p-value: {sqtl_p}. "
        f"eQTL regions: {eqtl_regions}. sQTL regions: {sqtl_regions}."
    )


def _task1_reference_text(query: str, fields: dict) -> str:
    coords = fields.get("coordinates") or "Not specified in the provided context."
    eqtl_regions = fields.get("eqtl_regions") or "Not specified in the provided context."
    sqtl_regions = fields.get("sqtl_regions") or "Not specified in the provided context."
    eqtl_p = fields.get("eqtl_p") or "Not specified in the provided context."
    sqtl_p = fields.get("sqtl_p") or "Not specified in the provided context."
    gene_context = fields.get("gene_context") or "Not specified in the provided context."
    omim = fields.get("omim") or "Not specified in the provided context."

    mode = _task1_query_mode(query)
    if mode == "qtl":
        return (
            f"Best eQTL p-value: {eqtl_p}.\n"
            f"Best sQTL p-value: {sqtl_p}.\n"
            f"eQTL regions: {eqtl_regions}.\n"
            f"sQTL regions: {sqtl_regions}."
        )
    if mode == "function":
        return (
            f"Gene context: {gene_context}.\n"
            f"OMIM function: {omim}.\n"
            f"eQTL regions: {eqtl_regions}.\n"
            f"sQTL regions: {sqtl_regions}."
        )
    if mode == "coordinates":
        return (
            f"Coordinates: {coords}.\n"
            f"eQTL regions: {eqtl_regions}.\n"
            f"sQTL regions: {sqtl_regions}."
        )
    return (
        f"Coordinates: {coords}.\n"
        f"Gene context: {gene_context}.\n"
        f"OMIM function: {omim}.\n"
        f"Best eQTL p-value: {eqtl_p}.\n"
        f"Best sQTL p-value: {sqtl_p}.\n"
        f"eQTL regions: {eqtl_regions}.\n"
        f"sQTL regions: {sqtl_regions}."
    )


def _format_task1_structured_answer(query: str, context_text: str) -> str:
    fields = _parse_task1_doc_text(context_text)
    reasoning = _task1_reasoning_text(query=query, fields=fields)
    answer = _task1_answer_text(query=query, fields=fields)
    reference = _task1_reference_text(query=query, fields=fields)
    return (
        "<reasoning>\n"
        f"{reasoning}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"{answer}\n"
        "</answer>\n"
        "<reference>\n"
        f"{reference}\n"
        "</reference>"
    )


def _is_low_quality_generation(task: str, answer: Optional[str], query: Optional[str] = None) -> bool:
    text = _norm_space(answer or "")
    if not text:
        return True
    bad_markers = [
        "<your answer>",
        "<your explanation>",
        "</your response>",
    ]
    if any(marker in text.lower() for marker in bad_markers):
        return True
    if len(text) < 25:
        return True
    toks = re.findall(r"\b\w+\b", text.lower())
    if len(toks) >= 30:
        unique_ratio = len(set(toks)) / max(1, len(toks))
        if unique_ratio < 0.35:
            return True
    if task == "task2":
        lower = text.lower()
        if "<reasoning>" not in lower or "<answer>" not in lower or "</answer>" not in lower:
            return True
        if "{pmid:" in lower and "<reference>" not in lower:
            return True
        if "title:" in lower and "<reference>" not in lower:
            return True
    if task == "task3":
        if not text.endswith((".", "!", "?")):
            return True
        if re.match(r"^[,.;:)\]]", text):
            return True
        expected_topic = infer_topic_from_question(query or "") if query else None
        if expected_topic and expected_topic != "general_autism":
            anchors = topic_anchor_terms(expected_topic)
            if anchors and count_anchor_hits(text, anchors) == 0:
                return True
        if text.count("- ") >= 3 and "summary:" not in text.lower() and "bottom line:" not in text.lower():
            return True
    return False


def _heuristic_answer(task: str, query: str, generation_contexts: Sequence[str]) -> str:
    if not generation_contexts:
        return ""
    if task == "task1":
        t1 = _parse_task1_doc_text(generation_contexts[0])
        gene = t1.get("gene") or _query_focus_gene(query) or "The gene"
        return _norm_space(
            f"{gene} has coordinates {t1.get('coordinates', 'Not specified')}. "
            f"The reported eQTL regions are {t1.get('eqtl_regions', 'None')}. "
            f"The reported sQTL regions are {t1.get('sqtl_regions', 'None')}."
        )
    if task == "task2":
        t2 = _parse_task2_doc_text(generation_contexts[0])
        t1 = _parse_task1_doc_text(generation_contexts[1]) if len(generation_contexts) > 1 else {}
        lit_entries = []
        for text in generation_contexts[2:]:
            parsed = _parse_task3_doc_text(text)
            if parsed.get("title"):
                lit_entries.append(parsed)
        gene = t2.get("gene") or t1.get("gene") or _query_focus_gene(query) or "this gene"
        support_profile = _task2_literature_support_profile(lit_entries, gene=gene)
        selected_lit_entries = _task2_select_support_entries(
            support_profile=support_profile,
            query=query,
            gene=gene,
        ) or (
            support_profile["direct_autism_entries"]
            or support_profile["autism_association_entries"]
            or support_profile["autism_phenotype_entries"]
            or support_profile["autism_generic_entries"]
            or support_profile["broader_ndd_entries"]
            or support_profile["generic_entries"]
            or lit_entries
        )
        support_profile = dict(support_profile, selected_entries=selected_lit_entries)
        support_mode = support_profile["mode"]
        verdict = _task2_support_verdict(gene=gene, support_profile=support_profile)
        snippet_calibrated = _env_flag("AUTISM_GPT_TASK2_SNIPPETCAL1")
        snippet_calibrated2 = _env_flag("AUTISM_GPT_TASK2_SNIPPETCAL2")
        q = _norm_space(query).lower()
        asks_literature = any(token in q for token in ("cite", "citation", "pmid", "reference", "supporting literature", "supporting studies"))
        asks_region = any(token in q for token in ("brain-region", "brain region", "region context", "where"))
        asks_mechanism = any(token in q for token in ("mechanism", "mechanistic", "pathway", "function clue", "mechanism clues"))
        asks_gene_context = any(token in q for token in ("omim", "function", "gene context", "coordinate", "coordinates", "ensembl", "genomic span"))
        stats_text = t2.get("stats", "no statistical minima reported")
        eqtl_regions = t2.get("eqtl_regions", "None")
        sqtl_regions = t2.get("sqtl_regions", "None")
        gene_context = t1.get("gene_context", "N/A")
        omim = t1.get("omim", t2.get("omim", "N/A"))
        postcheck = _task2_apply_claim_postcheck(
            gene=gene,
            verdict=verdict,
            support_profile=support_profile,
            selected_lit_entries=selected_lit_entries,
            asks_literature=asks_literature,
            asks_mechanism=asks_mechanism,
        )
        verdict = postcheck["verdict"]
        claim_verify = _task2_apply_claim_verifier(
            gene=gene,
            verdict=verdict,
            support_profile=support_profile,
            selected_lit_entries=selected_lit_entries,
            asks_literature=asks_literature,
        )
        verdict = claim_verify["verdict"]
        evidence_assembly = _task2_apply_evidence_profile_assembly(
            gene=gene,
            verdict=verdict,
            support_profile=support_profile,
            selected_lit_entries=selected_lit_entries,
        )
        verdict = evidence_assembly["verdict"]
        verdict = _task2_apply_neutral_claim_mode(
            gene=gene,
            verdict=verdict,
            support_profile=support_profile,
            asks_literature=asks_literature,
        )
        verdict = _task2_apply_context_lead_mode(
            gene=gene,
            verdict=verdict,
            support_profile=support_profile,
        )
        mechanism_clues = _task2_mechanism_clues(selected_lit_entries)
        study_mentions = _task2_study_mentions(selected_lit_entries, limit=2)
        best_support_excerpt = _task2_best_support_excerpt(
            selected_lit_entries,
            query=query,
            gene=gene,
            support_mode=support_mode,
            max_sentences=1 if snippet_calibrated else 2,
        )
        fallback_support_sentence = _task2_fallback_support_sentence(
            selected_lit_entries,
            gene=gene,
            support_mode=support_mode,
        )
        refstyle1 = _env_flag("AUTISM_GPT_TASK2_REFSTYLE1")
        reference_notice = "Supporting studies are listed in the references." if refstyle1 else None
        ref_lines = [
            f"Statistical minima: {stats_text}.",
            f"eQTL evidence regions: {eqtl_regions}.",
            f"sQTL evidence regions: {sqtl_regions}.",
        ]
        seen = set()
        cited_pmids = []
        cited_titles = []
        for entry in selected_lit_entries[:2]:
            title = entry.get("title") or "Untitled study"
            if title in seen:
                continue
            seen.add(title)
            pmid = entry.get("pmid", "")
            if pmid:
                cited_pmids.append(pmid)
            cited_titles.append(title)
            ref_lines.append(
                _format_task2_reference_entry(
                    entry=entry,
                    query=query,
                    gene=gene,
                    refstyle1=refstyle1,
                )
            )
        if asks_literature and not refstyle1:
            ref_lines.append(verdict["literature_line"])
        reference_block = "\n<reference>\n" + "\n".join(ref_lines) + "\n</reference>" if ref_lines else ""
        if evidence_assembly.get("use_profile_reasoning"):
            reasoning_lines = [
                verdict["reasoning_line"],
                f"Statistical minima: {stats_text}.",
                f"eQTL evidence regions: {eqtl_regions}.",
                f"sQTL evidence regions: {sqtl_regions}.",
            ]
        else:
            reasoning_lines = [
                verdict.get("reasoning_lead") or f"Autism association evidence for {gene}: {verdict['support_label']}.",
                f"Statistical minima: {stats_text}.",
                f"eQTL evidence regions: {eqtl_regions}.",
                f"sQTL evidence regions: {sqtl_regions}.",
            ]
        if asks_gene_context and (gene_context != "N/A" or omim != "N/A"):
            reasoning_lines.append(f"Gene context: {gene_context}. OMIM function: {omim}.")
        if (snippet_calibrated or snippet_calibrated2) and best_support_excerpt:
            reasoning_lines.append(f"Key supporting excerpt: {best_support_excerpt}.")
        elif study_mentions:
            reasoning_lines.append(f"Selected PMID-backed studies: {'; '.join(study_mentions)}.")
        if (
            postcheck.get("postcheck_mode") not in {"disabled", "unchanged"}
            and not evidence_assembly.get("use_profile_reasoning")
        ):
            reasoning_lines.append(verdict["reasoning_line"])
        elif (
            claim_verify.get("claim_verify_mode") not in {"disabled", "unchanged"}
            and not evidence_assembly.get("use_profile_reasoning")
        ):
            reasoning_lines.append(verdict["reasoning_line"])

        claim_clip_mode = "none"
        if _env_flag("AUTISM_GPT_TASK2_CLAIM_CLIP4"):
            claim_clip_mode = "softplus2"
        elif _env_flag("AUTISM_GPT_TASK2_CLAIM_CLIP3"):
            claim_clip_mode = "softplus"
        elif _env_flag("AUTISM_GPT_TASK2_CLAIM_CLIP2"):
            claim_clip_mode = "soft"
        elif _env_flag("AUTISM_GPT_TASK2_CLAIM_CLIP1"):
            claim_clip_mode = "hard"
        claim_clipped = claim_clip_mode != "none"
        asks_statistics = any(token in q for token in ("statistical", "statistics", "p-value", "p value", "qtl", "eqtl", "sqtl"))

        if claim_clip_mode == "hard":
            reasoning_lines = [verdict["reasoning_line"]]
            if best_support_excerpt:
                reasoning_lines.append(f"Key supporting excerpt: {best_support_excerpt}.")
            if asks_statistics:
                reasoning_lines.append(f"Statistical minima: {stats_text}.")
            if asks_region:
                reasoning_lines.append(f"eQTL evidence regions: {eqtl_regions}.")
                reasoning_lines.append(f"sQTL evidence regions: {sqtl_regions}.")
            if asks_literature and cited_pmids:
                reasoning_lines.append(reference_notice or f"Selected PMID-backed references: {', '.join(cited_pmids)}.")

        answer_parts = []
        if claim_clip_mode == "hard":
            answer_parts = [verdict["answer_lead"], verdict["association_statement"]]
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            if asks_statistics:
                answer_parts.append(f"Statistical minima: {stats_text}.")
            if asks_region:
                answer_parts.append(f"Brain-region context: eQTL regions {eqtl_regions}; sQTL regions {sqtl_regions}.")
            if asks_literature:
                if cited_pmids:
                    answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
                else:
                    answer_parts.append(reference_notice or verdict["literature_line"])
        elif claim_clip_mode == "soft":
            reasoning_lines = [verdict["reasoning_line"]]
            if best_support_excerpt:
                reasoning_lines.append(f"Key supporting excerpt: {best_support_excerpt}.")
            elif study_mentions:
                reasoning_lines.append(f"Selected PMID-backed study: {study_mentions[0]}.")
            reasoning_lines.append(f"Statistical minima: {stats_text}.")
            reasoning_lines.append(f"eQTL evidence regions: {eqtl_regions}.")
            reasoning_lines.append(f"sQTL evidence regions: {sqtl_regions}.")
            if asks_literature and cited_pmids:
                reasoning_lines.append(reference_notice or f"Selected PMID-backed references: {', '.join(cited_pmids)}.")

            answer_parts = [verdict["answer_lead"], verdict["association_statement"]]
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            elif study_mentions:
                answer_parts.append(f"Representative PMID-backed study: {study_mentions[0]}.")
            answer_parts.append(f"Statistical minima: {stats_text}.")
            answer_parts.append(f"Brain-region context: eQTL regions {eqtl_regions}; sQTL regions {sqtl_regions}.")
            if asks_literature:
                if cited_pmids:
                    answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
                else:
                    answer_parts.append(reference_notice or verdict["literature_line"])
        elif claim_clip_mode == "softplus":
            answer_parts = [verdict["answer_lead"], verdict["association_statement"]]
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            elif study_mentions:
                answer_parts.append(f"Representative PMID-backed study: {study_mentions[0]}.")
            answer_parts.append(f"Statistical minima: {stats_text}.")
            if asks_region:
                answer_parts.append(f"Brain-region context: eQTL regions {eqtl_regions}; sQTL regions {sqtl_regions}.")
            if asks_literature:
                if cited_pmids:
                    answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
                else:
                    answer_parts.append(reference_notice or verdict["literature_line"])
        elif claim_clip_mode == "softplus2":
            answer_parts = [verdict["answer_lead"], verdict["association_statement"]]
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            elif fallback_support_sentence:
                answer_parts.append(f"Representative cited evidence: {fallback_support_sentence}.")
            elif study_mentions:
                answer_parts.append(f"Representative PMID-backed study: {study_mentions[0]}.")
            answer_parts.append(f"Statistical minima: {stats_text}.")
            if asks_region:
                answer_parts.append(f"Brain-region context: eQTL regions {eqtl_regions}; sQTL regions {sqtl_regions}.")
            if asks_literature:
                if cited_pmids:
                    answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
                else:
                    answer_parts.append(reference_notice or verdict["literature_line"])
        elif snippet_calibrated2:
            answer_parts = [verdict["answer_lead"], verdict["association_statement"]]
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            answer_parts.append(f"Statistical minima: {stats_text}.")
        elif snippet_calibrated:
            answer_parts.append(verdict["association_statement"])
            if best_support_excerpt:
                answer_parts.append(f"Key cited evidence: {best_support_excerpt}.")
            answer_parts.append(f"Statistical minima: {stats_text}.")
        else:
            answer_parts = [
                verdict["answer_lead"],
                verdict["association_statement"],
                f"Statistical minima: {stats_text}.",
            ]
        if not claim_clipped and (asks_region or True):
            answer_parts.append(f"Brain-region context: eQTL regions {eqtl_regions}; sQTL regions {sqtl_regions}.")
        allow_mechanism = verdict["mode"] == "autism_direct" or (
            verdict["mode"] == "autism_association"
            and float(support_profile.get("strongest_support_score") or 0.0) >= 3.2
            and len(selected_lit_entries) >= 1
        )
        if (
            postcheck.get("suppress_mechanism")
            or claim_verify.get("suppress_mechanism")
            or evidence_assembly.get("suppress_mechanism")
            or claim_clipped
        ):
            allow_mechanism = False
        if asks_mechanism and mechanism_clues and allow_mechanism:
            answer_parts.append(f"Mechanism clues visible in the cited studies include {', '.join(mechanism_clues)}.")
        if study_mentions and not claim_clipped:
            if asks_literature:
                answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
        elif asks_literature and not claim_clipped:
            if cited_pmids:
                answer_parts.append(reference_notice or f"PMID-backed references: {', '.join(cited_pmids)}.")
            else:
                answer_parts.append(reference_notice or verdict["literature_line"])
        if _env_flag("AUTISM_GPT_TASK2_ARLEAD1"):
            clean_query = _norm_space(query or "").rstrip(".")
            if clean_query:
                answer_parts.insert(0, f"Question addressed: {clean_query}.")
        return (
            "<reasoning>\n"
            + "\n".join(reasoning_lines)
            + "\n"
            + "</reasoning>\n"
            "<answer>\n"
            + " ".join(answer_parts)
            + "\n"
            "</answer>"
            f"{reference_block}"
        )
    expected_topic = infer_topic_from_question(query) or "general_autism"
    anchors = topic_anchor_terms(expected_topic)
    parsed_entries = [_parse_task3_doc_text(text) for text in generation_contexts[:3]]
    context_summary = _env_flag("AUTISM_GPT_TASK3_CONTEXTSUMMARY1")
    simple_summary = _env_flag("AUTISM_GPT_TASK3_SIMPLESUMMARY1")
    guarded_summary = _env_flag("AUTISM_GPT_TASK3_SUMMARYGUARD1")
    summary_line = {
        "clinical_features": "Summary: Across the supplied studies, autism literature emphasizes developmental presentation, social-communication differences, repetitive behaviors, and sensory or language-related features.",
        "diagnosis_screening": "Summary: Across the supplied studies, autism diagnosis and screening work emphasizes early identification, validated instruments, developmental history, and multidisciplinary assessment.",
        "biomarkers": "Summary: Across the supplied studies, autism biomarker research highlights heterogeneous signals from imaging, electrophysiology, blood-based, and molecular measures rather than a single validated marker.",
        "epidemiology_population": "Summary: Across the supplied studies, autism epidemiology highlights prevalence differences, sex-related under-recognition, and population heterogeneity across cohorts.",
        "methods_study_design": "Summary: Across the supplied studies, autism research relies on reviews, cohort studies, intervention trials, and increasingly data-driven analytic methods.",
        "risk_pathogenesis": "Summary: Across the supplied studies, autism risk and pathogenesis are presented as multifactorial, spanning genetic, prenatal, environmental, immune, and neurodevelopmental mechanisms.",
        "general_autism": "Summary: Across the supplied studies, the literature reflects broad autism knowledge spanning clinical presentation, diagnosis, biology, population patterns, and research methods.",
    }.get(expected_topic, "Summary: The supplied studies summarize broad autism knowledge from the literature.")
    if guarded_summary:
        dynamic_summary = _task3_guarded_summary(query=query, expected_topic=expected_topic, entries=parsed_entries)
        if dynamic_summary:
            summary_line = dynamic_summary
    elif simple_summary:
        dynamic_summary = _task3_simple_summary(expected_topic=expected_topic, entries=parsed_entries)
        if dynamic_summary:
            summary_line = dynamic_summary
    elif context_summary:
        dynamic_summary = _task3_summary_from_entries(
            query=query,
            expected_topic=expected_topic,
            entries=parsed_entries,
            style="context",
        )
        if dynamic_summary:
            summary_line = dynamic_summary
    elif expected_topic == "general_autism" and _env_flag("AUTISM_GPT_TASK3_GENERAL_SUMMARY2"):
        dynamic_summary = _task3_summary_from_entries(query=query, expected_topic=expected_topic, entries=parsed_entries)
        if dynamic_summary:
            summary_line = dynamic_summary
    evidence_lines = ["Evidence highlights:"]
    seen_titles = set()
    for parsed in parsed_entries:
        title = parsed.get("title") or "Untitled study"
        if title in seen_titles:
            continue
        seen_titles.add(title)
        abstract = parsed.get("abstract", "")
        pub_type = parsed.get("publication_type", "")
        pmid = parsed.get("pmid", "")
        snippet = _best_support_snippet(
            text=abstract,
            query=query,
            topic=expected_topic,
            max_sentences=2,
        )
        suffix = f" [{pub_type}]" if pub_type else ""
        pmid_text = f" (PMID: {pmid})" if pmid else ""
        evidence_lines.append(f"- {title}{pmid_text}{suffix}: {snippet}")

    anchor_hint = ""
    if not context_summary and not simple_summary and not guarded_summary:
        if anchors:
            combined = " ".join(parsed.get("abstract", "") for parsed in parsed_entries)
            hit_terms = [anchor for anchor in anchors if anchor.lower() in combined.lower()][:3]
            if hit_terms:
                anchor_hint = f" Bottom line: Recurring themes include {', '.join(hit_terms)}."
        if expected_topic == "general_autism" and _env_flag("AUTISM_GPT_TASK3_GENERAL_SUMMARY2"):
            bottom_line = _task3_bottom_line(query=query, expected_topic=expected_topic, entries=parsed_entries)
            if bottom_line:
                anchor_hint = f"{anchor_hint} {bottom_line}".strip()

    answer = "\n".join([summary_line] + evidence_lines) + anchor_hint
    return _maybe_prepend_task3_ar_lead(task=task, query=query, answer=answer)


def cmd_run_query(args):
    router = RouterPredictor(Path(args.router_model_dir))
    retriever = TaskRetriever(Path(args.index_dir))
    expert_generator = TaskExpertGenerator(
        {
            "task1": Path(args.task1_expert_dir) if args.task1_expert_dir else None,
            "task2": Path(args.task2_expert_dir) if args.task2_expert_dir else None,
            "task3": Path(args.task3_expert_dir) if args.task3_expert_dir else None,
        },
        default_max_input_tokens=args.expert_max_input_tokens,
    )

    task, conf, scores = router.predict(args.query)
    initial = {"task": task, "confidence": conf, "scores": scores}

    if conf < args.router_threshold:
        refined_q = _refine_query(args.query, task)
        task2, conf2, scores2 = router.predict(refined_q)
        if conf2 > conf:
            task, conf, scores = task2, conf2, scores2
            query_used = refined_q
        else:
            query_used = args.query
    else:
        query_used = args.query

    task, conf = _maybe_override_task1(query=args.query, predicted_task=task, confidence=conf)
    task, conf = _maybe_override_task3(query=args.query, predicted_task=task, confidence=conf)

    if task == "reject":
        output = {
            "status": "abstain",
            "reason": "router_reject",
            "initial_router": initial,
            "final_router": {"task": task, "confidence": conf, "scores": scores},
            "query": args.query,
        }
        _print_json(output, ensure_ascii=False)
        return

    hits = retriever.search(task, query_used, topk=args.topk)
    hits = _promote_explicit_gene_hit(task=task, query=query_used, hits=hits, retriever=retriever)
    if not _context_ok(hits, args.context_threshold):
        # re-retrieve with softened query
        query_retry = _norm_space(re.sub(r"[^\w\s-]", " ", query_used))
        hits_retry = retriever.search(task, query_retry, topk=args.topk)
        hits_retry = _promote_explicit_gene_hit(task=task, query=query_retry, hits=hits_retry, retriever=retriever)
        if _context_ok(hits_retry, args.context_threshold):
            hits = hits_retry
            query_used = query_retry

    judge_trace = []
    if task in {"task2", "task3"}:
        judge_result = _maybe_apply_retrieval_judge(
            task=task,
            query=args.query,
            query_used=query_used,
            hits=hits,
            retriever=retriever,
            topk=args.topk,
            context_threshold=args.context_threshold,
        )
        query_used = judge_result["query_used"]
        hits = judge_result["hits"]
        judge_trace = judge_result.get("judge_trace") or []
        abstain_answer = judge_result.get("abstain_answer")
        if abstain_answer:
            output = {
                "status": "abstain",
                "reason": "retrieval_judge_abstain",
                "task": task,
                "confidence": conf,
                "query": args.query,
                "query_used_for_retrieval": query_used,
                "top_hits": hits,
                "final_answer": abstain_answer,
                "retrieval_judge": judge_trace,
            }
            _print_json(output, ensure_ascii=False)
            return

    if not _context_ok(hits, args.context_threshold):
        output = {
            "status": "abstain",
            "reason": "insufficient_context",
            "task": task,
            "confidence": conf,
            "query": args.query,
        }
        _print_json(output, ensure_ascii=False)
        return

    prompt_pack = {
        "system_prompt": _build_system_prompt(task),
        "user_query": args.query,
        "context": [h["text"] for h in hits],
    }
    generation = None
    generation_contexts = None
    if expert_generator.has_expert(task):
        generation_contexts = _select_generation_contexts(
            task=task,
            query=args.query,
            hits=hits,
            retriever=retriever,
        )
        generation = expert_generator.generate(
            task=task,
            user_query=args.query,
            contexts=generation_contexts,
            max_input_tokens=args.expert_max_input_tokens,
            max_new_tokens=args.expert_max_new_tokens,
            temperature=args.expert_temperature,
            top_p=args.expert_top_p,
        )
        if task == "task1" and generation_contexts:
            generation["raw_answer"] = generation.get("answer")
            generation["answer"] = _format_task1_structured_answer(
                query=args.query,
                context_text=generation_contexts[0],
            )
            generation["fallback_used"] = "task1_extractive_formatter"
        elif _is_low_quality_generation(task=task, answer=generation.get("answer"), query=args.query):
            generation["answer"] = _heuristic_answer(
                task=task,
                query=args.query,
                generation_contexts=generation_contexts,
            )
            generation["fallback_used"] = "heuristic_postprocess"
    output = {
        "status": "ok",
        "task": task,
        "router_confidence": conf,
        "query_used_for_retrieval": query_used,
        "top_hits": hits,
        "prompt_pack": prompt_pack,
        "final_answer": generation["answer"] if generation else None,
        "generation": generation,
        "generation_contexts": generation_contexts,
        "retrieval_judge": judge_trace,
    }
    _print_json(output, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autism GPT MVP (3-task revised-paper style).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p0 = sub.add_parser("prepare-data", help="Build task corpora and router synthetic dataset.")
    p0.add_argument("--root", required=True, help="Path to Autism_new_GPT directory.")
    p0.add_argument("--pubmed-dir", default=None, help="Optional parquet folder for PubMed corpus.")
    p0.add_argument("--pubmed-max-rows", type=int, default=200000)
    p0.add_argument("--task3-max-docs", type=int, default=120000)
    p0.add_argument("--task2-max-lit-docs-per-gene", type=int, default=4)
    p0.add_argument("--gene-registry", default=None, help="Optional gene registry file (csv/json/jsonl/txt).")
    p0.add_argument("--gene-list", default=None, help="Optional explicit gene list file (txt/csv/json/jsonl).")
    p0.add_argument("--support-status", default=None, help="Optional comma-separated support status filter, e.g. supported.")
    p0.add_argument("--tiers", default=None, help="Optional comma-separated tier filter, e.g. core,extended.")
    p0.add_argument("--out-dir", required=True, help="Output data directory.")
    p0.set_defaults(func=cmd_prepare_data)

    p1 = sub.add_parser("build-index", help="Build FAISS indices for task1/2/3.")
    p1.add_argument("--data-dir", required=True, help="Directory containing task*_docs.jsonl.")
    p1.add_argument("--out-dir", required=True, help="Index output directory.")
    p1.add_argument("--embed-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p1.add_argument("--embed-batch-size", type=int, default=128)
    p1.set_defaults(func=cmd_build_index)

    p2 = sub.add_parser("train-router", help="Train BERT-family task router.")
    p2.add_argument("--data-dir", required=True, help="Directory containing router_train/eval jsonl.")
    p2.add_argument("--out-dir", required=True, help="Router model output directory.")
    p2.add_argument("--router-model", default="distilbert-base-uncased")
    p2.add_argument("--epochs", type=int, default=3)
    p2.add_argument("--batch-size", type=int, default=32)
    p2.add_argument("--max-length", type=int, default=256)
    p2.add_argument("--lr", type=float, default=2e-5)
    p2.add_argument("--seed", type=int, default=42)
    p2.set_defaults(func=cmd_train_router)

    p3 = sub.add_parser("run-query", help="Route + retrieve + verifier + abstain.")
    p3.add_argument("--query", required=True)
    p3.add_argument("--router-model-dir", required=True)
    p3.add_argument("--index-dir", required=True)
    p3.add_argument("--topk", type=int, default=5)
    p3.add_argument("--router-threshold", type=float, default=0.55)
    p3.add_argument("--context-threshold", type=float, default=0.30)
    p3.add_argument("--task1-expert-dir", default=None)
    p3.add_argument("--task2-expert-dir", default=None)
    p3.add_argument("--task3-expert-dir", default=None)
    p3.add_argument("--expert-max-input-tokens", type=int, default=1024)
    p3.add_argument("--expert-max-new-tokens", type=int, default=220)
    p3.add_argument("--expert-temperature", type=float, default=0.0)
    p3.add_argument("--expert-top-p", type=float, default=1.0)
    p3.set_defaults(func=cmd_run_query)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
