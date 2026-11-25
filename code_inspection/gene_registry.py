from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set


def normalize_gene_symbol(value) -> str:
    return str(value or "").strip().upper()


def _normalize_labels(values: Optional[Sequence[str] | str]) -> Optional[Set[str]]:
    if values is None:
        return None
    if isinstance(values, str):
        raw_values = re.split(r"[;,]", values)
    else:
        raw_values = []
        for value in values:
            raw_values.extend(re.split(r"[;,]", str(value)))
    labels = {str(value).strip().lower() for value in raw_values if str(value).strip()}
    return labels or None


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _extract_registry_rows(payload) -> List[dict]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("rows", "genes", "registry"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def load_gene_registry_rows(path: str | Path) -> List[dict]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = _read_jsonl(path)
    elif suffix == ".json":
        rows = _extract_registry_rows(json.loads(path.read_text(encoding="utf-8", errors="replace")))
    elif suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            rows = list(csv.DictReader(f, delimiter=delimiter))
    elif suffix == ".txt":
        rows = [{"gene_symbol": line.strip()} for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    else:
        raise ValueError(f"Unsupported gene registry format: {path}")

    normalized = []
    for row in rows:
        gene = normalize_gene_symbol(
            row.get("gene_symbol")
            or row.get("gene")
            or row.get("symbol")
            or row.get("Gene symbol")
        )
        if not gene:
            continue
        item = dict(row)
        item["gene_symbol"] = gene
        if "support_status" in item and item["support_status"] is not None:
            item["support_status"] = str(item["support_status"]).strip().lower()
        if "tier" in item and item["tier"] is not None:
            item["tier"] = str(item["tier"]).strip().lower()
        normalized.append(item)
    return normalized


def load_gene_list(path: str | Path) -> Set[str]:
    rows = load_gene_registry_rows(path)
    return {row["gene_symbol"] for row in rows}


def resolve_gene_filter(
    gene_registry: Optional[str | Path] = None,
    gene_list: Optional[str | Path] = None,
    support_statuses: Optional[Sequence[str] | str] = None,
    tiers: Optional[Sequence[str] | str] = None,
) -> Optional[Set[str]]:
    registry_genes = None
    status_filter = _normalize_labels(support_statuses)
    tier_filter = _normalize_labels(tiers)

    if gene_registry:
        rows = load_gene_registry_rows(gene_registry)
        registry_genes = set()
        for row in rows:
            status = str(row.get("support_status", "")).strip().lower()
            tier = str(row.get("tier", "")).strip().lower()
            if status_filter is not None and status not in status_filter:
                continue
            if tier_filter is not None and tier not in tier_filter:
                continue
            registry_genes.add(row["gene_symbol"])

    list_genes = load_gene_list(gene_list) if gene_list else None
    if registry_genes is not None and list_genes is not None:
        return registry_genes & list_genes
    if registry_genes is not None:
        return registry_genes
    if list_genes is not None:
        return list_genes
    return None


def infer_gene_symbol(row: dict) -> Optional[str]:
    for key in ("gene_name", "gene_symbol"):
        gene = normalize_gene_symbol(row.get(key))
        if gene:
            return gene

    meta = row.get("meta")
    if isinstance(meta, dict):
        gene = normalize_gene_symbol(meta.get("gene_name") or meta.get("gene_symbol"))
        if gene:
            return gene

    for key in ("doc_id", "example_id"):
        value = str(row.get(key, ""))
        match = re.search(r"\btask[12]_([A-Za-z0-9-]+)(?:_|$)", value)
        if match:
            gene = normalize_gene_symbol(match.group(1))
            if gene:
                return gene

    for key in ("question", "query", "text"):
        value = str(row.get(key, ""))
        match = re.search(r"\b(?:for|gene|about|is)\s+([A-Za-z0-9-]{2,15})\b", value, flags=re.IGNORECASE)
        if match:
            gene = normalize_gene_symbol(match.group(1))
            if gene:
                return gene
    return None


def filter_rows_by_gene(rows: Iterable[dict], allowed_genes: Optional[Set[str]]) -> List[dict]:
    if allowed_genes is None:
        return list(rows)
    filtered = []
    for row in rows:
        gene = infer_gene_symbol(row)
        if gene and gene in allowed_genes:
            filtered.append(row)
    return filtered
