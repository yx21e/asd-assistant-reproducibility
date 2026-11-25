#!/usr/bin/env python3
"""
Evaluate the current Autism GPT pipeline with practical proxy metrics.

This evaluator is intentionally lightweight and reproducible. It focuses on:
1. Router accuracy on the existing synthetic eval set
2. End-to-end task accuracy proxies against synthetic SFT references
3. Grounding / fallback diagnostics for task experts

These are not the final AD-GPT paper metrics, but they provide a fast and
honest view of the system's current quality before larger benchmark work.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autism_gpt_mvp import (  # noqa: E402
    ID_TO_TASK,
    RouterPredictor,
    TASK_LABELS,
    TaskExpertGenerator,
    TaskRetriever,
    _context_ok,
    _format_task1_structured_answer,
    _heuristic_answer,
    _is_low_quality_generation,
    _maybe_apply_retrieval_judge,
    _maybe_override_task1,
    _maybe_override_task3,
    _maybe_prepend_task3_ar_lead,
    _norm_space,
    _promote_explicit_gene_hit,
    _read_jsonl,
    _refine_query,
    _select_generation_contexts,
)
from gene_registry import filter_rows_by_gene, resolve_gene_filter  # noqa: E402
from task3_topic_rules import (  # noqa: E402
    count_anchor_hits,
    infer_topic_from_question,
    is_probable_non_autism_record,
    topic_anchor_terms,
)


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
    "you",
    "your",
}


def read_jsonl(path: Path) -> List[dict]:
    return _read_jsonl(path)


def sample_rows(rows: Sequence[dict], max_samples: Optional[int], seed: int) -> List[dict]:
    rows = list(rows)
    if max_samples == 0:
        return []
    if max_samples is None or max_samples < 0 or len(rows) <= max_samples:
        return rows
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:max_samples]


def sample_task3_rows(rows: Sequence[dict], max_samples: Optional[int], seed: int) -> List[dict]:
    rows = list(rows)
    if max_samples == 0:
        return []
    if max_samples is None or max_samples < 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    grouped = defaultdict(list)
    for row in rows:
        topic = row.get("topic") or infer_topic_from_question(row.get("question", "")) or "general_autism"
        grouped[str(topic)].append(row)

    selected = []
    leftovers = []
    topics = sorted(grouped)
    base = max_samples // max(len(topics), 1)
    remainder = max_samples % max(len(topics), 1)
    for idx, topic in enumerate(topics):
        topic_rows = grouped[topic][:]
        rng.shuffle(topic_rows)
        take = min(len(topic_rows), base + (1 if idx < remainder else 0))
        selected.extend(topic_rows[:take])
        leftovers.extend(topic_rows[take:])

    if len(selected) < max_samples:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max_samples - len(selected)])
    rng.shuffle(selected)
    return selected[:max_samples]


def normalize_text(text: str) -> str:
    return _norm_space(text).lower()


def extract_tagged_section(text: str, tag: str) -> str:
    raw = str(text or "")
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, raw, flags=re.IGNORECASE | re.DOTALL)
    return _norm_space(match.group(1)) if match else ""


def answer_text_for_metrics(text: str) -> str:
    answer = extract_tagged_section(text, "answer")
    return answer if answer else _norm_space(text)


def token_f1(pred: str, ref: str) -> float:
    ptoks = normalize_text(pred).split()
    rtoks = normalize_text(ref).split()
    if not ptoks and not rtoks:
        return 1.0
    if not ptoks or not rtoks:
        return 0.0
    pc = Counter(ptoks)
    rc = Counter(rtoks)
    overlap = sum((pc & rc).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(ptoks)
    recall = overlap / len(rtoks)
    return 2 * precision * recall / (precision + recall)


def safe_mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./:+-]+", normalize_text(text))
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def grounded_token_ratio(answer: str, contexts: Sequence[str]) -> float:
    atoks = content_tokens(answer)
    if not atoks:
        return 0.0
    cset = set(content_tokens("\n".join(contexts)))
    return sum(tok in cset for tok in atoks) / len(atoks)


def critical_tokens(text: str) -> List[str]:
    patterns = [
        r"GRCh\d+:[0-9-]+",
        r"T2T:[0-9-]+",
        r"Brain_[A-Za-z0-9_\-]+",
        r"chr[0-9XYM]+_[0-9]+_[A-Z]+_[A-Z]+_b38",
        r"[0-9]+\.[0-9]+e[+-]?[0-9]+",
        r"PMID:\s*[0-9]+",
        r"\b[A-Z0-9-]{2,15}\b",
    ]
    found = []
    for pattern in patterns:
        found.extend(re.findall(pattern, text))
    cleaned = []
    seen = set()
    for item in found:
        item = _norm_space(item)
        if item and item not in seen:
            seen.add(item)
            cleaned.append(item)
    return cleaned


def critical_token_recall(pred: str, ref: str) -> float:
    ref_toks = critical_tokens(ref)
    if not ref_toks:
        return 1.0
    pred_norm = normalize_text(pred)
    hits = sum(tok.lower() in pred_norm for tok in ref_toks)
    return hits / len(ref_toks)


def hit_topics(hits: Sequence[dict]) -> List[str]:
    return [str((hit.get("meta") or {}).get("topic") or "") for hit in hits if hit]


def hit_contamination_rate(hits: Sequence[dict]) -> float:
    if not hits:
        return 0.0
    bad = 0
    for hit in hits:
        if is_probable_non_autism_record(abstract=hit.get("text", "")):
            bad += 1
    return bad / len(hits)


def run_pipeline(
    query: str,
    router: RouterPredictor,
    retriever: TaskRetriever,
    expert_generator: TaskExpertGenerator,
    topk: int,
    router_threshold: float,
    context_threshold: float,
    expert_max_input_tokens: int,
    expert_max_new_tokens: int,
    expert_temperature: float,
    expert_top_p: float,
) -> dict:
    task, conf, scores = router.predict(query)
    initial = {"task": task, "confidence": conf, "scores": scores}

    if conf < router_threshold:
        refined_q = _refine_query(query, task)
        task2, conf2, scores2 = router.predict(refined_q)
        if conf2 > conf:
            task, conf, scores = task2, conf2, scores2
            query_used = refined_q
        else:
            query_used = query
    else:
        query_used = query

    task, conf = _maybe_override_task1(query=query, predicted_task=task, confidence=conf)
    task, conf = _maybe_override_task3(query=query, predicted_task=task, confidence=conf)

    if task == "reject":
        return {
            "status": "abstain",
            "reason": "router_reject",
            "initial_router": initial,
            "final_router": {"task": task, "confidence": conf, "scores": scores},
            "query": query,
        }

    hits = retriever.search(task, query_used, topk=topk)
    hits = _promote_explicit_gene_hit(task=task, query=query_used, hits=hits, retriever=retriever)
    if not _context_ok(hits, context_threshold):
        query_retry = _norm_space(re.sub(r"[^\w\s-]", " ", query_used))
        hits_retry = retriever.search(task, query_retry, topk=topk)
        hits_retry = _promote_explicit_gene_hit(task=task, query=query_retry, hits=hits_retry, retriever=retriever)
        if _context_ok(hits_retry, context_threshold):
            hits = hits_retry
            query_used = query_retry

    judge_trace = []
    if task in {"task2", "task3"}:
        judge_result = _maybe_apply_retrieval_judge(
            task=task,
            query=query,
            query_used=query_used,
            hits=hits,
            retriever=retriever,
            topk=topk,
            context_threshold=context_threshold,
        )
        query_used = judge_result["query_used"]
        hits = judge_result["hits"]
        judge_trace = judge_result.get("judge_trace") or []
        abstain_answer = judge_result.get("abstain_answer")
        if abstain_answer:
            return {
                "status": "abstain",
                "reason": "retrieval_judge_abstain",
                "task": task,
                "router_confidence": conf,
                "query_used_for_retrieval": query_used,
                "top_hits": hits,
                "prompt_pack": {"user_query": query, "context": [h["text"] for h in hits]},
                "final_answer": abstain_answer,
                "generation": {
                    "answer": abstain_answer,
                    "fallback_used": "retrieval_judge_abstain",
                    "judge_trace": judge_trace,
                },
                "generation_contexts": [h["text"] for h in hits[:3]],
                "retrieval_judge": judge_trace,
            }

    if not _context_ok(hits, context_threshold):
        return {
            "status": "abstain",
            "reason": "insufficient_context",
            "task": task,
            "confidence": conf,
            "query": query,
        }

    prompt_pack = {
        "user_query": query,
        "context": [h["text"] for h in hits],
    }

    generation = None
    generation_contexts = None
    if expert_generator.has_expert(task):
        generation_contexts = _select_generation_contexts(
            task=task,
            query=query,
            hits=hits,
            retriever=retriever,
        )
        generation = expert_generator.generate(
            task=task,
            user_query=query,
            contexts=generation_contexts,
            max_input_tokens=expert_max_input_tokens,
            max_new_tokens=expert_max_new_tokens,
            temperature=expert_temperature,
            top_p=expert_top_p,
        )
        if task == "task1" and generation_contexts:
            generation["raw_answer"] = generation.get("answer")
            generation["answer"] = _format_task1_structured_answer(
                query=query,
                context_text=generation_contexts[0],
            )
            generation["fallback_used"] = "task1_extractive_formatter"
        elif _is_low_quality_generation(task=task, answer=generation.get("answer"), query=query):
            generation["answer"] = _heuristic_answer(
                task=task,
                query=query,
                generation_contexts=generation_contexts,
            )
            generation["fallback_used"] = "heuristic_postprocess"
        if task == "task3":
            generation["answer"] = _maybe_prepend_task3_ar_lead(
                task=task,
                query=query,
                answer=generation.get("answer") or "",
            )

    return {
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


def evaluate_router(router: RouterPredictor, router_eval_rows: Sequence[dict]) -> dict:
    totals = 0
    correct = 0
    per_label = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in router_eval_rows:
        pred_task, _, _ = router.predict(row["text"])
        gold_task = ID_TO_TASK[int(row["label"])]
        totals += 1
        per_label[gold_task]["total"] += 1
        if pred_task == gold_task:
            correct += 1
            per_label[gold_task]["correct"] += 1

    summary = {
        "samples": totals,
        "accuracy": correct / totals if totals else 0.0,
        "per_task_accuracy": {},
    }
    for task, counts in sorted(per_label.items()):
        total = counts["total"]
        summary["per_task_accuracy"][task] = counts["correct"] / total if total else 0.0
    return summary


def evaluate_task_rows(
    task_name: str,
    rows: Sequence[dict],
    router: RouterPredictor,
    retriever: TaskRetriever,
    expert_generator: TaskExpertGenerator,
    args,
) -> dict:
    predictions = []
    route_correct = 0
    abstains = 0
    fallback_count = 0
    exact_matches = []
    token_f1s = []
    grounded_ratios = []
    critical_recalls = []
    topic_hit_rates = []
    contamination_rates = []
    anchor_recalls = []
    examples = []

    for row in rows:
        pred = run_pipeline(
            query=row["question"],
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
        predictions.append(pred)
        if pred.get("task") == task_name:
            route_correct += 1
        if pred.get("status") != "ok":
            abstains += 1
            examples.append(
                {
                    "query": row["question"],
                    "status": pred.get("status"),
                    "reason": pred.get("reason"),
                }
            )
            continue

        answer = pred.get("final_answer") or ""
        reference = row["response"]
        metric_answer = answer_text_for_metrics(answer)
        metric_reference = answer_text_for_metrics(reference)
        exact = normalize_text(metric_answer) == normalize_text(metric_reference)
        exact_matches.append(1.0 if exact else 0.0)
        token_f1s.append(token_f1(metric_answer, metric_reference))
        grounded_ratios.append(
            grounded_token_ratio(metric_answer, pred.get("generation_contexts") or pred["prompt_pack"]["context"])
        )
        critical_recalls.append(critical_token_recall(metric_answer, metric_reference))
        generation = pred.get("generation") or {}
        if generation.get("fallback_used"):
            fallback_count += 1

        if task_name == "task3":
            expected_topic = row.get("topic") or infer_topic_from_question(row.get("question", "")) or "general_autism"
            hits = pred.get("top_hits") or []
            top3_topics = hit_topics(hits[:3])
            topic_hit_rates.append(1.0 if expected_topic in top3_topics else 0.0)
            contamination_rates.append(hit_contamination_rate(hits[:3]))
            anchors = topic_anchor_terms(str(expected_topic))
            anchor_recalls.append(
                count_anchor_hits(metric_answer, anchors) / len(anchors) if anchors else 0.0
            )

        if (not exact or token_f1s[-1] < 0.7) and len(examples) < 12:
            payload = {
                "query": row["question"],
                "prediction": answer[:800],
                "reference": reference[:800],
                "task": pred.get("task"),
            "fallback_used": generation.get("fallback_used"),
                "token_f1": token_f1s[-1],
                "critical_token_recall": critical_recalls[-1],
            }
            if task_name == "task3":
                expected_topic = row.get("topic") or infer_topic_from_question(row.get("question", "")) or "general_autism"
                payload["expected_topic"] = expected_topic
                payload["top_hit_topics"] = hit_topics((pred.get("top_hits") or [])[:3])
                payload["top3_contamination_rate"] = hit_contamination_rate((pred.get("top_hits") or [])[:3])
            examples.append(payload)

    total = len(rows)
    ok_count = total - abstains
    summary = {
        "samples": total,
        "route_accuracy": route_correct / total if total else 0.0,
        "abstain_rate": abstains / total if total else 0.0,
        "fallback_rate": fallback_count / ok_count if ok_count else 0.0,
        "exact_match": safe_mean(exact_matches),
        "avg_token_f1": safe_mean(token_f1s),
        "avg_grounded_token_ratio": safe_mean(grounded_ratios),
        "avg_critical_token_recall": safe_mean(critical_recalls),
        "example_diagnostics": examples,
    }
    if task_name == "task3":
        summary["retrieval_top3_topic_hit_rate"] = safe_mean(topic_hit_rates)
        summary["retrieval_top3_contamination_rate"] = safe_mean(contamination_rates)
        summary["answer_topic_anchor_recall"] = safe_mean(anchor_recalls)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the current Autism GPT pipeline.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--sft-dir", required=True)
    parser.add_argument("--task3-sft-dir", default=None, help="Optional separate SFT dir for task3.")
    parser.add_argument("--router-model-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--task1-expert-dir", default=None)
    parser.add_argument("--task2-expert-dir", default=None)
    parser.add_argument("--task3-expert-dir", default=None)
    parser.add_argument("--gene-registry", default=None, help="Optional gene registry file for task1/task2 filtering.")
    parser.add_argument("--gene-list", default=None, help="Optional explicit gene list file for task1/task2 filtering.")
    parser.add_argument("--support-status", default=None, help="Optional comma-separated support status filter.")
    parser.add_argument("--tiers", default=None, help="Optional comma-separated tier filter.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task1-max-samples", type=int, default=120)
    parser.add_argument("--task2-max-samples", type=int, default=80)
    parser.add_argument("--task3-max-samples", type=int, default=80)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--router-threshold", type=float, default=0.55)
    parser.add_argument("--context-threshold", type=float, default=0.30)
    parser.add_argument("--expert-max-input-tokens", type=int, default=1024)
    parser.add_argument("--expert-max-new-tokens", type=int, default=220)
    parser.add_argument("--expert-temperature", type=float, default=0.0)
    parser.add_argument("--expert-top-p", type=float, default=1.0)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    sft_dir = Path(args.sft_dir)
    task3_sft_dir = Path(args.task3_sft_dir) if args.task3_sft_dir else sft_dir
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    allowed_genes = resolve_gene_filter(
        gene_registry=args.gene_registry,
        gene_list=args.gene_list,
        support_statuses=args.support_status,
        tiers=args.tiers,
    )

    router_eval_rows = read_jsonl(data_dir / "router_eval.jsonl")
    task1_rows = sample_rows(filter_rows_by_gene(read_jsonl(sft_dir / "task1_sft.jsonl"), allowed_genes), args.task1_max_samples, args.seed)
    task2_rows = sample_rows(filter_rows_by_gene(read_jsonl(sft_dir / "task2_sft.jsonl"), allowed_genes), args.task2_max_samples, args.seed + 1)
    task3_rows = sample_task3_rows(read_jsonl(task3_sft_dir / "task3_sft.jsonl"), args.task3_max_samples, args.seed + 2)

    summary = {
        "config": {
            "data_dir": str(data_dir),
            "sft_dir": str(sft_dir),
            "task3_sft_dir": str(task3_sft_dir),
            "router_model_dir": str(args.router_model_dir),
            "index_dir": str(args.index_dir),
            "task1_expert_dir": args.task1_expert_dir,
            "task2_expert_dir": args.task2_expert_dir,
            "task3_expert_dir": args.task3_expert_dir,
            "gene_filter_count": len(allowed_genes) if allowed_genes is not None else None,
            "gene_registry": args.gene_registry,
            "gene_list": args.gene_list,
            "support_status": args.support_status,
            "tiers": args.tiers,
        },
        "router": evaluate_router(router, router_eval_rows),
        "task1": evaluate_task_rows("task1", task1_rows, router, retriever, expert_generator, args),
        "task2": evaluate_task_rows("task2", task2_rows, router, retriever, expert_generator, args),
        "task3": evaluate_task_rows("task3", task3_rows, router, retriever, expert_generator, args),
        "paper_alignment_note": (
            "These are proxy metrics against synthetic SFT references plus grounding diagnostics. "
            "They are not yet the final AD-GPT paper metrics such as CSR/CVR/AR/Faithfulness or benchmark comparisons."
        ),
    }

    summary_path = out_dir / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    sys.stdout.buffer.write(payload.encode("utf-8", errors="replace") + b"\n")


if __name__ == "__main__":
    main()
