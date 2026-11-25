#!/usr/bin/env python3
"""
Evaluate Task 3 on a more realistic set of user-style autism literature queries.

This benchmark avoids synthetic exact-match references and instead measures:
1. routing correctness
2. topic-aligned retrieval
3. contamination avoidance
4. grounded, topic-aware answers
5. preference for review-style evidence when the query is broad
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autism_gpt_mvp import (  # noqa: E402
    RouterPredictor,
    TaskExpertGenerator,
    TaskRetriever,
    _context_ok,
    _heuristic_answer,
    _is_low_quality_generation,
    _maybe_override_task3,
    _maybe_prepend_task3_ar_lead,
    _norm_space,
    _promote_explicit_gene_hit,
    _refine_query,
    _select_generation_contexts,
)
from task3_topic_rules import (  # noqa: E402
    count_anchor_hits,
    has_review_signal,
    infer_topic_from_question,
    is_probable_non_autism_record,
    topic_anchor_terms,
    wants_review_style,
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
}


QUERY_BANK: List[dict] = [
    {"topic": "clinical_features", "question": "What are common clinical signs and early developmental features of autism spectrum disorder?"},
    {"topic": "clinical_features", "question": "How are social communication differences and repetitive behaviors described in autism literature?"},
    {"topic": "clinical_features", "question": "What sensory or language-related features are commonly discussed in autism clinical studies?"},
    {"topic": "clinical_features", "question": "Summarize clinically important autism features reported in recent literature."},
    {"topic": "diagnosis_screening", "question": "What screening tools are commonly used to identify autism in young children?"},
    {"topic": "diagnosis_screening", "question": "How do autism diagnostic studies describe assessment instruments such as ADOS or M-CHAT?"},
    {"topic": "diagnosis_screening", "question": "Summarize practical autism screening and diagnostic approaches from the literature."},
    {"topic": "diagnosis_screening", "question": "What does the literature emphasize about multidisciplinary autism assessment?"},
    {"topic": "biomarkers", "question": "What biomarkers are most commonly discussed in autism research?"},
    {"topic": "biomarkers", "question": "How are EEG, MRI, or blood-based biomarkers described in autism studies?"},
    {"topic": "biomarkers", "question": "Summarize major autism biomarkers discussed in recent literature."},
    {"topic": "biomarkers", "question": "What does the literature say about whether autism has a single validated biomarker?"},
    {"topic": "epidemiology_population", "question": "How do sex differences appear in autism epidemiology?"},
    {"topic": "epidemiology_population", "question": "What population disparities or underdiagnosis patterns are discussed in autism literature?"},
    {"topic": "epidemiology_population", "question": "How do autism prevalence studies describe differences across cohorts or demographic groups?"},
    {"topic": "epidemiology_population", "question": "Summarize population-level patterns discussed in autism research."},
    {"topic": "methods_study_design", "question": "What study designs are common in autism research?"},
    {"topic": "methods_study_design", "question": "How do autism papers use reviews, cohort studies, or trials?"},
    {"topic": "methods_study_design", "question": "Summarize research methods discussed in autism literature."},
    {"topic": "methods_study_design", "question": "What role do machine learning or data-driven methods play in autism studies?"},
    {"topic": "risk_pathogenesis", "question": "What mechanisms and risk factors are commonly discussed in autism pathogenesis?"},
    {"topic": "risk_pathogenesis", "question": "How do autism studies discuss prenatal, environmental, or immune-related risk?"},
    {"topic": "risk_pathogenesis", "question": "Summarize autism risk and pathogenesis themes discussed in the literature."},
    {"topic": "risk_pathogenesis", "question": "What does the literature suggest about multifactorial autism etiology?"},
    {"topic": "general_autism", "question": "What broad themes dominate recent autism research?"},
    {"topic": "general_autism", "question": "Give a high-level literature-backed overview of autism research directions."},
    {"topic": "general_autism", "question": "Summarize broad themes in the current autism literature."},
    {"topic": "general_autism", "question": "What high-level autism knowledge is reflected across recent studies?"},
]


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


def query_text(row: dict) -> str:
    return str(row.get("question") or row.get("query") or "")


def content_tokens(text: str) -> List[str]:
    toks = re.findall(r"[A-Za-z0-9_./:+-]+", normalize_text(text))
    return [t for t in toks if len(t) > 1 and t not in STOPWORDS]


def grounded_token_ratio(answer: str, contexts: Sequence[str]) -> float:
    atoks = content_tokens(answer)
    if not atoks:
        return 0.0
    cset = set(content_tokens("\n".join(contexts)))
    return sum(tok in cset for tok in atoks) / len(atoks)


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

    task, conf = _maybe_override_task3(query=query, predicted_task=task, confidence=conf)

    hits = retriever.search(task, query_used, topk=topk)
    hits = _promote_explicit_gene_hit(task=task, query=query_used, hits=hits, retriever=retriever)
    if not _context_ok(hits, context_threshold):
        query_retry = _norm_space(re.sub(r"[^\w\s-]", " ", query_used))
        hits_retry = retriever.search(task, query_retry, topk=topk)
        hits_retry = _promote_explicit_gene_hit(task=task, query=query_retry, hits=hits_retry, retriever=retriever)
        if _context_ok(hits_retry, context_threshold):
            hits = hits_retry
            query_used = query_retry

    generation = None
    generation_contexts = None
    if task == "task3" and expert_generator.has_expert(task) and _context_ok(hits, context_threshold):
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
        if _is_low_quality_generation(task=task, answer=generation.get("answer"), query=query):
            generation["answer"] = _heuristic_answer(
                task=task,
                query=query,
                generation_contexts=generation_contexts,
            )
            generation["fallback_used"] = "heuristic_postprocess"
        generation["answer"] = _maybe_prepend_task3_ar_lead(
            task=task,
            query=query,
            answer=generation.get("answer") or "",
        )

    return {
        "task": task,
        "router_confidence": conf,
        "query_used_for_retrieval": query_used,
        "top_hits": hits,
        "final_answer": generation["answer"] if generation else None,
        "generation": generation,
        "generation_contexts": generation_contexts,
    }


def evaluate_queries(rows: Sequence[dict], router: RouterPredictor, retriever: TaskRetriever, expert_generator: TaskExpertGenerator, args) -> dict:
    route_correct = 0
    fallback_count = 0
    grounded_scores = []
    topic_hit_rates = []
    contamination_rates = []
    answer_anchor_recalls = []
    review_hit_rates = []
    per_topic = defaultdict(lambda: {"samples": 0, "route": 0, "topic_hit": [], "anchor": []})
    diagnostics = []
    predictions = []

    for row in rows:
        pred = run_pipeline(
            query=query_text(row),
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
        predictions.append({"query": query_text(row), "topic": row["topic"], **pred})
        expected_topic = row["topic"]
        per_topic[expected_topic]["samples"] += 1
        generation = pred.get("generation") or {}

        if pred.get("task") == "task3":
            route_correct += 1
            per_topic[expected_topic]["route"] += 1

        hits = pred.get("top_hits") or []
        top3 = hits[:3]
        top_topics = [str((hit.get("meta") or {}).get("topic") or "") for hit in top3]
        topic_hit = 1.0 if expected_topic in top_topics else 0.0
        contamination = sum(
            1
            for hit in top3
            if is_probable_non_autism_record(
                title=(hit.get("meta") or {}).get("title", ""),
                abstract=hit.get("text", ""),
                keywords=(hit.get("meta") or {}).get("keywords", ""),
                publication_type=(hit.get("meta") or {}).get("publication_type", ""),
            )
        ) / max(len(top3), 1)
        topic_hit_rates.append(topic_hit)
        contamination_rates.append(contamination)
        per_topic[expected_topic]["topic_hit"].append(topic_hit)

        answer = str(pred.get("final_answer") or "")
        if generation.get("fallback_used"):
            fallback_count += 1
        if answer:
            grounded_scores.append(
                grounded_token_ratio(answer, pred.get("generation_contexts") or [hit.get("text", "") for hit in top3])
            )
        else:
            grounded_scores.append(0.0)

        anchors = row.get("anchors") or topic_anchor_terms(expected_topic)
        anchor_recall = count_anchor_hits(answer, anchors) / len(anchors) if anchors else 0.0
        answer_anchor_recalls.append(anchor_recall)
        per_topic[expected_topic]["anchor"].append(anchor_recall)

        if wants_review_style(query_text(row)):
            review_hit = 1.0 if any(
                has_review_signal(
                    publication_type=(hit.get("meta") or {}).get("publication_type", ""),
                    title=(hit.get("meta") or {}).get("title", ""),
                    keywords=(hit.get("meta") or {}).get("keywords", ""),
                )
                for hit in top3
            ) else 0.0
            review_hit_rates.append(review_hit)

        if len(diagnostics) < 18 and (topic_hit == 0.0 or anchor_recall < 0.34 or not answer):
            diagnostics.append(
                {
                    "query": query_text(row),
                    "expected_topic": expected_topic,
                    "predicted_task": pred.get("task"),
                    "top_hit_topics": top_topics,
                    "retrieval_top3_contamination_rate": contamination,
                    "answer_anchor_recall": anchor_recall,
                    "grounded_token_ratio": grounded_scores[-1],
                    "fallback_used": generation.get("fallback_used"),
                    "answer": answer[:1200],
                }
            )

    total = len(rows)
    summary = {
        "samples": total,
        "route_accuracy": route_correct / total if total else 0.0,
        "fallback_rate": fallback_count / total if total else 0.0,
        "retrieval_top3_topic_hit_rate": safe_mean(topic_hit_rates),
        "retrieval_top3_contamination_rate": safe_mean(contamination_rates),
        "avg_grounded_token_ratio": safe_mean(grounded_scores),
        "avg_answer_topic_anchor_recall": safe_mean(answer_anchor_recalls),
        "review_preference_hit_rate": safe_mean(review_hit_rates),
        "per_topic": {},
        "example_diagnostics": diagnostics,
    }
    for topic, stats in sorted(per_topic.items()):
        total_topic = stats["samples"]
        summary["per_topic"][topic] = {
            "samples": total_topic,
            "route_accuracy": stats["route"] / total_topic if total_topic else 0.0,
            "retrieval_top3_topic_hit_rate": safe_mean(stats["topic_hit"]),
            "avg_answer_topic_anchor_recall": safe_mean(stats["anchor"]),
        }
    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Autism GPT task3 on a realistic query bank.")
    parser.add_argument("--router-model-dir", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--task3-expert-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--query-bank-json",
        default=None,
        help="Optional JSON file containing either a list of query rows or an object with a `queries` field.",
    )
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--router-threshold", type=float, default=0.6)
    parser.add_argument("--context-threshold", type=float, default=0.2)
    parser.add_argument("--expert-max-input-tokens", type=int, default=1024)
    parser.add_argument("--expert-max-new-tokens", type=int, default=220)
    parser.add_argument("--expert-temperature", type=float, default=0.0)
    parser.add_argument("--expert-top-p", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    router = RouterPredictor(Path(args.router_model_dir))
    retriever = TaskRetriever(Path(args.index_dir))
    expert_generator = TaskExpertGenerator(
        {
            "task1": None,
            "task2": None,
            "task3": Path(args.task3_expert_dir),
        },
        default_max_input_tokens=args.expert_max_input_tokens,
    )

    query_bank = QUERY_BANK
    if args.query_bank_json:
        raw = read_json(Path(args.query_bank_json))
        query_bank = raw.get("queries", raw) if isinstance(raw, dict) else raw

    summary, predictions = evaluate_queries(query_bank, router, retriever, expert_generator, args)
    write_json(out_dir / "realistic_query_bank.json", {"queries": query_bank})
    write_json(out_dir / "task3_realistic_summary.json", summary)
    write_jsonl(out_dir / "task3_realistic_predictions.jsonl", predictions)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
