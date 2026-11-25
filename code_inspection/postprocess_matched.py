"""Summarize a completed matched run and prepare genuinely blank blind labels."""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def interval(values, seed=20261112):
    v = np.asarray(values, dtype=float)
    if not len(v):
        return None, None, None
    rng = np.random.default_rng(seed)
    boot = v[rng.integers(0, len(v), size=(5000, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.quantile(boot, .025)), float(np.quantile(boot, .975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--watch', action='store_true')
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    begin = time.monotonic()
    while True:
        status = json.loads((run / 'status.json').read_text(encoding='utf-8'))
        if status['state'].startswith('completed'):
            break
        if status['state'] == 'failed' or not args.watch or time.monotonic() - begin > 43200:
            write_json(out / 'status.json', {'state': 'not_ready', 'run_status': status})
            raise RuntimeError('Run is not complete; no full-run estimates or expert labels created.')
        write_json(out / 'status.json', {'state': 'waiting_for_generation', 'completed': status.get('completed', 0), 'total': status.get('total')})
        time.sleep(30)
    rows = [json.loads(s) for s in (run / 'predictions.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    keys = {(r['task'], r['example_id'], r['configuration']) for r in rows}
    assert len(keys) == len(rows), 'Duplicate outputs: manual audit required'
    assert len(rows) == status['total'], 'Incomplete run: do not publish as full'
    metrics = ('faithfulness', 'ar', 'ar_without_question_lead', 'equal_unit_faithfulness', 'G', 'S', 'csr', 'cvr')
    groups = defaultdict(list)
    for r in rows:
        groups[(r['task'], r['configuration'])].append(r)
    aggregates = []
    for (task, arm), rr in groups.items():
        for metric in metrics:
            values = [r[metric] for r in rr if isinstance(r.get(metric), (int, float))]
            mean, lo, hi = interval(values)
            aggregates.append({'task': task, 'configuration': arm, 'metric': metric, 'n_requested': len(rr),
                'n_scored': len(values), 'n_errors': sum(r['status'] == 'error' for r in rr),
                'n_abstentions': sum(r['status'] == 'abstain' for r in rr),
                'mean': mean, 'query_bootstrap_low': lo, 'query_bootstrap_high': hi,
                'n_fallback': sum(bool(r.get('fallback_used')) for r in rr),
                'n_truncated_prompts': sum((r.get('raw_generation') or {}).get('untruncated_input_tokens', 0) >
                    (r.get('raw_generation') or {}).get('actual_input_tokens', 0) for r in rr)})
    write_csv(out / 'matched_results.csv', aggregates)
    paired, sensitivity = [], []
    for task in sorted({r['task'] for r in rows}):
        primary = {r['example_id']: r for r in groups[(task, 'task_conditioned')]}
        for arm in ('shared_pool_retained_selection', 'shared_pool_plain_topk'):
            comparator = {r['example_id']: r for r in groups[(task, arm)]}
            assert set(primary) == set(comparator), 'Unpaired query banks'
            for metric in metrics:
                ids = [i for i in primary if isinstance(primary[i].get(metric), (int, float))
                       and isinstance(comparator[i].get(metric), (int, float))]
                values = [primary[i][metric] - comparator[i][metric] for i in ids]
                mean, lo, hi = interval(values)
                clusters = defaultdict(list)
                for i, value in zip(ids, values):
                    clusters[primary[i].get('topic') or i].append(value)
                cm, cl, ch = interval([float(np.mean(v)) for v in clusters.values()])
                paired.append({'task': task, 'contrast': 'task_conditioned_minus_' + arm,
                    'metric': metric, 'n_pairs': len(ids), 'n_query_clusters': len(clusters),
                    'paired_mean_difference': mean, 'paired_query_CI_low': lo, 'paired_query_CI_high': hi,
                    'macro_cluster_difference': cm, 'cluster_CI_low': cl, 'cluster_CI_high': ch})
        for arm in ('task_conditioned', 'shared_pool_retained_selection', 'shared_pool_plain_topk'):
            rr = groups[(task, arm)]
            for w in (0, .25, .5, .55, .75, 1):
                vals = [w * r['G'] + (1-w) * r['S'] for r in rr if 'G' in r and 'S' in r]
                mean, lo, hi = interval(vals)
                sensitivity.append({'task': task, 'configuration': arm, 'G_weight': w, 'n_with_units': len(vals),
                                    'mean': mean, 'CI_low': lo, 'CI_high': hi})
    write_csv(out / 'paired_comparisons.csv', paired)
    write_csv(out / 'matched_weight_sensitivity.csv', sensitivity)
    release, private = out / 'expert_release', out / 'private_analysis'
    release.mkdir(exist_ok=True)
    private.mkdir(exist_ok=True)
    rng = random.Random(20261112)
    selected = []
    for task in sorted({r['task'] for r in rows}):
        primary = groups[(task, 'task_conditioned')]
        pools = defaultdict(list)
        for r in primary:
            pools[r.get('topic') or 'gene_level'].append(r['example_id'])
        for pool in pools.values():
            rng.shuffle(pool)
        ids = []
        target = min(50, len(primary))
        while len(ids) < target:
            for topic in sorted(pools):
                if pools[topic] and len(ids) < target:
                    ids.append(pools[topic].pop())
        selected.extend(r for r in rows if r['task'] == task and r['example_id'] in ids
                        and r['configuration'] in ('task_conditioned', 'shared_pool_plain_topk'))
    rng.shuffle(selected)
    answer_rows, claims, citations, key_rows = [], [], [], []
    for idx, r in enumerate(selected, 1):
        aid = f'A{idx:04d}'
        evidence = '\n\n'.join(f'E{i+1}: {c}' for i, c in enumerate(r.get('generation_contexts') or []))
        answer = r.get('answer') or '[No answer returned; see system_status]'
        answer_rows.append({'audit_id': aid, 'task': r['task'], 'query': r['query'], 'system_status': r['status'],
            'answer': answer, 'selected_evidence': evidence, 'scientific_support_0_2': '',
            'evidence_strength_appropriate_0_2': '', 'answer_relevance_0_2': '',
            'abstention_appropriate_yes_no_na': '', 'expert_notes': ''})
        text = r.get('answer') or ''
        if r['task'] == 'task2':
            text = '\n'.join(re.findall(r'<(?:reasoning|answer)>\s*(.*?)\s*</(?:reasoning|answer)>', text, flags=re.S | re.I)) or text
        text = re.sub(r'Question addressed:.*?(?=\n|Association evidence|$)', '', text, flags=re.I)
        segments = [s.strip() for s in re.split(r'\n+|(?<=[.!?])\s+(?=[A-Z])', text) if s.strip()]
        for j, segment in enumerate(segments, 1):
            if segment.rstrip(':').lower() in ('evidence highlights', 'references'):
                continue
            claims.append({'audit_id': aid, 'provisional_claim_id': f'{aid}-C{j:02d}',
                'machine_segment_REVIEW_AND_CORRECT': segment, 'corrected_atomic_claim': '',
                'is_scientific_claim_yes_no': '', 'support_supported_partial_contradicted_absent_unassessable': '',
                'strength_appropriate_too_strong_too_weak_na': '', 'supporting_evidence_ids': '',
                'error_type_overclaim_indirect_missing_other': '', 'notes': ''})
        for pmid in sorted(set(re.findall(r'PMID\s*[:=]?\s*(\d+)', r.get('answer') or '', flags=re.I))):
            citations.append({'audit_id': aid, 'pmid': pmid, 'identity_verified_yes_no_unassessable': '',
                'supports_associated_claim_yes_partial_no_unassessable': '', 'associated_claim_ids': '', 'notes': ''})
        key_rows.append({'audit_id': aid, 'task': r['task'], 'example_id': r['example_id'],
                         'configuration': r['configuration'], 'faithfulness': r.get('faithfulness'), 'ar': r.get('ar')})
    for judge in (1, 2):
        rr = answer_rows.copy()
        random.Random(20261112 + judge).shuffle(rr)
        write_csv(release / f'reviewer_{judge}_answers.csv', rr)
        write_csv(release / f'reviewer_{judge}_claims.csv', claims)
        write_csv(release / f'reviewer_{judge}_citations.csv', citations)
    write_csv(private / 'blinding_key_DO_NOT_SHARE.csv', key_rows)
    write_json(release / 'rubric.json', {
        'labels_are_blank': True, 'human_experts_required': 2,
        'support': {'supported': 'The cited/provided evidence supports the precise scientific claim.',
                    'partial': 'Some but not all content is supported, or support is indirect.',
                    'contradicted': 'Evidence states an incompatible result.', 'absent': 'No supplied evidence supports the claim.',
                    'unassessable': 'Evidence is unavailable or insufficient to judge.'},
        'strength': 'Assess association vs causation, direct vs indirect evidence, population, outcome, and uncertainty.',
        'answer_scores': {'0': 'Major problem', '1': 'Partial support / material qualification required', '2': 'Adequately supported / appropriate'},
        'claim_segmentation': 'Machine segments are suggestions, NOT gold claim units. Correct/split/merge before rating.',
        'citation_protocol': 'Apply identical identity and claim-support checks to all answers. Real PMID does not imply support.',
        'adjudication': 'Two independent ratings first, then domain-expert adjudication; preserve both original ratings.',
        'agreement_analysis': 'After labels exist: weighted kappa/ordinal agreement and query-cluster-bootstrap Spearman correlation with the frozen metric.',
        'dataset_role': 'Historical paired diagnostic sample; NOT the prospective challenge evaluation.'})
    (release / 'README.md').write_text(
        '# Blinded expert review\n\nOnly share this expert_release directory, not private_analysis.\n'
        'Review answers independently before discussing with the second reviewer. All rating fields are intentionally blank.\n'
        'Inspect the query, full answer, and supplied evidence; citation presence is not proof of scientific support.\n'
        'Correct machine-proposed claim boundaries before annotation. Use the same rubric for every answer.\n'
        'Do not use an LLM to fill these fields and then describe the result as human review.\n'
        'This packet samples historical queries; a separately frozen prospective challenge set is still required.\n', encoding='utf-8')
    finish = {'state': 'completed', 'prediction_rows': len(rows), 'blind_answer_count': len(answer_rows),
              'blind_query_pairs': len(answer_rows)//2, 'human_labels_filled': 0,
              'run_has_errors': any(r['status'] == 'error' for r in rows)}
    write_json(out / 'status.json', finish)
    print(json.dumps(finish), flush=True)


if __name__ == '__main__':
    main()
