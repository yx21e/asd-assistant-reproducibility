"""Describe all frozen attempts and paired query-cluster uncertainty."""
import json
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
METRICS = ('faithfulness', 'ar', 'ar_without_question_lead')


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def paired(rows, left, right, metric):
    a = {r['example_id']: r for r in rows if (r['configuration'], r['condition']) == left}
    b = {r['example_id']: r for r in rows if (r['configuration'], r['condition']) == right}
    differences = []
    ids = []
    for key in sorted(set(a) & set(b)):
        x = (a[key].get('scores') or {}).get(metric)
        y = (b[key].get('scores') or {}).get(metric)
        if x is not None and y is not None and np.isfinite(x) and np.isfinite(y):
            differences.append(x - y)
            ids.append(key)
    if not differences:
        return {'paired_n': 0, 'mean_difference': None, 'ci95': None}
    d = np.asarray(differences)
    rng = np.random.default_rng(20261120)
    samples = d[rng.integers(0, len(d), size=(2000, len(d)))].mean(axis=1)
    return {'paired_n': len(d), 'query_ids': ids, 'mean_difference': float(d.mean()),
            'ci95': [float(v) for v in np.quantile(samples, [.025, .975])]}


def main():
    bank = read('FROZEN_BANK.json')
    status = read('STATUS.json')
    rows = [json.loads(s) for s in (ROOT / 'OUTPUTS.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    expected = {(r['query_id'], a, c) for r in bank for a in ('task_conditioned', 'shared_pool_plain_topk') for c in ('source_available', 'source_withheld')}
    actual = [(r['example_id'], r['configuration'], r['condition']) for r in rows]
    assert len(actual) == len(set(actual)) and set(actual) == expected
    assert status['state'] == 'completed' and status['inputs_unchanged']
    for item in bank:
        group = [r for r in rows if r['example_id'] == item['query_id']]
        assert len({json.dumps(r['final_router'], sort_keys=True) for r in group}) == 1
    cells = []
    comparisons = []
    for task in ('task2', 'task3'):
        task_rows = [r for r in rows if r['intended_task'] == task]
        for condition in ('source_available', 'source_withheld'):
            for arm in ('task_conditioned', 'shared_pool_plain_topk'):
                group = [r for r in task_rows if r['configuration'] == arm and r['condition'] == condition]
                cell = {'task': task, 'arm': arm, 'condition': condition, 'attempts': len(group),
                        'statuses': dict(Counter(r['status'] for r in group)),
                        'route_correct': sum(r['route_correct'] for r in group),
                        'generated': sum(r.get('raw_generation') is not None for r in group),
                        'nonempty_final_answer': sum(bool(r.get('final_answer')) for r in group)}
                for key in ('source_in_hits', 'source_in_selected_contexts', 'source_marker_in_actual_input', 'input_truncated'):
                    cell[key] = sum((r.get('diagnostics') or {}).get(key, False) for r in group)
                cell['metrics'] = {}
                for metric in METRICS:
                    values = [(r.get('scores') or {}).get(metric) for r in group]
                    values = [v for v in values if v is not None and np.isfinite(v)]
                    cell['metrics'][metric] = {'n_valid': len(values), 'mean': float(np.mean(values)) if values else None,
                        'sd': float(np.std(values, ddof=1)) if len(values) > 1 else None}
                cells.append(cell)
            comparisons.append({'task': task, 'contrast': 'task_conditioned_minus_shared_pool', 'condition': condition,
                'metrics': {m: paired(task_rows, ('task_conditioned', condition), ('shared_pool_plain_topk', condition), m) for m in METRICS}})
        for arm in ('task_conditioned', 'shared_pool_plain_topk'):
            comparisons.append({'task': task, 'contrast': 'available_minus_withheld', 'arm': arm,
                'metrics': {m: paired(task_rows, (arm, 'source_available'), (arm, 'source_withheld'), m) for m in METRICS}})
    report = {'task_counts': dict(Counter(r['task'] for r in bank)), 'all_attempts': len(rows),
        'complete_grid': True, 'no_paid_calls': status['paid_calls'] == 0, 'inputs_unchanged': status['inputs_unchanged'],
        'role': 'small_new_question_retained_source_diagnostic_not_powered_or_expert_validated',
        'score_task': 'intended_task_including_wrong_routes', 'bootstrap': {'resamples': 2000, 'seed': 20261120,
        'cluster': 'question/designated_source', 'interval': 'paired_percentile_95_descriptive'},
        'cells': cells, 'paired_comparisons': comparisons,
        'open_gaps': ['blinded_expert_validation', 'verified_direct_conflict_pairs', 'foundation_training_independence', 'reviewer_access', 'author_declarations']}
    with (ROOT / 'RESULT.json').open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
