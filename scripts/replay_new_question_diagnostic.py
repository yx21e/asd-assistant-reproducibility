"""Replay saved diagnostic statistics only; no inference or metric rescoring."""
import argparse
import ast
import hashlib
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def verify(root):
    folder = root / 'evidence/new_question'
    bank, result = read(folder / 'bank.json'), read(folder / 'result.json')
    rows = [json.loads(line) for line in (folder / 'outputs.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
    checks = []

    def check(label, value):
        if not value:
            raise AssertionError(label)
        checks.append(label)

    def near(label, x, y):
        check(label, math.isclose(float(x), float(y), abs_tol=1e-12, rel_tol=0))

    check('32 questions', len(bank) == 32)
    check('distinct designated sources', len({q['source_group'] for q in bank}) == 32)
    check('16 questions per task', dict(Counter(q['task'] for q in bank)) == {'task2': 16, 'task3': 16})
    expected = {(q['query_id'], a, c) for q in bank for a in ('task_conditioned', 'shared_pool_plain_topk')
                for c in ('source_available', 'source_withheld')}
    actual = [(r['example_id'], r['configuration'], r['condition']) for r in rows]
    check('complete unique 128-condition grid', len(actual) == len(set(actual)) == 128 and set(actual) == expected)
    for question in bank:
        group = [r for r in rows if r['example_id'] == question['query_id']]
        check('same actual route ' + question['query_id'], len({json.dumps(r['final_router'], sort_keys=True) for r in group}) == 1)
        check('intended task preserved ' + question['query_id'], all(r['intended_task'] == question['task'] for r in group))
    table = (root / 'latex/tables/table_new_question_diagnostic.tex').read_text(encoding='utf-8')
    for cell in result['cells']:
        key = (cell['task'], cell['arm'], cell['condition'])
        group = [r for r in rows if (r['intended_task'], r['configuration'], r['condition']) == key]
        check('attempt denominator ' + str(key), len(group) == cell['attempts'])
        check('statuses ' + str(key), dict(Counter(r['status'] for r in group)) == cell['statuses'])
        check('route count ' + str(key), sum(r['selected_task'] == r['intended_task'] for r in group) == cell['route_correct'])
        for field in ('source_in_hits', 'source_in_selected_contexts', 'source_marker_in_actual_input', 'input_truncated'):
            check(field + str(key), sum(r['diagnostics'][field] for r in group) == cell[field])
        if cell['condition'] == 'source_withheld':
            check('source markers absent ' + str(key), all(not r['diagnostics']['source_in_hits'] and
                  not r['diagnostics']['source_in_selected_contexts'] and
                  not r['diagnostics']['source_marker_in_actual_input'] for r in group))
        for metric, values in cell['metrics'].items():
            observed = [(r.get('scores') or {}).get(metric) for r in group]
            observed = [v for v in observed if v is not None and math.isfinite(v)]
            check('valid score denominator ' + str(key) + metric, len(observed) == values['n_valid'])
            near('mean ' + str(key) + metric, statistics.mean(observed), values['mean'])
            near('sd ' + str(key) + metric, statistics.stdev(observed), values['sd'])
        condition = 'Available' if cell['condition'] == 'source_available' else 'Withheld'
        policy = 'Task-conditioned' if cell['arm'] == 'task_conditioned' else 'Shared Top-5'
        f, ar = cell['metrics']['faithfulness'], cell['metrics']['ar_without_question_lead']
        line = ' & '.join([cell['task'][-1], condition, policy,
                          str(f['n_valid']) + '/' + str(len(group)),
                          str(cell['source_marker_in_actual_input']) + '/' + str(len(group)),
                          f'{f["mean"]:.4f}', f'{ar["mean"]:.4f}'])
        check('printed table row ' + str(key), line in table)
    source_path = root / 'code_inspection/final_challenge/summarize_challenge.py'
    source = source_path.read_bytes()
    freeze = read(root / 'documentation/new_question_freeze.json')
    pinned = next(v['sha256'] for v in freeze['files'] if v['basename'] == 'summarize_challenge.py')
    check('frozen paired-bootstrap function source', hashlib.sha256(source).hexdigest() == pinned)
    tree = ast.parse(source.decode('utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'paired')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), 'exec'), namespace)
    total_intervals = 0
    for contrast in result['paired_comparisons']:
        task_rows = [r for r in rows if r['intended_task'] == contrast['task']]
        if contrast['contrast'] == 'task_conditioned_minus_shared_pool':
            a, b = ('task_conditioned', contrast['condition']), ('shared_pool_plain_topk', contrast['condition'])
        else:
            a, b = (contrast['arm'], 'source_available'), (contrast['arm'], 'source_withheld')
        for metric, saved in contrast['metrics'].items():
            replayed = namespace['paired'](task_rows, a, b, metric)
            label = str((contrast['task'], a, b, metric))
            check('pairing ' + label, replayed['query_ids'] == saved['query_ids'] and replayed['paired_n'] == saved['paired_n'])
            near('difference ' + label, replayed['mean_difference'], saved['mean_difference'])
            for i in range(2):
                near('interval ' + str(i) + label, replayed['ci95'][i], saved['ci95'][i])
            total_intervals += 1
    return {'passed': True, 'check_count': len(checks), 'questions': len(bank), 'condition_records': len(rows),
            'table_rows': len(result['cells']), 'paired_intervals': total_intervals,
            'resamples_per_interval': 2000, 'scope': 'Saved scores and paired-statistic replay; no new metric scoring or scientific validation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
