"""Replay historical saved-score statistics. Does not regenerate answers or scores."""
import argparse
import json
import random
import re
import statistics
from pathlib import Path
from replay_reported_tables import tex_rows


def verify(root):
    data = {name: [json.loads(s) for s in (root / 'evidence/historical' / (name+'.jsonl')).read_text(encoding='utf-8').splitlines()]
            for name in ('task2_curated', 'task2_large', 'task3_curated', 'task3_large')}
    checks = []

    def check(label, passed):
        if not passed:
            raise AssertionError(label)
        checks.append(label)

    def mean(rows, metric):
        return statistics.mean(r[metric] for r in rows)

    def fmt(value):
        return f'{value:.4f}'

    def meanstd(rows, metric):
        return r'\meanstd{' + fmt(mean(rows, metric)) + '}{' + fmt(statistics.stdev(r[metric] for r in rows)) + '}'

    def check_group_table(file, subsets, offset):
        rows = tex_rows(root / 'latex/tables' / file)
        check(file+' row count', len(rows) == len(subsets))
        for i, (cells, members) in enumerate(zip(rows, subsets)):
            expected = [str(len(members)), meanstd(members, 'faithfulness'), meanstd(members, 'ar'),
                        fmt(mean(members, 'csr')), fmt(mean(members, 'cvr'))]
            check(file+' row '+str(i), cells[offset:] == expected)

    for (name, rows), expected_n in zip(data.items(), (20, 100, 28, 98)):
        check(name+' size', len(rows) == expected_n)
        check(name+' unique IDs', len({r['example_id'] for r in rows}) == expected_n)
    t2, t3 = data['task2_large'], data['task3_large']
    check_group_table('table_task2_group_robustness.tex', [t2]+[[r for r in t2 if r['tier'] == t]
        for t in ('core', 'extended', 'weak_support')]+[[r for r in t2 if r['autism_doc'] == b] for b in (True, False)], 1)
    topics = ('clinical_features', 'diagnosis_screening', 'biomarkers', 'epidemiology_population',
              'methods_study_design', 'risk_pathogenesis', 'general_autism')
    check_group_table('table_task3_topic_robustness.tex', [[r for r in t3 if r['topic'] == t] for t in topics], 1)
    check_group_table('table_task3_prompt_topic_matrix.tex',
        [[r for r in t3 if r['query'].strip().lower().startswith(p)] for p in ('what', 'how', 'summarize')], 1)
    large_table = tex_rows(root / 'latex/tables/table_large_sample_robustness.tex')
    check('larger bank rows', len(large_table) == 3)  # Includes its plain-text header.
    for cells, members in zip(large_table[1:], (t2, t3)):
        check('larger bank n '+cells[0], cells[1] == str(len(members)))
        check('larger bank metrics '+cells[0], cells[4:] == [fmt(mean(members, k)) for k in ('faithfulness', 'ar', 'csr', 'cvr')])
    routes = tex_rows(root / 'latex/tables/table_routing_diagnostics.tex')
    check('routing rows', len(routes) == 4)
    for cells, members in zip(routes, data.values()):
        correct = sum(r['predicted_task'] == r['expected_task'] for r in members)
        check('routing '+cells[0], cells[1:3] == [str(len(members)), f'{correct}/{len(members)}'])
        check('routing abstentions '+cells[0], cells[3] == str(sum(r['predicted_task'] == 'reject' for r in members)))

    def bank(label):
        task = 'task2' if '2' in label else 'task3'
        return data[task + ('_curated' if 'curated' in label.lower() else '_large')]

    for file in ('table_threshold_sweep.tex', 'table_error_rate_diagnostics.tex'):
        table = tex_rows(root / 'latex/tables' / file)
        check(file+' number of rows', len(table) == (11 if file == 'table_threshold_sweep.tex' else 8))
        for i, cells in enumerate(table):
            members = bank(cells[0])
            metric = 'faithfulness' if cells[1].startswith('Faith') else cells[1].lower()
            below = sum(r[metric] < float(cells[2]) for r in members)
            check(file+str(i), cells[3:] == [str(len(members)), f'{below}/{len(members)}'])

    def quantile(values, q):
        values = sorted(values)
        pos = (len(values)-1)*q
        lo = int(pos)
        hi = min(lo+1, len(values)-1)
        return values[lo]*(1-(pos-lo))+values[hi]*(pos-lo)

    for i, cells in enumerate(tex_rows(root / 'latex/tables/table_percentile_diagnostics.tex')):
        members, metric = bank(cells[0]), cells[1].lower()
        check('percentile '+str(i), cells[2:] == [str(len(members))]+[
            fmt(quantile([r[metric] for r in members], q)) for q in (.1, .25, .5, .75, .9)])
    bootstrap = tex_rows(root / 'latex/tables/table_bootstrap_stability.tex')
    check('bootstrap rows', len(bootstrap) == 4)
    for cells, members in zip(bootstrap, data.values()):
        rng = random.Random(20260501)
        samples = [[members[rng.randrange(len(members))] for _ in members] for _ in range(1000)]
        vals = {k: [mean(sample, k) for sample in samples] for k in ('faithfulness', 'ar')}
        expected = [r'\meanstd{'+fmt(statistics.mean(vals[k]))+'}{'+fmt(statistics.stdev(vals[k]))+'}' for k in vals]
        check('bootstrap Faithfulness/AR '+cells[0], cells[1:5] == [str(len(members)), '1000']+expected)
        # The current table's CSR/CVR columns are point estimates, not bootstrap summaries.
        check('bootstrap-table citation point estimates '+cells[0], cells[5:] == [fmt(mean(members, k)) for k in ('csr','cvr')])

    development = json.loads((root / 'evidence/historical/development_scores.json').read_text(encoding='utf-8'))
    for group, file in (('main', 'table_development_diagnostics.tex'), ('supplement', 'table_numeric_ablation.tex')):
        rows = tex_rows(root / 'latex/tables' / file)
        selected = [r for r in development if r['group'] == group]
        check('development rows '+group, len(rows) == len(selected))
        for i, (cells, record) in enumerate(zip(rows, selected)):
            scores = record['scores']
            check('development size '+group+str(i), len(scores) == record['n'])
            check('development mean '+group+str(i), cells[-3:] == [str(len(scores)),
                fmt(mean(scores, 'faithfulness')), fmt(mean(scores, 'ar'))])
    return {'passed': True, 'checks': len(checks), 'historical_bank_rows': sum(map(len, data.values())),
            'development_rows': sum(len(r['scores']) for r in development),
            'scope': 'Saved-score statistics for eleven historical table files, not raw-text rescoring.',
            'warnings': ['Bootstrap table caption needs to distinguish CSR/CVR point estimates.',
                         'Multifactor reference-count rows are excluded: labels/SDs do not match the recovered aggregation.',
                         'Historical curated and development contexts are not retained in these records.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
