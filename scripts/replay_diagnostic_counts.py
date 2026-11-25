"""Check retained routing decisions and exact-text fingerprints; no new inference."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from replay_reported_tables import tex_rows


def verify(root):
    checks = []
    def check(label, condition):
        if not condition:
            raise AssertionError(label)
        checks.append(label)
    boundary = json.loads((root / 'evidence/historical/boundary_routing.json').read_text(encoding='utf-8'))
    groups = defaultdict(list)
    for row in boundary:
        groups[row['group']].append(row)
    check('140 boundary questions', len(boundary) == 140)
    rows = tex_rows(root / 'latex/tables/table_intent_boundary_diagnostic.tex')
    check('seven boundary groups', len(rows) == len(groups) == 7)
    for cells, members in zip(rows, groups.values()):
        check('boundary size '+cells[0], len(members) == int(cells[1]) == 20)
        values = [sum(r['raw_correct'] for r in members), sum(r['gated_correct'] for r in members),
                  sum(r['gated_task'] == 'abstain' for r in members)]
        for cell, n in zip(cells[2:], values):
            check('boundary numerator '+cells[0]+cell, cell.startswith(f'{n}/20 '))
    outputs = json.loads((root / 'evidence/historical/evidence_change_hashes.json').read_text(encoding='utf-8'))
    check('96 evidence-change outputs', len(outputs) == 96)
    groups = defaultdict(dict)
    for row in outputs:
        key = (row['task'], row['arm'], row['family_id'])
        check('unique condition '+str(key)+row['category'], row['category'] not in groups[key])
        groups[key][row['category']] = row
    for key, conditions in groups.items():
        check('all four conditions '+str(key), set(conditions) == {'consistent', 'conflict', 'population_difference', 'insufficient'})
    table = tex_rows(root / 'latex/tables/table_evidence_change_response.tex')[1:]
    check('evidence-change table rows', len(table) == 4)
    keys = [(t, a) for t in ('task2', 'task3') for a in ('retained_answering', 'direct_evidence')]
    for cells, (task, arm) in zip(table, keys):
        families = [v for (t, a, _), v in groups.items() if t == task and a == arm]
        check('six families '+task+arm, len(families) == 6)
        for i, field in enumerate(('raw_answer_sha256', 'final_answer_sha256'), 2):
            count = sum(len({v[c][field] for c in ('consistent', 'conflict', 'population_difference')}) == 1 for v in families)
            check('exact invariance '+task+arm+field, cells[i] == f'{count}/6')
    return {'passed': True, 'checks': len(checks), 'routing_rows': 140, 'evidence_change_rows':96,
            'scope': 'Saved routing decisions and exact-text hashes, not routing reruns or semantic judgments.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    print(json.dumps(verify(p.parse_args().root), indent=2))
