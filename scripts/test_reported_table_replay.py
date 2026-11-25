"""Ensure the added verifiers reject corrupted score/coverage/table inputs."""
import csv
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from replay_reported_tables import verify

ROOT = Path(__file__).resolve().parents[1]


class CorruptionChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=ROOT)
        self.root = Path(self.tmp.name)
        for folder in ('evidence', 'latex', 'documentation'):
            shutil.copytree(ROOT / folder, self.root / folder)

    def tearDown(self):
        self.tmp.cleanup()

    def change_csv(self, rel, field, value):
        path = self.root / rel
        with path.open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        rows[0][field] = value
        with path.open('w', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    def test_reference_passes(self):
        self.assertTrue(verify(self.root)['passed'])

    def test_external_score(self):
        self.change_csv('evidence/external/per_query_scores.csv', 'exact_match', '0.123456')
        with self.assertRaises(AssertionError):
            verify(self.root)

    def test_coverage_count(self):
        self.change_csv('evidence/external/coverage.csv', 'available_answers', '59')
        with self.assertRaises(AssertionError):
            verify(self.root)

    def test_matched_csr(self):
        path = self.root / 'evidence/matched/outputs.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        rows[0]['csr'] = .123456
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaises(AssertionError):
            verify(self.root)

    def test_duplicate_pair(self):
        path = self.root / 'evidence/matched/outputs.jsonl'
        rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
        group = [i for i, r in enumerate(rows) if (r['task'],r['configuration']) == (rows[0]['task'],rows[0]['configuration'])]
        rows[group[1]]['example_id'] = rows[0]['example_id']
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows), encoding='utf-8')
        with self.assertRaises(AssertionError):
            verify(self.root)

    def test_table_cell(self):
        path = self.root / 'latex/tables/table_single_pipeline_rag_combined.tex'
        text = path.read_text(encoding='utf-8')
        self.assertIn('0.8646', text)
        path.write_text(text.replace('0.8646', '0.9646', 1), encoding='utf-8')
        with self.assertRaises(AssertionError):
            verify(self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
