"""Reject stale denominators, labels and SDs rather than silently accepting them."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from replay_revision_tables import verify

ROOT=Path(__file__).resolve().parents[1]


class RevisionTables(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.root=Path(self.tmp.name)
        for name in ('evidence','latex'):shutil.copytree(ROOT/name,self.root/name)

    def tearDown(self):
        self.tmp.cleanup()

    def mutate(self,name,old,new):
        p=self.root/'latex/tables'/name
        text=p.read_text(); self.assertIn(old,text)
        p.write_text(text.replace(old,new),encoding='utf-8')

    def test_valid(self):
        self.assertTrue(verify(self.root)['passed'])

    def test_stale_csr_denominator(self):
        self.mutate('table_metric_audit.tex','Task 2 curated CSR & 20/20','Task 2 curated CSR & 48/48')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_stale_sd(self):
        self.mutate('table_task2_multifactor_robustness.tex',r'\meanstd{0.8630}{0.0164}',r'\meanstd{0.8630}{0.0161}')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_entries_mislabeled_as_papers(self):
        self.mutate('table_task2_multifactor_robustness.tex','Distinct PMIDs & 1 &','Distinct PMIDs & 2 &')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_missing_audit_query(self):
        p=self.root/'evidence/historical/citation_unit_counts.json'
        data=json.loads(p.read_text());data['task3_large'].pop()
        p.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaises(AssertionError):verify(self.root)

    def test_later_task1_audit_cannot_replace_original(self):
        p=self.root/'evidence/historical/task1_recorded_summary.json'
        data=json.loads(p.read_text());data['values']['samples']=1074
        p.write_text(json.dumps(data),encoding='utf-8')
        with self.assertRaises(AssertionError):verify(self.root)


if __name__=='__main__':unittest.main()
