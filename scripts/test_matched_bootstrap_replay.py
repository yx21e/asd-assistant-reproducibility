"""Local parity and corruption tests; never run historical analysis main()."""
import argparse
import ast
import copy
import importlib.util
import json
import unittest
from pathlib import Path

import numpy as np

PARSER = argparse.ArgumentParser()
PARSER.add_argument("--package",required=True,type=Path)
PARSER.add_argument("--original-analysis",required=True,type=Path)
ARGS,REST = PARSER.parse_known_args()
SPEC = importlib.util.spec_from_file_location("replay",Path(__file__).with_name("replay_matched_bootstrap.py"))
REPLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPLAY)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads((ARGS.package/"documentation/bootstrap_protocol.json").read_text())
        with (ARGS.package/"evidence/matched/outputs.jsonl").open(encoding="utf-8") as f:
            cls.rows = [json.loads(line) for line in f if line.strip()]
        original_tree = ast.parse(ARGS.original_analysis.read_text(encoding="utf-8-sig"))
        functions = [node for node in original_tree.body if isinstance(node,ast.FunctionDef) and node.name == "interval"]
        if len(functions) != 1:
            raise AssertionError("Cannot identify one original interval function")
        namespace = {"np":np}
        exec(compile(ast.Module(body=functions,type_ignores=[]),str(ARGS.original_analysis),"exec"),namespace)
        cls.original_interval = staticmethod(namespace["interval"])

    def test_complete_replay(self):
        report = REPLAY.replay(ARGS.package)
        self.assertTrue(report["summary"]["passed"])
        self.assertEqual(report["summary"]["total_intervals"],148)
        self.assertEqual(report["summary"]["printed_supplement_rows_checked"],7)

    def test_original_function_parity(self):
        cases = ([0.0]*100,[1.0],[.1,.7,-.2,.9],np.linspace(-1,1,98),[.01,-.02,.03,0,.02,-.01,.04])
        for values in cases:
            self.assertEqual(REPLAY.interval(values,self.protocol),self.original_interval(values))

    def test_missing_record_rejected(self):
        with self.assertRaisesRegex(ValueError,"594"):
            REPLAY.validate_rows(self.rows[:-1],self.protocol)

    def test_duplicate_record_rejected(self):
        rows = list(self.rows)
        rows[1] = rows[0]
        with self.assertRaisesRegex(ValueError,"Duplicate"):
            REPLAY.validate_rows(rows,self.protocol)

    def test_unpaired_id_rejected(self):
        rows = list(self.rows)
        i = next(i for i,r in enumerate(rows) if r["configuration"] == "shared_pool_plain_topk")
        rows[i] = dict(rows[i],example_id="UNPAIRED_TEST_ID")
        with self.assertRaisesRegex(ValueError,"Unpaired"):
            REPLAY.validate_rows(rows,self.protocol)

    def test_missing_metric_rejected(self):
        rows = list(self.rows)
        rows[0] = dict(rows[0],faithfulness=None)
        with self.assertRaisesRegex(ValueError,"Missing/nonfinite"):
            REPLAY.validate_rows(rows,self.protocol)

    def test_missing_topic_rejected(self):
        rows = list(self.rows)
        i = next(i for i,r in enumerate(rows) if r["task"] == "task3")
        rows[i] = dict(rows[i],topic=None)
        with self.assertRaisesRegex(ValueError,"topic"):
            REPLAY.validate_rows(rows,self.protocol)

    def test_order_change_rejected(self):
        rows = list(self.rows)
        rows[0],rows[1] = rows[1],rows[0]
        with self.assertRaisesRegex(ValueError,"order"):
            REPLAY.validate_rows(rows,self.protocol)

    def test_modified_interval_rejected(self):
        expected = [{"id":"a","low":"0.1","high":"0.2"}]
        actual = [{"id":"a","low":.1,"high":.21}]
        with self.assertRaisesRegex(ValueError,"Replay mismatch"):
            REPLAY.compare_rows(expected,actual,("id",),"test",1e-12)

    def test_duplicate_summary_rejected(self):
        with self.assertRaisesRegex(ValueError,"Duplicate summary"):
            REPLAY.compare_rows([{"id":"a","v":"1"}]*2,[{"id":"a","v":1}],("id",),"test",1e-12)

    def test_empty_and_nonfinite_samples_rejected(self):
        for values in ([],[float("nan")],[float("inf")]):
            with self.assertRaises(ValueError):
                REPLAY.interval(values,self.protocol)

    def test_rng_is_reset_not_shared(self):
        values = list(range(7))
        first = REPLAY.interval(values,self.protocol)
        REPLAY.interval(list(range(100)),self.protocol)
        self.assertEqual(first,REPLAY.interval(values,self.protocol))


if __name__ == "__main__":
    unittest.main(argv=[__file__]+REST,verbosity=2)
