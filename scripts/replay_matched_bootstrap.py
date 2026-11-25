"""Reproduce retained matched-study intervals, not model outputs or metric scores."""
import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("faithfulness", "ar", "ar_without_question_lead", "equal_unit_faithfulness", "G", "S", "csr", "cvr")
CONFIGS = ("task_conditioned", "shared_pool_retained_selection", "shared_pool_plain_topk")


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def order_hash(rows):
    keys = [[r["task"], r["example_id"], r["configuration"]] for r in rows]
    return hashlib.sha256(json.dumps(keys, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()


def validate_rows(rows, protocol):
    if len(rows) != 594:
        raise ValueError("Expected all 594 retained matched outputs; no silent row removal")
    groups = defaultdict(list)
    seen = set()
    for r in rows:
        key = (r["task"], r["example_id"], r["configuration"])
        if key in seen:
            raise ValueError("Duplicate output key: " + repr(key))
        seen.add(key)
        if r["task"] not in ("task2", "task3") or r["configuration"] not in CONFIGS:
            raise ValueError("Unexpected task/configuration")
        if r.get("status") != "ok":
            raise ValueError("The retained full bank contains successful outputs only; changed status requires audit")
        for metric in METRICS:
            value = r.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Missing/nonfinite metric: " + repr(key) + "/" + metric)
        if r["task"] == "task3" and not r.get("topic"):
            raise ValueError("Task 3 topic label missing")
        groups[(r["task"], r["configuration"])].append(r)
    for task,n in (("task2",100),("task3",98)):
        primary = {r["example_id"] for r in groups[(task,"task_conditioned")]}
        for config in CONFIGS:
            group = groups[(task,config)]
            if len(group) != n or {r["example_id"] for r in group} != primary:
                raise ValueError("Unpaired or incomplete query bank: " + task + "/" + config)
    if order_hash(rows) != protocol["input_order_sha256"]:
        raise ValueError("Retained row order changed; finite seeded bootstrap draws depend on that order")
    return groups


def interval(values, protocol):
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or not len(vector) or not np.isfinite(vector).all():
        raise ValueError("A nonempty finite one-dimensional sample is required")
    # The retained analysis restarts the RNG for every interval, not every study.
    rng = np.random.default_rng(protocol["seed"])
    samples = vector[rng.integers(0, len(vector), size=(protocol["resamples"], len(vector)))].mean(axis=1)
    return (float(vector.mean()),
            float(np.quantile(samples, .025, method=protocol["quantile_method"])),
            float(np.quantile(samples, .975, method=protocol["quantile_method"])))


def compare_rows(expected, reproduced, keys, label, tolerance):
    expected_map = {tuple(r[k] for k in keys):r for r in expected}
    actual_map = {tuple(str(r[k]) for k in keys):r for r in reproduced}
    if len(expected_map) != len(expected) or len(actual_map) != len(reproduced):
        raise ValueError("Duplicate summary keys: " + label)
    if set(expected_map) != set(actual_map):
        raise ValueError("Summary coverage mismatch: " + label)
    max_error, numeric_checks = 0.0, 0
    for key,wanted in expected_map.items():
        actual = actual_map[key]
        if set(wanted) != set(actual):
            raise ValueError("Summary schema mismatch: " + label)
        for field,value in wanted.items():
            if field in keys:
                continue
            a,b = float(actual[field]),float(value)
            if not math.isfinite(a) or not math.isfinite(b):
                raise ValueError("Nonfinite archived/replayed statistic")
            error = abs(a-b)
            max_error = max(max_error,error)
            numeric_checks += 1
            if error > tolerance:
                raise ValueError(f"Replay mismatch {label}/{key}/{field}: actual={a!r}, archived={b!r}, error={error!r}")
    return {"rows":len(reproduced),"numeric_checks":numeric_checks,"max_absolute_error":max_error}


def check_printed_table(path, paired):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.rstrip().endswith(r"\\") and not line.lstrip().startswith((r"\textbf",r"\multicolumn")):
            rows.append([c.strip() for c in line.strip().removesuffix(r"\\").split("&")])
    keys = [("task2",m) for m in ("faithfulness","ar","ar_without_question_lead")]
    keys += [("task3",m) for m in ("faithfulness","ar","ar_without_question_lead","equal_unit_faithfulness")]
    selected = {(r["task"],r["metric"]):r for r in paired
                if r["contrast"] == "task_conditioned_minus_shared_pool_plain_topk"}
    if len(rows) != len(keys):
        raise ValueError("Unexpected printed paired-interval table structure")
    for cells,key in zip(rows,keys):
        r = selected[key]
        difference = r["paired_mean_difference"]
        text = f"{difference:+.4f}" if difference != 0 else "0.0000"
        query = f'[{r["paired_query_CI_low"]:.4f}, {r["paired_query_CI_high"]:.4f}]'
        topic = f'[{r["cluster_CI_low"]:.4f}, {r["cluster_CI_high"]:.4f}]' if key[0] == "task3" else "---"
        if cells[2:] != [text,query,topic]:
            raise ValueError("Printed interval differs from replay: " + repr(key))
    return len(rows)


def replay(root):
    protocol = json.loads((root/"documentation/bootstrap_protocol.json").read_text(encoding="utf-8"))
    if (protocol["seed"], protocol["resamples"], protocol["quantile_method"]) != (20261112, 5000, "linear"):
        raise ValueError("Protocol differs from the inspected retained analysis; no tuning overrides are accepted")
    with (root/"evidence/matched/outputs.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    groups = validate_rows(rows, protocol)
    aggregate, paired, sensitivity = [], [], []
    for (task,arm),group in groups.items():
        for metric in METRICS:
            mean,low,high = interval([r[metric] for r in group],protocol)
            aggregate.append({"task":task,"configuration":arm,"metric":metric,
                "n_requested":len(group),"n_scored":len(group),"n_errors":0,"n_abstentions":0,
                "mean":mean,"query_bootstrap_low":low,"query_bootstrap_high":high,
                "n_fallback":sum(bool(r.get("fallback_used")) for r in group),
                "n_truncated_prompts":sum(r["raw_generation"].get("untruncated_input_tokens",0) >
                    r["raw_generation"].get("actual_input_tokens",0) for r in group)})
    for task in ("task2","task3"):
        primary = {r["example_id"]:r for r in groups[(task,"task_conditioned")]}
        for arm in CONFIGS[1:]:
            comparator = {r["example_id"]:r for r in groups[(task,arm)]}
            for metric in METRICS:
                ids = list(primary)
                differences = [primary[i][metric]-comparator[i][metric] for i in ids]
                mean,low,high = interval(differences,protocol)
                clusters = defaultdict(list)
                for key,value in zip(ids,differences):
                    clusters[primary[key].get("topic") or key].append(value)
                macro,clow,chigh = interval([float(np.mean(v)) for v in clusters.values()],protocol)
                paired.append({"task":task,"contrast":"task_conditioned_minus_"+arm,"metric":metric,
                    "n_pairs":len(ids),"n_query_clusters":len(clusters),"paired_mean_difference":mean,
                    "paired_query_CI_low":low,"paired_query_CI_high":high,"macro_cluster_difference":macro,
                    "cluster_CI_low":clow,"cluster_CI_high":chigh})
        for arm in CONFIGS:
            group = groups[(task,arm)]
            for weight in (0,.25,.5,.55,.75,1):
                mean,low,high = interval([weight*r["G"]+(1-weight)*r["S"] for r in group],protocol)
                sensitivity.append({"task":task,"configuration":arm,"G_weight":weight,
                    "n_with_units":len(group),"mean":mean,"CI_low":low,"CI_high":high})
    base = root/"evidence/matched"
    comparisons = {
        "aggregate":compare_rows(read_csv(base/"results.csv"),aggregate,("task","configuration","metric"),"aggregate",protocol["absolute_tolerance"]),
        "paired":compare_rows(read_csv(base/"paired_comparisons.csv"),paired,("task","contrast","metric"),"paired",protocol["absolute_tolerance"]),
        "weight_sensitivity":compare_rows(read_csv(base/"weight_sensitivity.csv"),sensitivity,("task","configuration","G_weight"),"weight",protocol["absolute_tolerance"])}
    table_rows = check_printed_table(root/"latex/tables/table_single_pipeline_rag_ci.tex",paired)
    summary = {"passed":True,"numpy_version":np.__version__,"saved_output_count":len(rows),
        "resamples_per_interval":protocol["resamples"],"seed":protocol["seed"],"rng_restarted_for_each_interval":True,
        "group_intervals":len(aggregate),"paired_query_intervals":len(paired),"paired_cluster_intervals":len(paired),
        "weight_sensitivity_intervals":len(sensitivity),"total_intervals":len(aggregate)+2*len(paired)+len(sensitivity),
        "printed_supplement_rows_checked":table_rows,"numeric_checks":sum(v["numeric_checks"] for v in comparisons.values()),
        "max_absolute_difference":max(v["max_absolute_error"] for v in comparisons.values()),
        "absolute_tolerance":protocol["absolute_tolerance"],"new_model_outputs":0,"metric_rescoring":False,
        "scope":"Reproduces the retained matched-study bootstrap numerically; does not establish independence, expert validity or generalization."}
    return {"summary":summary,"comparisons":comparisons,
            "reproduced":{"matched_results":aggregate,"paired_comparisons":paired,"weight_sensitivity":sensitivity}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output",type=Path,help="Optional new JSON report beneath the package's reproduced/ directory")
    args = parser.parse_args()
    report = replay(args.root)
    if args.output:
        output = args.output.resolve()
        if not output.is_relative_to((args.root/"reproduced").resolve()) or output.exists():
            parser.error("Output must be a new file under the package's reproduced/ directory")
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(report["summary"],indent=2))
