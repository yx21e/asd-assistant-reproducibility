"""Reaggregate retained scores and check reported table cells, without rescoring."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def csv_rows(root, rel):
    with (root / rel).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def tex_rows(path):
    return [[cell.strip() for cell in line.strip().removesuffix(r"\\").split("&")]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.rstrip().endswith(r"\\")
            and not line.lstrip().startswith((r"\textbf", r"\multicolumn"))]


def verify(root):
    checks = []

    def check(label, passed):
        if not passed:
            raise AssertionError(label)
        checks.append(label)

    def near(label, actual, expected):
        a, b = float(actual), float(expected)
        check(label, math.isfinite(a) and math.isfinite(b)
              and math.isclose(a, b, rel_tol=0, abs_tol=1e-10))

    matched = [json.loads(line) for line in
               (root / "evidence/matched/outputs.jsonl").read_text(encoding="utf-8").splitlines()
               if line.strip()]
    configs = ("task_conditioned", "shared_pool_retained_selection", "shared_pool_plain_topk")
    expected_groups = {(task, config) for task in ("task2", "task3") for config in configs}
    groups = defaultdict(list)
    for row in matched:
        groups[(row["task"], row["configuration"])].append(row)
    check("matched groups", set(groups) == expected_groups)
    for (task, config), rows in groups.items():
        check("matched denominator " + task + config, len(rows) == (100 if task == "task2" else 98))
        check("unique paired IDs " + task + config, len({r["example_id"] for r in rows}) == len(rows))
    for task in ("task2", "task3"):
        banks = [{(r["example_id"], r["query"]) for r in groups[(task, c)]} for c in configs]
        check("identical matched query sets " + task, all(b == banks[0] for b in banks))
    summaries = csv_rows(root, "evidence/matched/results.csv")
    results = {}
    for row in summaries:
        task, config, metric = row["task"], row["configuration"], row["metric"]
        key = (task, config, metric)
        check("unique summary " + str(key), key not in results)
        members = groups[(task, config)]
        vals = [r[metric] for r in members if r.get(metric) is not None]
        check("matched scored n " + str(key), len(vals) == int(row["n_scored"]))
        near("matched mean " + str(key), statistics.mean(vals), row["mean"])
        check("fallback count " + str(key),
              sum(bool(r["fallback_used"]) for r in members) == int(row["n_fallback"]))
        check("truncation count " + str(key),
              sum(r["raw_generation"]["untruncated_input_tokens"] > r["raw_generation"]["actual_input_tokens"]
                  for r in members) == int(row["n_truncated_prompts"]))
        results[key] = row
    sensitivity_map = {}
    for row in csv_rows(root, "evidence/matched/weight_sensitivity.csv"):
        task, config, w = row["task"], row["configuration"], float(row["G_weight"])
        vals = [w * r["G"] + (1-w) * r["S"] for r in groups[(task, config)]
                if r.get("G") is not None and r.get("S") is not None]
        check("sensitivity n " + str((task, config, w)), len(vals) == int(row["n_with_units"]))
        near("sensitivity mean " + str((task, config, w)), statistics.mean(vals), row["mean"])
        sensitivity_map[(task, config, w)] = row

    external = csv_rows(root, "evidence/external/per_query_scores.csv")
    providers = ("openai", "anthropic", "google", "deepseek")
    check("890 external and retained-local scores", len(external) == 890)
    ext_groups = defaultdict(list)
    group_fields = ("system", "task", "origin", "model", "response_model")
    for row in external:
        ext_groups[tuple(row[k] for k in group_fields)].append(row)
    external_summary = csv_rows(root, "evidence/external/summary_by_inference_regime.csv")
    summary_keys = [tuple(r[k] for k in group_fields) for r in external_summary]
    check("external summary unique complete groups", len(set(summary_keys)) == len(summary_keys)
          and set(summary_keys) == set(ext_groups))
    metrics = ("exact_match", "token_f1", "critical_token_recall", "reference_support_proxy",
               "ar_frozen_question_lead", "ar_without_question_lead", "equal_unit_support_proxy")
    for row in external_summary:
        key = tuple(row[k] for k in group_fields)
        members = ext_groups[key]
        check("external summary n " + str(key), len(members) == int(row["n"]))
        for metric in metrics:
            vals = [float(r[metric]) for r in members if r[metric] != ""]
            if row[metric] == "":
                check("absent external metric " + str(key) + metric, not vals)
            else:
                near("external mean " + str(key) + metric, statistics.mean(vals), row[metric])
    coverage = csv_rows(root, "evidence/external/coverage.csv")
    check("12 coverage groups", len(coverage) == 12 and
          {(r["system"], r["task"]) for r in coverage} ==
          {(p, t) for p in providers for t in ("task1", "task2", "task3")})
    for row in coverage:
        p, t = row["system"], row["task"]
        members = [r for r in external if r["system"] == p and r["task"] == t]
        n = 70 if t == "task3" else 60
        check("coverage actual n " + p + t, len(members) == n == int(row["available_answers"]))
        check("coverage expected n " + p + t, int(row["expected_queries"]) == n)
        check("coverage missing " + p + t, int(row["missing_answers"]) == 0 and row["complete"] == "True")
        archived = sum(r["origin"] == "archived_v2" for r in members)
        check("coverage archived " + p + t, archived == int(row["archived_answers"]))
        check("coverage supplemental " + p + t, len(members)-archived == int(row["supplemental_answers"]))
    for task, n in (("task1", 60), ("task2", 60), ("task3", 70)):
        banks = []
        for provider in providers:
            queries = [r["query"] for r in external if r["task"] == task and r["system"] == provider]
            check("unique external queries " + task + provider, len(queries) == len(set(queries)) == n)
            banks.append(set(queries))
        check("same questions across providers " + task, all(b == banks[0] for b in banks))

    tables = root / "latex/tables"
    keys = [(t, c) for t in ("task2", "task3") for c in configs]
    rows = tex_rows(tables / "table_single_pipeline_rag_combined.tex")
    check("main matched rows", len(rows) == len(keys))
    for cells, (task, config) in zip(rows, keys):
        check("main matched table n " + task + config, cells[2] == results[(task, config, "faithfulness")]["n_scored"])
        for col, metric in ((3, "faithfulness"), (4, "ar"), (5, "csr"), (6, "cvr"), (7, "ar_without_question_lead")):
            check("main matched table " + str((task, config, metric)),
                  cells[col] == f'{float(results[(task, config, metric)]["mean"]):.4f}')
    rows = tex_rows(tables / "table_matched_weight_sensitivity.tex")
    check("sensitivity table rows", len(rows) == len(keys))
    for cells, (task, config) in zip(rows, keys):
        for col, w in enumerate((0, .25, .5, .55, .75, 1), 2):
            check("sensitivity table " + str((task, config, w)),
                  cells[col] == f'{float(sensitivity_map[(task, config, w)]["mean"]):.4f}')
    rows = tex_rows(tables / 'table_single_pipeline_rag_baseline.tex')
    check('supplement matched rows', len(rows) == len(keys))
    for cells, (task, config) in zip(rows, keys):
        for col, metric in enumerate(('equal_unit_faithfulness', 'csr', 'cvr'), 2):
            check('supplement matched cell '+str((task, config, metric)),
                  cells[col] == f'{float(results[(task, config, metric)]["mean"]):.4f}')
        entry = results[(task, config, 'faithfulness')]
        for col, field in ((5, 'n_fallback'), (6, 'n_truncated_prompts')):
            check('supplement branch count '+str((task, config, field)),
                  cells[col] == f'{entry[field]}/{entry["n_scored"]}')
    coverage_rows = tex_rows(tables / 'table_external_coverage.tex')
    check('coverage table rows', len(coverage_rows) == 8)
    regimes = [('openai', 'Archived'), ('openai', 'Supplemental'), ('anthropic', 'Archived'),
               ('anthropic', 'Supplemental'), ('google', 'Supplemental'), ('deepseek', 'Archived')]
    for cells, (provider, regime) in zip(coverage_rows[:6], regimes):
        counts = [sum(r['system'] == provider and r['task'] == task and
                      (r['origin'] == 'archived_v2') == (regime == 'Archived') for r in external)
                  for task in ('task1', 'task2', 'task3')]
        check('coverage table '+provider+regime, cells[3:] == [str(n) for n in counts])
    check('coverage external totals', coverage_rows[6][3:] == ['240', '240', '280'])
    check('coverage separate local totals', coverage_rows[7][3:] == ['---', '60', '70'])
    synthesis_rows = [r for r in external_summary if r["task"] in ("task2", "task3")]
    synthesis = {(r["task"], r["system"]): r for r in synthesis_rows}
    check("unambiguous synthesis regimes", len(synthesis) == len(synthesis_rows))
    rows = tex_rows(tables / "table_external_baselines_numeric.tex")
    check("main external rows", len(rows) == 10)
    for cells, (task, provider) in zip(rows, [(t, p) for t in ("task2", "task3") for p in ("local_historical_final",)+providers]):
        for col, metric in enumerate(metrics[3:6], 2):
            check("main external table " + str((task, provider, metric)),
                  cells[col] == f'{float(synthesis[(task, provider)][metric]):.4f}')
    task1_text = (root / "documentation/task1_external_summary.md").read_text(encoding="utf-8")
    names = dict(zip(providers, ("OpenAI", "Claude", "Gemini", "DeepSeek")))
    for row in external_summary:
        if row["task"] == "task1":
            regime = "Archived" if row["origin"] == "archived_v2" else "Supplemental"
            cells = [names[row["system"]], regime, row["n"]] + [f'{float(row[k]):.4f}' for k in metrics[:3]]
            check("Task 1 external report " + row["system"] + regime,
                  "| " + " | ".join(cells) + " |" in task1_text)
    return {"passed": True, "checks": len(checks), "matched_rows": len(matched),
            "external_and_local_rows": len(external), "external_regimes": len(external_summary),
            "scope": "Saved-score aggregation, pairing, coverage, five table files and Task 1 external summary.",
            "metric_rescoring": False, "fresh_inference": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = verify(args.root)
    text = json.dumps(result, indent=2)
    if args.report:
        args.report.write_text(text + "\n", encoding="utf-8")
    print(text)
