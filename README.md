# ASD-Assistant: code and saved-score replay

This repository provides implementation code and statistical replay artifacts
for ASD-Assistant, a task-structured retrieval framework for autism research.
It contains every retained row in the included matched-retrieval and new-question
comparisons. It does not select new results, regenerate answers, or rescore them.

## Run the offline checks

Use Python 3.11 or 3.12. No API keys, model downloads or GPU are required.

    python -m pip install -r requirements-replay.txt
    python scripts/reproduce.py

The command checks the distribution manifest, 20 table sources, 148 matched-study
bootstrap intervals, 24 new-question paired intervals, external-model aggregates,
historical statistics, routing decisions and exact-text invariance. It also runs
18 replay/parity/corruption tests. These operations replay existing scores and
decisions, not scientific judgments. They do not cover every historical result.

See documentation/REPRODUCTION_COVERAGE.md for every compiled table/figure and
remaining blockers. A strict full-paper gate is available and currently fails:

    python scripts/reproduce.py --require-complete

Three recovered figures can be rendered with matplotlib and numpy installed:

    python scripts/render_historical_figures.py --out figure_replay

Use an empty output directory. Fonts/layout/PDF metadata may vary by host.

## Contents

| Directory | Contents |
| --- | --- |
| code_inspection/ | Retained runtime, evaluators, analysis, helpers and diagnostic runners |
| evidence/matched/ | All 594 matched-study score records, summaries and sensitivity results |
| evidence/new_question/ | 32 question records and all 128 diagnostic score records |
| evidence/external/ | External-provider coverage, per-question scores and aggregate summaries |
| evidence/historical/ | 246 historical bank records, 560 development score rows, 140 routing records and 96 text-fingerprint records |
| documentation/ | Retained prompts, configuration, training summaries and asset inventories |
| latex/tables/ | Twenty numeric table sources, including measured CSR and CVR |
| scripts/ | Offline replay and tests |

## Public data scope

JSON/JSONL inputs are explicit field projections of the retained artifacts.
All questions/condition rows, numeric values, pairing keys and ordering needed
for replay are preserved. Article excerpts, retrieved contexts, generated answer
text and full request traces are omitted from the public distribution. Thus an
outputs.jsonl file here is a numeric replay input, not a complete answer log.
PROVENANCE.json records source digests, distributed digests and retained fields.
The historical additions are documented separately in HISTORICAL_PROVENANCE.json.
The complete source artifacts are preserved separately; this repository does not
establish their external availability or redistribution permission.

The public files support statistical replay and implementation inspection. They
do not support re-evaluation of every answer against its source text. Model
weights and retrieval indexes are not included. See documentation/FRESH_RUN.md
for required assets and the distinction between a generic fresh run and exact
historical reproduction.

## Interpretation

Automated Faithfulness and citation checks are evidence-alignment proxies, not
expert validation of scientific conclusions. The 32-question diagnostic uses
assistant-authored reference constraints and is not a fully expert-validated
difficult benchmark. Historical training-row membership is not established by
the retained summaries. See documentation/BENCHMARK_STATUS.md.

No open-source license or permission to redistribute third-party material is
inferred. See RIGHTS_AND_SCOPE.md. Review correspondence, manuscript drafts,
credentials, private configuration and original Git history are not included.
