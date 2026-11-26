# Reproduction coverage: current revision

Manuscript table state: round_0010. No per-query scores were changed.
Replaying saved values is distinct from reproducing inference or scientific judgments.

| Item | Verification scope | Public replay |
| --- | --- | --- |
| main figure 1 | reconstructed_numeric_panels: New renderer uses retained score rows and recorded Task 1 summary. Not the recovered original layout; not pixel-identical. | True |
| main table 1 | saved_score_replay:  | True |
| main table 2 | saved_score_replay:  | True |
| main table 3 | saved_score_replay:  | True |
| main figure 2 | rendered: Recovered renderer executed; all extracted numeric labels match retained PDF. | True |
| main figure 3 | conceptual_not_applicable: Conceptual vector asset retained with checksum in the local review package; no numerical experiment to recompute. Editable original drawing source not established. | False |
| supplement table 1 | synthesis_count_replay_task1_summary_only: Task 2/3 frozen-scorer units replay. Original Task 1 generated-output log is still unavailable. | True |
| supplement table 2 | recorded_audit_only: Overlap sources retained; all original train/bank comparisons not rerun in this audit. | False |
| supplement table 3 | partial: Synthesis bank sizes verified; construction exclusions not fully re-executed. | False |
| supplement table 4 | configuration_record: Recorded settings are provided, not a reproduced optimizer history. | False |
| supplement table 5 | configuration_record: Index inventory and hashes retained; full rebuild not executed. | False |
| supplement table 6 | saved_score_replay:  | True |
| supplement table 7 | saved_score_replay:  | True |
| supplement table 8 | saved_score_replay:  | True |
| supplement table 9 | saved_score_replay:  | True |
| supplement table 10 | saved_score_replay:  | True |
| supplement table 11 | saved_score_replay:  | True |
| supplement table 12 | summary_to_table: Recovered summary and rendering agree; per-answer metric recomputation remains unverified. | False |
| supplement figure 1 | rendered: Recovered renderer executed; numeric labels match. Optional-stack series are traced to retained summaries only. | True |
| supplement table 13 | saved_score_replay:  | True |
| supplement table 14 | saved_score_replay:  | True |
| supplement table 15 | saved_score_replay:  | True |
| supplement figure 2 | rendered: Recovered renderer executed from saved scores; numeric labels match. Does not establish pixel-identical curve parity. | True |
| supplement table 16 | saved_score_replay: Caption now distinguishes bootstrap F/AR summaries from original query-averaged CSR/CVR. | True |
| supplement table 17 | saved_score_replay:  | True |
| supplement table 18 | saved_score_replay:  | True |
| supplement table 19 | saved_score_replay: All seven groups replay; distinct-PMID groups and two corrected sample SDs verified. | True |
| supplement table 20 | saved_score_replay:  | True |
| supplement table 21 | saved_score_replay:  | True |
| supplement table 22 | saved_score_replay:  | True |
| supplement table 23 | qualitative_not_applicable: Qualitative taxonomy; no numerical experiment to replay. | False |
| supplement table 24 | saved_score_replay:  | True |
| supplement table 25 | saved_score_replay:  | True |

## Remaining historical limits

- Original Task 1 generated-answer logs are unavailable; summary/reference files cannot recreate conditional EM/F1.
- The 48 curated answers and the 24 historical development configurations lack retained generation contexts; full historical Faithfulness rescoring is not established.
- Full fresh inference/training is not reproduced; public distribution lacks model weights, indexes, answer/source text and complete historical training provenance.
- The original combined-layout and editable conceptual-diagram sources are not established; a new numeric reconstruction and unchanged original diagram asset are supplied separately.
