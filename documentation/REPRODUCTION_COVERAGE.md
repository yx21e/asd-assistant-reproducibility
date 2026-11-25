# Reproduction coverage: current paper

This is a per-result audit, not a claim of complete end-to-end reproduction.
The public runner checks 20 of 28 table sources from saved per-query scores/decisions or exact-text hashes.
Three recovered figure renderers run; matching numeric labels do not prove pixel-identical figures.
Locally, all 198 larger-bank Faithfulness/AR scores and 246 historical CSR/CVR scores were recalculated from retained answer text.
All 890 external-comparison/local-reference rows were also rescored locally; all recorded numeric metrics matched exactly.
Those local text/index assets are not included publicly. No new inference or training was performed.

| Item | Level verified | Note |
| --- | --- | --- |
| main figure 1 | partial | Source scores traced, but exact combined-scorecard rendering code unlocated; Task 1 original answer logs missing. |
| main table 1 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| main table 2 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| main table 3 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| main figure 2 | rendered | Recovered renderer executed; all extracted numeric labels match retained PDF. |
| main figure 3 | conceptual_not_applicable | Conceptual diagram, not a numerical result; original asset retained, renderer not verified. |
| supplement table 1 | partial | Task 1 counts are summary-only; citation totals replay, but some historical CSR descriptions refer to a different script. |
| supplement table 2 | recorded_audit_only | Overlap sources retained; all original train/bank comparisons not rerun in this audit. |
| supplement table 3 | partial | Synthesis bank sizes verified; construction exclusions not fully re-executed. |
| supplement table 4 | configuration_record | Recorded settings are provided, not a reproduced optimizer history. |
| supplement table 5 | configuration_record | Index inventory and hashes retained; full rebuild not executed. |
| supplement table 6 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 7 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 8 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 9 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 10 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 11 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 12 | summary_to_table | Recovered summary and rendering agree; per-answer metric recomputation remains unverified. |
| supplement figure 1 | rendered | Recovered renderer executed; numeric labels match. Optional-stack series are traced to retained summaries only. |
| supplement table 13 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 14 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 15 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement figure 2 | rendered | Recovered renderer executed from saved scores; numeric labels match. Does not establish pixel-identical curve parity. |
| supplement table 16 | saved_score_replay | Faithfulness/AR are bootstrap mean/SD; CSR/CVR cells reproduce point estimates. Caption does not distinguish these. |
| supplement table 17 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 18 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 19 | discrepancy | Reference strata contain 66/34 records. Saved citation counts are 2/4 entries, table labels say 1/2 refs. Second stratum sample SDs replay as 0.0164/0.0052, not 0.0161/0.0051. |
| supplement table 20 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 21 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 22 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 23 | qualitative_not_applicable | Qualitative taxonomy; no numerical experiment to replay. |
| supplement table 24 | saved_score_replay | See numerical replay scripts; no fresh inference. |
| supplement table 25 | saved_score_replay | See numerical replay scripts; no fresh inference. |

## Remaining blockers

- Original Task 1 generated-answer logs are unavailable; summary/reference files cannot recreate conditional EM/F1.
- The 48 curated answers and the 24 historical development configurations lack retained generation contexts; full historical Faithfulness rescoring is not established.
- Supplement Table 19 has unresolved citation-count labeling and two reference-stratum standard deviations.
- Supplement Table 16 mixes bootstrap summaries and citation point estimates without distinguishing them in its caption.
- Supplement Table 1 historical citation-support descriptions do not uniformly describe the replayed frozen scorer.
- The original combined-scorecard renderer is unlocated; conceptual-diagram renderer is not verified.
- Full fresh inference/training is not reproduced; public distribution lacks model weights, indexes, answer/source text and complete historical training provenance.

Strict full-coverage gate:

    python scripts/reproduce.py --require-complete

This command intentionally exits nonzero while any blocker remains. Do not remove a blocker merely because saved-score checks pass.
