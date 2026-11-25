# Difficult held-out benchmark: completed work and remaining scope

## Completed

- System/corpus fixed before new-question authoring; questions/reference
  constraints frozen before outputs, with retained hashes.
- 16 new questions per task, 32 total; no canonical exact duplicates against
  3,925 recorded historical questions, plus inspection of three nearest semantic
  neighbors per question.
- Two retrieval policies and two source conditions: all 128 outputs retained,
  including wrong routes, missing evidence, and lower-scoring cases.
- Offline replay includes route/source coverage, saved scores, and paired intervals.

## Remaining

The requested difficult benchmark also calls for conflicting-evidence coverage.
The source audit obtained two review full texts and five primary full texts;
two primary full texts were unavailable. Of nine candidate comparisons, two
showed heterogeneous directions, one was consistent, one non-comparable, and
five remained pending primary text. None was verified as a comparable
direct-conflict primary-study pair. Mixed populations or subgroup-versus-overall
results must not be relabelled contradictions.

Reference constraints are assistant-authored, not independent expert gold
labels. The diagnostic addresses new-question and evidence-availability behavior
but does not establish full coverage of the requested hard benchmark. The gap
is not an unexecuted script or a requirement to prove that a foundation model
never saw a source.

## Bounded next step if conflict coverage is pursued

First verify comparable primary-study pairs on the same specific claim:
population, exposure/intervention, outcome, analysis, and qualifiers. Only then
freeze a small conflict subset and source-based constraints before generation.
Run existing conditions unchanged, retain every attempt, and report the subset
separately. Do not alter the completed 32-question bank, retune evaluators on
new answers, or claim expert ground truth without it. Stop source acquisition
rather than manufacture conflict labels if no valid pairs can be established
within the agreed scope.
