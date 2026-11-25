# Fresh inference and training

Offline replay needs no model credentials or GPU. Fresh inference is a separate
operation. Do not use default CLI values to claim a historical experiment was
reproduced.

## Code and environment

The main entry point is code_inspection/autism_gpt_mvp.py. The missing
gene_registry.py and task3_topic_rules.py modules have been restored from the
frozen runtime, with hashes matching the diagnostic freeze.

    python code_inspection/autism_gpt_mvp.py --help
    python code_inspection/autism_gpt_mvp.py run-query --help
    python code_inspection/evaluate_paper_style.py --help

These commands need the modules imported by the respective scripts.
environment_versions.json records observed versions; it is not a complete
cross-platform installation lock. Runtime imports include NumPy, pandas,
requests, torch, transformers, peft, sentence-transformers, faiss, datasets, and
evaluate. Install platform-appropriate torch before a GPU run. API evaluators,
if explicitly enabled, require credentials outside this repository.

## Assets for a fresh local query

Provide the trained router directory, task-specific adapters with tokenizer
files, and a complete index directory. Each index includes index_config.json,
three task*.faiss files, and three task*_meta.jsonl files. The recorded embedding
model is BAAI/bge-base-en-v1.5; do not silently rebuild with the generic MiniLM
default. Adapters use Qwen/Qwen2.5-3B-Instruct. See asset_inventory.json for
digests and availability; large weights and full indexes are not in this candidate.

The index's embed_model and adapter's base_model_name_or_path must resolve
locally or through the model provider. Relocate these references only in a
separate copy, document the path-only change, and retain original digests.
Pin recorded model revisions rather than an unpinned latest checkpoint.

The CLI accepts --query, --router-model-dir, --index-dir, and --task1-expert-dir,
--task2-expert-dir, --task3-expert-dir. The matched protocol recorded Top-5,
router/context thresholds 0.65/0.25, greedy generation, 220 output tokens, and
task-specific input limits of 2048/1536 for Tasks 2/3. These are protocol values,
not assertions that one generic CLI call recreates both experimental arms.
See runtime_flags.csv for the five enabled switches. The retrieval API judge
was disabled.

The new-question study additionally used frozen routing/bridge and source
withholding. Its orchestration refers to auxiliary study assets. These runners
are supplied as inspectable source, not as a completed portable fresh-run command.
Saved-answer replay does not rely on them.

## Historical training

Summaries specify the base model and LoRA settings, but do not establish exact
training row IDs, split seed, or sequence limit. A trainer implementation does
not recover those missing records. The current local trainer is supplied under
code_inspection/scripts/ for inspection with source provenance; it is not
labelled the exact pre-training snapshot. New training must record its own data
IDs and configuration.
