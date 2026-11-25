"""One frozen, local evidence-boundary study; no test-set optimization."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
ALIGN = ROOT.parent
BASE = ALIGN.parent
RUNS = BASE.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(mapping):
    for path, digest in mapping.items():
        if sha(path) != digest:
            raise RuntimeError('Frozen input changed: ' + path)


def freeze():
    if (ROOT / 'STUDY_FREEZE.json').exists():
        raise RuntimeError('Study is already frozen')
    runtime = read(ROOT / 'RUNTIME_FREEZE.json')
    verify(runtime['files'])
    candidates = read(ROOT / 'CANDIDATE_QUESTIONS.json')
    audit = read(ROOT / 'OVERLAP_AUDIT.json')
    review = read(ROOT / 'OVERLAP_REVIEW.json')
    if review['audit_sha256'] != sha(ROOT / 'OVERLAP_AUDIT.json'):
        raise RuntimeError('Overlap review refers to a different audit')
    if audit['candidate_sha256'] != sha(ROOT / 'CANDIDATE_QUESTIONS.json'):
        raise RuntimeError('Candidate bank changed after overlap audit')
    decisions = {r['query_id']: r for r in review['decisions']}
    if set(decisions) != {r['query_id'] for r in candidates}:
        raise RuntimeError('Every candidate needs an overlap decision')
    source_by_id = {r['source_key']: r for r in read(ROOT / 'SOURCE_RECORDS.json')}
    bank = []
    for row in candidates:
        if decisions[row['query_id']]['decision'] != 'eligible':
            continue
        source = source_by_id[row['source_key']]
        bank.append(dict(row, source_pmids=[source['pmid']], source_group=source['source_group'],
                         title=source['title'], human_gold=False,
                         evidence_spans=[s for s in source['sentences'] if s['id'] in row['anchors']]))
    if not bank or len(bank) != len({r['source_group'] for r in bank}):
        raise RuntimeError('Empty bank or repeated source clusters')
    write(ROOT / 'FROZEN_BANK.json', bank, exclusive=True)
    files = dict(runtime['files'])
    for name in ('RUNTIME_FREEZE.json', 'CANDIDATE_QUESTIONS.json', 'audit_overlap.py',
                 'OVERLAP_AUDIT.json', 'OVERLAP_REVIEW.json', 'FROZEN_BANK.json',
                 'run_challenge.py', 'summarize_challenge.py'):
        files[str(ROOT / name)] = sha(ROOT / name)
    write(ROOT / 'STUDY_FREEZE.json', {'frozen_utc': now(), 'files': files,
          'task_counts': dict(Counter(r['task'] for r in bank)), 'expected_records': len(bank) * 4,
          'questions_only_enter_inference': True, 'reference_constraints_not_model_input': True,
          'study_role': 'new_question_retained_source_boundary_diagnostic_not_expert_gold',
          'score_task': 'intended_task_including_misroutes', 'paired_bootstrap_resamples': 2000,
          'paired_bootstrap_seed': 20261120, 'no_new_evaluator': True}, exclusive=True)
    print('Frozen study:', len(bank), 'questions;', len(bank) * 4, 'planned records', flush=True)


def run():
    freeze_info = read(ROOT / 'STUDY_FREEZE.json')
    verify(freeze_info['files'])
    os.environ.update(HF_HOME=str(RUNS / 'cache/hf'), HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false',
        PYTHONDONTWRITEBYTECODE='1', TEMP=str(RUNS / 'tmp'), TMP=str(RUNS / 'tmp'),
        XDG_CACHE_HOME=str(RUNS / 'cache'), TORCH_HOME=str(RUNS / 'cache/torch'),
        TORCHINDUCTOR_CACHE_DIR=str(RUNS / 'cache/torchinductor'), TRITON_CACHE_DIR=str(RUNS / 'cache/triton'))
    network_attempts = []

    def deny_network(event, args):
        if event in ('socket.connect', 'socket.sendto'):
            network_attempts.append(event)
            raise RuntimeError('Network prohibited in this local diagnostic')

    sys.addaudithook(deny_network)
    contract = import_file('final_challenge_contract', ALIGN / 'prospective_runner_contract_v001/runner_contract.py')
    spec = contract.read_spec()
    assembly = contract.load_assembly(spec)
    sys.path[:0] = [str(Path(spec['code']['autism_gpt_mvp.py']['path']).parent),
                    str(Path(spec['code']['gene_registry.py']['path']).parent)]
    native = import_file('final_challenge_native', ALIGN / 'revision_closure_20261120/live_bridge.py')
    import numpy as np
    import torch
    import autism_gpt_mvp as rt
    import evaluate_autism_gpt as pipeline
    import evaluate_paper_style as ev
    for name, module in [('autism_gpt_mvp.py', rt), ('evaluate_autism_gpt.py', pipeline), ('evaluate_paper_style.py', ev)]:
        if sha(module.__file__) != spec['code'][name]['sha256']:
            raise RuntimeError('Wrong imported module: ' + name)
    # Load the unchanged scoring function without legacy bank-loader side effects.
    source = Path(spec['code']['isolated_support.py']['path'])
    tree = ast.parse(source.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'score_answer')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    score_answer = namespace['score_answer']
    torch.set_num_threads(6)

    def reset_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    class Generator(rt.TaskExpertGenerator):
        def _load(self, task):
            if task not in self._cache and self._cache:
                self._cache.clear()
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return super()._load(task)

        def generate(self, **kwargs):
            result = super().generate(**kwargs)
            tokenizer = self._cache[kwargs['task']]['tokenizer']
            prompt = result['prompt_text']
            encoded = tokenizer(prompt, truncation=True, max_length=kwargs['max_input_tokens'])['input_ids']
            result.update(actual_model_input=tokenizer.decode(encoded, skip_special_tokens=False),
                          actual_input_tokens=len(encoded),
                          untruncated_input_tokens=len(tokenizer(prompt, truncation=False)['input_ids']))
            return result

    bank = read(ROOT / 'FROZEN_BANK.json')
    outputs_file = ROOT / 'OUTPUTS.jsonl'
    previous = [json.loads(s) for s in outputs_file.read_text(encoding='utf-8').splitlines() if s.strip()] if outputs_file.exists() else []
    completed = {(r['example_id'], r['configuration'], r['condition']) for r in previous}
    if len(completed) != len(previous):
        raise RuntimeError('Duplicate result keys; no automatic overwrite')
    status = {'state': 'initializing', 'started_utc': now(), 'planned_records': len(bank) * 4,
              'completed_records': len(previous), 'paid_calls': 0, 'new_evaluator': False}
    write(ROOT / 'STATUS.json', status)
    started = time.monotonic()
    try:
        with contract.frozen_flags(spec['flags']):
            retriever = rt.TaskRetriever(ROOT / 'index')
            router = rt.RouterPredictor(RUNS / 'models/router_model_v1')
            generator = Generator({'task1': RUNS / 'experts/task1_expert_supported_v1',
                'task2': RUNS / 'experts/task2_expert_qwen25_3b_pubtator_v1',
                'task3': RUNS / 'experts/task3_expert_qwen25_3b_task2lit_v1'})
            deps = SimpleNamespace(router=router, refine_query=rt._refine_query,
                override_task1=rt._maybe_override_task1, override_task3=rt._maybe_override_task3)
            route_runner = contract.MatchedDryRunner(spec, deps, assembly)
            bridge = native.LiveBridge(spec, rt, pipeline._format_task1_structured_answer,
                                      assembly, route_runner, generator, reset_seed)
            for item in bank:
                request = contract.InferenceRequest(item['query_id'], item['query'])
                route = route_runner.route(request)
                masked, removed, forbidden = native.mask_view(retriever, item['source_pmids'], [item['source_group']])
                for condition in contract.CONDITIONS:
                    for arm in contract.ARMS:
                        key = (item['query_id'], arm, condition)
                        if key in completed:
                            continue
                        stamp = time.monotonic()
                        try:
                            record = bridge.run_one(request, item['task'], route,
                                retriever if condition == 'source_available' else masked, arm, condition,
                                forbidden=None if condition == 'source_available' else forbidden, generate=True)
                        except Exception as exc:
                            record = {'example_id': item['query_id'], 'query': item['query'],
                                'configuration': arm, 'condition': condition, 'selected_task': route.selected_task,
                                'final_router': {'task': route.selected_task, 'confidence': route.confidence, 'scores': route.final_scores},
                                'status': 'execution_error', 'error': str(exc), 'traceback': traceback.format_exc(),
                                'raw_generation': None, 'final_answer': None, 'generation_contexts': [], 'top_hits': []}
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        record.update(record_role=freeze_info['study_role'], intended_task=item['task'],
                            source_key=item['source_key'], source_pmids=item['source_pmids'],
                            source_group=item['source_group'], route_correct=route.selected_task == item['task'],
                            removed_records=removed if condition == 'source_withheld' else [],
                            finished_utc=now(), generation_seconds=round(time.monotonic() - stamp, 3))
                        try:
                            record['scores'] = score_answer(ev, item['task'], item['query'],
                                record.get('final_answer'), record.get('generation_contexts') or [], retriever.embed_model)
                        except Exception as exc:
                            record['scores'] = None
                            record['score_error'] = str(exc)
                            record['score_traceback'] = traceback.format_exc()
                        raw = record.get('raw_generation') or {}
                        selected = '\n'.join(record.get('generation_contexts') or [])
                        actual = raw.get('actual_model_input') or ''
                        norm = native.normalized
                        def contains_marker(text):
                            return norm(item['title']) in norm(text) or any('pmid:' + p in norm(text).replace(' ', '') for p in item['source_pmids'])
                        record['diagnostics'] = {
                            'source_in_hits': any('pmid:' + item['source_pmids'][0] in native.identities(h) for h in record.get('top_hits') or []),
                            'source_in_selected_contexts': bool(selected) and contains_marker(selected),
                            'source_marker_in_actual_input': bool(actual) and contains_marker(actual),
                            'full_anchor_quotes_in_context': sum(norm(s['text']) in norm(selected) for s in item['evidence_spans']),
                            'full_anchor_quotes_in_actual_input': sum(norm(s['text']) in norm(actual) for s in item['evidence_spans']),
                            'input_truncated': raw.get('untruncated_input_tokens', 0) > raw.get('actual_input_tokens', 0)}
                        append(outputs_file, record)
                        completed.add(key)
                        status.update(state='running', completed_records=len(completed), last_id=item['query_id'],
                            last_configuration=arm, last_condition=condition, updated_utc=now(),
                            elapsed_seconds=round(time.monotonic() - started, 1))
                        write(ROOT / 'STATUS.json', status)
                        print(f'{len(completed)}/{len(bank)*4} {item["query_id"]} {arm} {condition} {record["status"]}', flush=True)
            verify(freeze_info['files'])
            status.update(state='completed', finished_utc=now(), inputs_unchanged=True,
                          network_attempts=network_attempts, elapsed_seconds=round(time.monotonic() - started, 1))
            write(ROOT / 'STATUS.json', status)
            print(json.dumps(status), flush=True)
    except Exception as exc:
        status.update(state='failed', error=str(exc), traceback=traceback.format_exc(), updated_utc=now())
        write(ROOT / 'STATUS.json', status)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run'])
    args = parser.parse_args()
    freeze() if args.action == 'freeze' else run()
