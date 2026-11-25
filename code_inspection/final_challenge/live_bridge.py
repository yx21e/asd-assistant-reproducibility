"""Native retrieval bridge. Development smoke outputs are never held-out results."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path


def normalized(text):
    return re.sub(r'\s+', ' ', str(text)).strip().lower()


def identities(row):
    meta = row.get('meta') or {}
    result = set()
    for owner in (row, meta):
        pmid = str(owner.get('pmid') or '').strip()
        if pmid:
            result.add('pmid:' + pmid)
        for key in ('doi', 'source_group'):
            value = normalized(owner.get(key) or '')
            if value:
                value = re.sub(r'^https?://(?:dx\.)?doi\.org/', '', value)
                result.add('group:' + value)
        for value in owner.get('source_pmids') or []:
            result.add('pmid:' + str(value))
    for pmid in re.findall(r'\bPMID\s*[:=]\s*(\d+)', str(row.get('text') or ''), re.I):
        result.add('pmid:' + pmid)
    return result


def mask_view(original, pmids, groups):
    """Filter native indexes and all native secondary-lookup stores in memory."""
    import numpy as np
    excluded = {'pmid:' + str(p) for p in pmids}
    excluded.update('group:' + normalized(g) for g in groups)
    all_rows = [r for rows in original.metas.values() for r in rows]
    named = [r for r in all_rows if identities(r) & excluded]
    titles = {normalized((r.get('meta') or {}).get('title') or '') for r in named}
    titles = {t for t in titles if len(t) >= 20}
    texts = {normalized(r['text']) for r in named if len(normalized(r.get('text', ''))) >= 40}

    def withheld(row):
        text = normalized(row.get('text') or '')
        title = normalized((row.get('meta') or {}).get('title') or '')
        return bool(identities(row) & excluded or title in titles or text in texts
                    or any(t in text for t in titles))

    view = copy.copy(original)
    view.indices, view.metas = {}, {}
    removed = []
    for task, rows in original.metas.items():
        keep = [i for i, row in enumerate(rows) if not withheld(row)]
        removed.extend({'task': task, 'doc_id': r['doc_id'], 'identities': sorted(identities(r))}
                       for r in rows if withheld(r))
        view.metas[task] = [copy.deepcopy(rows[i]) for i in keep]
        index = original.faiss.IndexFlatIP(original.indices[task].d)
        if keep:
            index.add(np.asarray([original.indices[task].reconstruct(i) for i in keep], dtype='float32'))
        view.indices[task] = index
    view.task3_doc_lookup = {k: copy.deepcopy(r) for k, r in original.task3_doc_lookup.items() if not withheld(r)}
    view.task3_topic_rows = {k: [copy.deepcopy(r) for r in rows if not withheld(r)]
                             for k, rows in original.task3_topic_rows.items()}
    if not removed:
        raise ValueError('Withheld manipulation removed no indexed records')
    return view, removed, withheld


class NativeEvidenceAccess:
    def __init__(self, native, runtime, arm, forbidden=None):
        self.native, self.runtime, self.arm = native, runtime, arm
        self.forbidden = forbidden or (lambda row: False)
        self.issued = {}
        self.calls = []

    def __getattr__(self, name):
        return getattr(self.native, name)

    def register(self, rows, method, task, query=None):
        for row in rows:
            if self.forbidden(row):
                raise ValueError('Withheld source reintroduced by ' + method)
            key = (str(row.get('task') or task), str(row['doc_id']), str(row['text']))
            self.issued[key] = copy.deepcopy(row)
        self.calls.append({'method': method, 'task': task, 'query': query,
                           'returned_ids': [r['doc_id'] for r in rows]})
        return rows

    def search(self, task, query, topk=5):
        if self.arm == 'task_conditioned':
            rows = self.native.search(task, query, topk=topk)
        else:
            vector = self.runtime._encode_texts(self.native.embed_model, [query], batch_size=1)
            rows = []
            for pool, index in self.native.indices.items():
                if not index.ntotal:
                    continue
                scores, positions = index.search(vector, min(topk, index.ntotal))
                for score, position in zip(scores[0], positions[0]):
                    if int(position) < 0:
                        continue
                    source = self.native.metas[pool][int(position)]
                    rows.append({'score': float(score), 'doc_id': source['doc_id'],
                                 'task': pool, 'text': source['text'],
                                 'meta': copy.deepcopy(source.get('meta') or {})})
            rows.sort(key=lambda r: (-r['score'], r['task'], str(r['doc_id'])))
            rows = rows[:topk]
        return self.register(rows, 'search', task, query)

    def find_gene_docs(self, task, gene_name, doc_type=None, limit=None):
        rows = self.native.find_gene_docs(task, gene_name, doc_type=doc_type, limit=limit)
        return self.register(rows, 'find_gene_docs', task, gene_name)

    def find_gene_doc(self, task, gene_name):
        rows = self.find_gene_docs(task, gene_name, limit=1)
        return rows[0] if rows else None

    def trace_contexts(self, contexts):
        trace = []
        for context in contexts:
            candidates = []
            for row in self.issued.values():
                text = str(row['text'])
                if not text or not context.endswith(text):
                    continue
                prefix = context[:-len(text)]
                if prefix and not re.fullmatch(r'(?:\{[^{}\r\n]+\})+\s*', prefix):
                    continue
                if self.forbidden(row):
                    raise ValueError('Withheld selected context')
                candidates.append({'task': row['task'], 'doc_id': row['doc_id'],
                                   'source_identities': sorted(identities(row)),
                                   'metadata_prefix': prefix,
                                   'source_text_sha256': hashlib.sha256(text.encode()).hexdigest()})
            if not candidates:
                raise ValueError('Selected context cannot be traced to issued source text')
            trace.append({'context_sha256': hashlib.sha256(context.encode()).hexdigest(),
                          'possible_source_records': candidates})
        return trace


class LiveBridge:
    """Reuse route/assembly contracts while preserving native ranking and context prefixes."""
    def __init__(self, spec, runtime, formatter, assembly, route_runner, generator, reset_seed):
        self.spec, self.rt, self.formatter, self.assembly = spec, runtime, formatter, assembly
        self.route_runner, self.generator, self.reset_seed = route_runner, generator, reset_seed
        self.p = spec['historical_parameters_to_retain']
        if runtime._retrieval_judge_enabled('task2') or runtime._retrieval_judge_enabled('task3'):
            raise ValueError('Pinned local configuration has retrieval judge disabled; no API allowed')

    def run_one(self, request, budget_task, route, native, arm, condition, forbidden=None, generate=False):
        task = route.selected_task
        access = NativeEvidenceAccess(native, self.rt, arm, forbidden)
        result = {'record_role': 'development_smoke_not_heldout_benchmark',
                  'example_id': request.example_id, 'query': request.query,
                  'configuration': arm, 'condition': condition,
                  'candidate_id': self.spec['candidate_id'], 'selected_task': task,
                  'initial_router': route.initial,
                  'final_router': {'task': task, 'confidence': route.confidence, 'scores': route.final_scores},
                  'budget_task': budget_task,
                  'max_input_tokens': self.p['input_budget_by_intended_task'][budget_task],
                  'common_settings': {k: self.p[k] for k in ('topk', 'router_threshold', 'context_threshold',
                                                           'max_new_tokens', 'temperature', 'top_p', 'seed')},
                  'retrieval_judge': 'disabled_in_pinned_configuration',
                  'retrieval_calls': access.calls, 'generation_contexts': [],
                  'top_hits': [], 'raw_generation': None, 'final_answer': None}

        def abstain(reason):
            result.update(status='abstain', reason=reason)
            return result

        if task == 'reject':
            return abstain('router_reject')

        def retrieve(query):
            rows = access.search(task, query, topk=self.p['topk'])
            if arm == 'task_conditioned':
                rows = self.rt._promote_explicit_gene_hit(task, query, rows, access)
            return rows

        query_used = route.query_used
        hits = retrieve(query_used)
        if not self.rt._context_ok(hits, self.p['context_threshold']):
            retry = self.rt._norm_space(re.sub(r'[^\w\s-]', ' ', query_used))
            retried = retrieve(retry)
            if self.rt._context_ok(retried, self.p['context_threshold']):
                hits, query_used = retried, retry
        result.update(top_hits=hits, query_used_for_retrieval=query_used)
        if not self.rt._context_ok(hits, self.p['context_threshold']):
            return abstain('insufficient_context')
        if not self.generator.has_expert(task):
            return abstain('missing_selected_expert')
        contexts = (self.rt._select_generation_contexts(task, request.query, hits, access)
                    if arm == 'task_conditioned' else [r['text'] for r in hits[:self.p['topk']]])
        if not contexts:
            return abstain('missing_generation_context')
        result.update(generation_contexts=contexts, context_provenance=access.trace_contexts(contexts))
        if not generate:
            result.update(status='retrieval_checked_not_generated', reason=None)
            return result
        self.reset_seed(self.p['seed'])
        raw = self.generator.generate(task=task, user_query=request.query, contexts=contexts,
            max_input_tokens=result['max_input_tokens'], max_new_tokens=self.p['max_new_tokens'],
            temperature=self.p['temperature'], top_p=self.p['top_p'])
        result['raw_generation'] = copy.deepcopy(raw)
        record = {'query': request.query, 'task': task, 'selected_task': task,
                  'generation_contexts': list(contexts), 'raw_generation': copy.deepcopy(raw),
                  'has_answer': bool(raw['answer'])}
        rendered, branch = self.assembly.render_complete(self.rt, self.formatter, record, 'typed_contexts')
        clean = self.assembly.omit_empty_fields(rendered)
        fixed = self.assembly.repair(answer=clean, query=request.query, contexts=contexts,
            selected_task=task, fallback_used=branch, context_kind=self.assembly.context_kind)
        needs_review = fixed.status == 'review_required' or not fixed.answer.strip()
        result.update(status='review_required' if needs_review else 'ok', reason=None,
                      final_answer=fixed.answer, fallback_used=branch, assembly_applied_once=True,
                      repair_status=fixed.status, repair_reasons=list(fixed.reasons),
                      repair_edits=list(fixed.edits), review_required=needs_review)
        return result
