"""Audit all candidate questions before any challenge outputs are generated."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def canonical(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def main():
    target = ROOT / 'OVERLAP_AUDIT.json'
    if target.exists():
        raise RuntimeError('Do not overwrite the pre-output overlap audit')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      TOKENIZERS_PARALLELISM='false')
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(6)
    candidates = read('CANDIDATE_QUESTIONS.json')
    history = read('HISTORICAL_QUERIES.json')
    sources = {r['source_key']: r for r in read('SOURCE_RECORDS.json')}
    selection = read('SOURCE_SELECTION.json')
    assert {r['source_key'] for r in candidates} == set(selection['task2_review_ids'] + selection['task3_candidate_ids'])
    assert len(candidates) == len({r['query_id'] for r in candidates}) == 32
    for row in candidates:
        source = sources[row['source_key']]
        assert row['task'] == source['task']
        available = {s['id'] for s in source['sentences']}
        assert set(row['anchors']).issubset(available), row['query_id']
    model = SentenceTransformer(read('RUNTIME_FREEZE.json')['embedding_snapshot'], device='cuda')
    h = model.encode([r['query'] for r in history], normalize_embeddings=True,
                     batch_size=64, convert_to_numpy=True, show_progress_bar=False)
    q = model.encode([r['query'] for r in candidates], normalize_embeddings=True,
                     batch_size=32, convert_to_numpy=True, show_progress_bar=False)
    similarities = q @ h.T
    rows = []
    for i, candidate in enumerate(candidates):
        order = np.argsort(-similarities[i])[:3]
        cq = canonical(candidate['query'])
        exact = [j for j, old in enumerate(history) if cq == canonical(old['query'])]
        neighbors = [{'query': history[j]['query'], 'cosine': float(similarities[i, j]),
                      'categories': history[j]['categories'], 'examples': history[j]['examples']}
                     for j in order]
        tokens = set(cq.split())
        jaccard = max(len(tokens & set(canonical(r['query']).split())) /
                      max(1, len(tokens | set(canonical(r['query']).split()))) for r in history)
        rows.append({'query_id': candidate['query_id'], 'query': candidate['query'],
                     'canonical_exact_count': len(exact), 'max_token_jaccard': jaccard,
                     'review_flag': bool(exact or similarities[i, order[0]] >= .90 or jaccard >= .65),
                     'nearest': neighbors})
    result = {'scope': 'Recorded historical questions only; not optimizer membership or foundation pretraining',
              'historical_unique_queries': len(history), 'candidate_count': len(candidates),
              'flags_fixed_before_embedding': {'cosine_at_least': .90, 'token_jaccard_at_least': .65},
              'review_policy': 'Inspect all top-three neighbors before freezing; no output-informed replacement',
              'candidate_sha256': hashlib.sha256((ROOT / 'CANDIDATE_QUESTIONS.json').read_bytes()).hexdigest(),
              'rows': rows}
    with target.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    for row in rows:
        print(json.dumps({'id': row['query_id'], 'flag': row['review_flag'],
                          'exact': row['canonical_exact_count'], 'query': row['query'],
                          'nearest': [{k: n[k] for k in ('query', 'cosine')} for n in row['nearest']]}), flush=True)


if __name__ == '__main__':
    main()
