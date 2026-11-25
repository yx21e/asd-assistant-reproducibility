"""Verify the public distribution and replay saved statistics without inference."""
import hashlib
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-complete', action='store_true',
                        help='Fail if the full-paper coverage audit has any unresolved blocker.')
    cli_args = parser.parse_args()
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files_sha256'].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise RuntimeError('Missing or unsafe manifest path: '+rel)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Changed artifact: '+rel)
    print('Manifest verified: '+str(len(manifest['files_sha256']))+' files',flush=True)
    commands = [
        ['scripts/replay_reported_tables.py'],
        ['scripts/replay_historical_statistics.py'],
        ['scripts/replay_diagnostic_counts.py'],
        ['scripts/replay_matched_bootstrap.py'],
        ['scripts/replay_new_question_diagnostic.py'],
        ['scripts/test_matched_bootstrap_replay.py','--package','.',
         '--original-analysis','code_inspection/postprocess_matched.py'],
        ['scripts/test_reported_table_replay.py']]
    for args in commands:
        subprocess.run([sys.executable,'-B']+args,cwd=ROOT,check=True,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    coverage = json.loads((ROOT / 'documentation/reproduction_coverage.json').read_text(encoding='utf-8'))
    print('All checks for the documented public subset passed. No model/API calls or raw-text rescoring.',flush=True)
    print('Full historical reproduction: '+str(coverage['full_historical_reproduction']),flush=True)
    for blocker in coverage['blockers']:
        print('OPEN: '+blocker,flush=True)
    if cli_args.require_complete and (not coverage['full_historical_reproduction'] or coverage['blockers']):
        raise SystemExit('Full-paper reproduction gate failed: documented blockers remain.')


if __name__ == '__main__':
    main()
