"""Verify the public distribution and replay saved statistics without inference."""
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'MANIFEST.json').read_text(encoding='utf-8'))
    for rel, expected in manifest['files_sha256'].items():
        path = (ROOT / rel).resolve()
        if not path.is_relative_to(ROOT.resolve()) or not path.is_file():
            raise RuntimeError('Missing or unsafe manifest path: '+rel)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Changed artifact: '+rel)
    print('Manifest verified: '+str(len(manifest['files_sha256']))+' files',flush=True)
    commands = [
        ['scripts/replay_matched_bootstrap.py'],
        ['scripts/replay_new_question_diagnostic.py'],
        ['scripts/test_matched_bootstrap_replay.py','--package','.',
         '--original-analysis','code_inspection/postprocess_matched.py']]
    for args in commands:
        subprocess.run([sys.executable,'-B']+args,cwd=ROOT,check=True,
                       env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
    print('All public numeric-replay checks passed. No model/API calls or rescoring.',flush=True)


if __name__ == '__main__':
    main()
