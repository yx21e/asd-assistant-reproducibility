"""Operational import-path amendment; leave frozen runner and evaluators intact."""
import sys
from pathlib import Path
import run_challenge as study

ROOT = Path(__file__).resolve().parent
spec = study.read(ROOT.parent / 'prospective_protocol_v001/SYSTEM_SPEC.json')
dependency = spec['code']['evaluate_task3_realistic.py']
assert study.sha(dependency['path']) == dependency['sha256']
sys.path.append(str(Path(dependency['path']).parent))
amendment = {'reason': 'Initial launch failed before any outputs: missing evaluate_task3_realistic import path.',
             'change': 'Add the already frozen dependency directory to sys.path; no source edits.',
             'failed_launch_produced_answers': 0, 'questions_changed': False,
             'model_settings_changed': False, 'metric_changed': False,
             'files': {str(Path(__file__)): study.sha(__file__), dependency['path']: dependency['sha256']},
             'runner_sha256': study.sha(ROOT / 'run_challenge.py')}
path = ROOT / 'OPERATIONAL_AMENDMENT_001.json'
if path.exists():
    assert study.read(path) == amendment
else:
    study.write(path, amendment, exclusive=True)
study.run()
module = sys.modules['evaluate_task3_realistic']
assert study.sha(module.__file__) == dependency['sha256']
