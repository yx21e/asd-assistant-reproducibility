"""Render three recovered historical figures; fonts/PDF metadata can vary by host."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise RuntimeError('Choose an empty output directory; retained figures are not overwritten.')
    args.out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    os.environ['MPLBACKEND'] = 'Agg'
    path = root / 'code_inspection/historical_plot_functions.py'
    spec = importlib.util.spec_from_file_location('historical_plots', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    m.FIG_DIR = args.out
    m.TASK3_LARGE = root / 'evidence/historical/task3_large.jsonl'
    payload = json.loads((root / 'evidence/historical/plot_payload.json').read_text(encoding='utf-8'))
    m.build_threshold_sweep_figure(payload['thresholds'])
    m.build_memory_design_figure(payload['development'], payload['optional_stack'])
    m.setup_style()
    m.task3_heatmap()
    print(json.dumps({'rendered':[p.name for p in sorted(args.out.glob('*.pdf'))],
        'scope':'Three recovered historical figures only. Original combined scorecard renderer remains unlocated.'},indent=2))


if __name__ == '__main__':
    main()
