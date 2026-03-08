#!/usr/bin/env python3
"""
Collect and compare all ablation/baseline results.

Reads final_results.json (or test_detailed_results.json) from each
experiment's metrics directory, prints a comparison table, and saves
experiments/ablation_comparison.json.

Usage:
    cd ~/OncoVision-X
    source venv/bin/activate
    python scripts/collect_ablation_results.py
"""

import json
import sys
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Experiments to compare (in display order)
# ─────────────────────────────────────────────────────────────
EXPERIMENTS = [
    ('Full Model (DCA-Net)',           'full_model'),
    ('Ablation: No Context Stream',    'ablation_no_context'),
    ('Ablation: No Attention',         'ablation_no_attention'),
    ('Ablation: No Curriculum',        'ablation_no_curriculum'),
    ('Ablation: No Uncertainty',       'ablation_no_uncertainty'),
    ('Baseline: 3D ResNet-18',         'baseline_resnet3d18'),
    ('Baseline: 2D ResNet-18',         'baseline_resnet2d18'),
]

# Metric key lookup — different evaluators may store under different keys
AUC_KEYS   = ['auc_roc', 'test_auc', 'val_auc', 'roc_auc']
SENS_KEYS  = ['sensitivity', 'recall', 'tpr']
SPEC_KEYS  = ['specificity', 'tnr']


def _get(data: dict, keys: list, default=None):
    """Return first matching key value from a dict."""
    for k in keys:
        if k in data:
            return data[k]
    return default


def _find_results_file(exp_dir: str) -> Path | None:
    """
    Search for results in priority order:
      1. experiments/<exp>/evaluation/final_results.json
      2. experiments/<exp>/metrics/final_results.json
      3. experiments/<exp>/metrics/test_detailed_results.json
      4. experiments/<exp>/metrics/test_results.json
    """
    base = Path('experiments') / exp_dir

    candidates = [
        base / 'evaluation' / 'final_results.json',
        base / 'metrics'    / 'final_results.json',
        base / 'metrics'    / 'test_detailed_results.json',
        base / 'metrics'    / 'test_results.json',
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def load_experiment(name: str, exp_dir: str) -> dict:
    """Load metrics for one experiment."""
    result = {
        'name':        name,
        'experiment':  exp_dir,
        'auc_roc':     None,
        'sensitivity': None,
        'specificity': None,
        'found':       False,
    }

    results_file = _find_results_file(exp_dir)

    if results_file is None:
        return result

    try:
        with open(results_file) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"  ⚠  Could not parse {results_file}: {exc}")
        return result

    # Flatten nested dicts one level (e.g. test_results → top-level)
    if 'test_results' in data and isinstance(data['test_results'], dict):
        data.update(data['test_results'])

    result['auc_roc']     = _get(data, AUC_KEYS)
    result['sensitivity'] = _get(data, SENS_KEYS)
    result['specificity'] = _get(data, SPEC_KEYS)
    result['found']       = True
    result['source_file'] = str(results_file)

    return result


def main():
    print('=' * 70)
    print('ABLATION & BASELINE RESULTS COMPARISON')
    print('=' * 70)

    results = []
    full_model_auc = None

    for (name, exp_dir) in EXPERIMENTS:
        r = load_experiment(name, exp_dir)
        results.append(r)

        print(f'\n{name}:')

        if not r['found']:
            print(f"  ✗  Results not found (experiment not yet evaluated)")
            continue

        auc  = r['auc_roc']
        sens = r['sensitivity']
        spec = r['specificity']
        src  = r.get('source_file', '')

        if auc is not None:
            print(f"  AUC-ROC:     {auc:.4f}")
        else:
            print("  AUC-ROC:     N/A")

        if sens is not None:
            print(f"  Sensitivity: {sens:.4f}")
        else:
            print("  Sensitivity: N/A")

        if spec is not None:
            print(f"  Specificity: {spec:.4f}")
        else:
            print("  Specificity: N/A")

        print(f"  Source:      {src}")

        if exp_dir == 'full_model' and auc is not None:
            full_model_auc = auc

    # ── Component contribution table ──────────────────────────
    if full_model_auc is not None:
        print('\n' + '=' * 70)
        print('COMPONENT CONTRIBUTIONS (Δ from Full Model)')
        print('=' * 70)
        print(f'\nFull Model AUC: {full_model_auc:.4f} (reference)')

        for r in results[1:]:  # skip full model row
            auc = r['auc_roc']
            if auc is None:
                continue

            delta     = auc - full_model_auc
            delta_pct = (delta / full_model_auc) * 100
            direction = '✓ (lower, expected)' if delta < 0 else '✗ WARNING (higher than full model!)'

            print(f"\n{r['name']}:")
            print(f"  AUC: {auc:.4f}  ({delta:+.4f},  {delta_pct:+.2f}%)  {direction}")

    else:
        print('\n⚠  Full model results not found — skipping contribution table.')
        print('   Run full_model evaluation first:')
        print('   python scripts/evaluate_experiment.py --experiment full_model --checkpoint best --split test')

    # ── Formatted table ───────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY TABLE')
    print('=' * 70)
    header = f"{'Model':<36} {'AUC-ROC':>8}  {'Sensitivity':>11}  {'Specificity':>11}"
    print(header)
    print('-' * 70)
    for r in results:
        auc  = f"{r['auc_roc']:.4f}"   if r['auc_roc']  is not None else 'N/A'
        sens = f"{r['sensitivity']:.4f}" if r['sensitivity'] is not None else 'N/A'
        spec = f"{r['specificity']:.4f}" if r['specificity'] is not None else 'N/A'
        tag  = '' if r['found'] else ' (missing)'
        print(f"{r['name'][:35]:<36} {auc:>8}  {sens:>11}  {spec:>11}{tag}")

    # ── Save comparison JSON ──────────────────────────────────
    comparison_file = Path('experiments') / 'ablation_comparison.json'
    comparison_file.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        'full_model_auc': full_model_auc,
        'results': [
            {k: v for k, v in r.items() if k != 'source_file'}
            for r in results
        ],
    }

    with open(comparison_file, 'w') as f:
        json.dump(payload, f, indent=2)

    print(f'\n{"=" * 70}')
    print(f'Results saved to: {comparison_file}')
    print('=' * 70)


if __name__ == '__main__':
    # Always run from project root
    root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(root)
    main()
