#!/usr/bin/env python3
"""
Validate all ablation/baseline config files before training.

Checks:
  - File existence
  - Data paths (preprocessed_dir)
  - Core hyperparameters (batch_size, focal params, optimizer)
  - Ablation-specific settings (curriculum.enabled, uncertainty weight)
  - Single-GPU settings

Usage:
    cd ~/OncoVision-X
    source venv/bin/activate
    python scripts/validate_configs.py
"""

import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML not installed.  pip install pyyaml")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────
# Configs to validate
# ─────────────────────────────────────────────────────────────
CONFIGS = [
    'configs/ablation_no_context.yaml',
    'configs/ablation_no_attention.yaml',
    'configs/ablation_no_curriculum.yaml',
    'configs/ablation_no_uncertainty.yaml',
    'configs/baseline_resnet3d18.yaml',
    'configs/baseline_resnet2d18.yaml',
]

# Expected values (shared across all experiments)
EXPECTED = {
    'preprocessed_dir': 'preprocessed_data',
    'batch_size':        64,
    'learning_rate':     0.0003,
    'num_epochs':        150,
    'focal_alpha':       0.995,
    'focal_gamma':       2.5,
    'optimizer':         'AdamW',
    'weight_decay':      0.00001,
}


def check(label: str, actual, expected, issues: list, config_path: str) -> bool:
    """Check a single value; log mismatch to issues list. Return True if OK."""
    if actual == expected:
        print(f"    ✓  {label}: {actual}")
        return True
    else:
        msg = f"✗  {config_path}: {label} is '{actual}' (expected '{expected}')"
        issues.append(msg)
        print(f"    ✗  {label}: {actual!r}  ← expected {expected!r}")
        return False


def validate_config(config_path: str, issues: list) -> bool:
    """Validate a single config file. Returns True if valid."""
    path = Path(config_path)

    print(f"\n{'─' * 60}")
    print(f"  {config_path}")
    print(f"{'─' * 60}")

    # ── File exists ──────────────────────────────────────────
    if not path.exists():
        msg = f"✗  {config_path}: FILE NOT FOUND"
        issues.append(msg)
        print(f"  {msg}")
        return False

    with open(path) as f:
        config = yaml.safe_load(f)

    if config is None:
        msg = f"✗  {config_path}: empty YAML"
        issues.append(msg)
        print(f"  {msg}")
        return False

    ok = True
    data_cfg     = config.get('data', {}) or {}
    training_cfg = config.get('training', {}) or {}
    gpu_cfg      = config.get('gpu', {}) or {}

    # ── Data path ────────────────────────────────────────────
    preproc = data_cfg.get('preprocessed_dir', '')
    ok &= check('data.preprocessed_dir', preproc,
                EXPECTED['preprocessed_dir'], issues, config_path)

    # metadata_dir is optional but recommended
    meta = data_cfg.get('metadata_dir', None)
    if meta:
        print(f"    ✓  data.metadata_dir: {meta}")
    else:
        print(f"    ⚠  data.metadata_dir: not set (optional but recommended)")

    # ── Core hyperparameters ─────────────────────────────────
    ok &= check('training.batch_size',   training_cfg.get('batch_size'),
                EXPECTED['batch_size'], issues, config_path)
    ok &= check('training.learning_rate', training_cfg.get('learning_rate'),
                EXPECTED['learning_rate'], issues, config_path)
    ok &= check('training.num_epochs',   training_cfg.get('num_epochs'),
                EXPECTED['num_epochs'], issues, config_path)
    ok &= check('training.optimizer',    training_cfg.get('optimizer'),
                EXPECTED['optimizer'], issues, config_path)
    ok &= check('training.weight_decay', training_cfg.get('weight_decay'),
                EXPECTED['weight_decay'], issues, config_path)
    ok &= check('training.focal_alpha',  training_cfg.get('focal_alpha'),
                EXPECTED['focal_alpha'], issues, config_path)
    ok &= check('training.focal_gamma',  training_cfg.get('focal_gamma'),
                EXPECTED['focal_gamma'], issues, config_path)

    # ── AMP ──────────────────────────────────────────────────
    use_amp = training_cfg.get('use_amp', False)
    if use_amp:
        print(f"    ✓  training.use_amp: {use_amp}")
    else:
        issues.append(f"✗  {config_path}: use_amp not enabled (recommended)")
        print(f"    ⚠  training.use_amp: {use_amp} (recommended: true)")

    # ── Single-GPU ───────────────────────────────────────────
    multi_gpu_training = training_cfg.get('use_data_parallel', True)
    multi_gpu_section  = gpu_cfg.get('use_multi_gpu', True)

    if not multi_gpu_training:
        print(f"    ✓  training.use_data_parallel: False (single GPU)")
    else:
        issues.append(f"✗  {config_path}: training.use_data_parallel should be False for ablations")
        print(f"    ✗  training.use_data_parallel: {multi_gpu_training} (should be False)")
        ok = False

    if not multi_gpu_section:
        print(f"    ✓  gpu.use_multi_gpu: False (single GPU)")
    else:
        issues.append(f"✗  {config_path}: gpu.use_multi_gpu should be False for ablations")
        print(f"    ⚠  gpu.use_multi_gpu: {multi_gpu_section}")

    # ── Ablation-specific checks ─────────────────────────────
    name = config_path.lower()

    # no_curriculum: curriculum must be disabled
    if 'no_curriculum' in name:
        curriculum_cfg  = training_cfg.get('curriculum', {}) or {}
        enabled         = curriculum_cfg.get('enabled', True)  # default True = not disabled
        has_stage_keys  = 'stage1_epochs' in curriculum_cfg

        if not enabled and not has_stage_keys:
            print(f"    ✓  curriculum.enabled: False (ablation correct)")
        elif not enabled and has_stage_keys:
            print(f"    ⚠  curriculum.enabled: False but stage epoch keys present (harmless)")
        else:
            issues.append(
                f"✗  {config_path}: curriculum should be disabled "
                f"(set training.curriculum.enabled: false)"
            )
            print(f"    ✗  training.curriculum.enabled: {enabled} (should be false)")
            ok = False

        # model.ablation should NOT be 'no_curriculum' (that's not a model variant)
        model_ablation = (config.get('model', {}) or {}).get('ablation', None)
        if model_ablation == 'no_curriculum':
            issues.append(
                f"✗  {config_path}: model.ablation should be null for no_curriculum "
                f"(curriculum is training-side only)"
            )
            print(f"    ✗  model.ablation: '{model_ablation}' — should be null")
            ok = False
        else:
            print(f"    ✓  model.ablation: {model_ablation} (correct)")

    # no_uncertainty: uncertainty loss weight must be 0.0
    if 'no_uncertainty' in name:
        loss_weights  = training_cfg.get('loss_weights', {}) or {}
        unc_weight    = loss_weights.get('uncertainty', 0.2)
        if unc_weight == 0.0:
            print(f"    ✓  loss_weights.uncertainty: 0.0 (ablation correct)")
        else:
            issues.append(
                f"✗  {config_path}: loss_weights.uncertainty should be 0.0"
            )
            print(f"    ✗  loss_weights.uncertainty: {unc_weight} (should be 0.0)")
            ok = False

    # no_context: model.ablation must be 'no_context'
    if 'no_context' in name:
        model_ablation = (config.get('model', {}) or {}).get('ablation', None)
        if model_ablation == 'no_context':
            print(f"    ✓  model.ablation: 'no_context'")
        else:
            issues.append(f"✗  {config_path}: model.ablation should be 'no_context'")
            print(f"    ✗  model.ablation: '{model_ablation}' (should be 'no_context')")
            ok = False

    # no_attention: model.ablation must be 'no_attention'
    if 'no_attention' in name:
        model_ablation = (config.get('model', {}) or {}).get('ablation', None)
        if model_ablation == 'no_attention':
            print(f"    ✓  model.ablation: 'no_attention'")
        else:
            issues.append(f"✗  {config_path}: model.ablation should be 'no_attention'")
            print(f"    ✗  model.ablation: '{model_ablation}' (should be 'no_attention')")
            ok = False

    # baselines: model type must be recognised
    if 'baseline' in name:
        model_type = (config.get('model', {}) or {}).get('type', '')
        if model_type in ('resnet3d18', 'resnet2d18'):
            print(f"    ✓  model.type: '{model_type}'")
        else:
            issues.append(f"✗  {config_path}: model.type '{model_type}' unrecognised")
            print(f"    ✗  model.type: '{model_type}' (expected resnet3d18 or resnet2d18)")
            ok = False

    status = '✓ VALID' if ok else '✗ HAS ISSUES'
    print(f"\n  → {status}")
    return ok


def main():
    print('=' * 70)
    print('CONFIG VALIDATION')
    print('=' * 70)

    issues: list[str] = []
    all_valid = True

    for config_path in CONFIGS:
        valid = validate_config(config_path, issues)
        if not valid:
            all_valid = False

    print('\n' + '=' * 70)

    if issues:
        print(f'ISSUES FOUND ({len(issues)}):')
        for issue in issues:
            print(f'  {issue}')
        print('\nFix these issues before training!')
        sys.exit(1)
    else:
        print('✓ ALL CONFIGS VALID — ready to train')

    print('=' * 70)


if __name__ == '__main__':
    # Always run from project root
    root = Path(__file__).resolve().parent.parent
    import os
    os.chdir(root)
    main()
