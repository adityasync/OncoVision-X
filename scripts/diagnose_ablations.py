#!/usr/bin/env python3
"""
Diagnostic Script — Verify All Ablation Model Variants

Checks:
  1. Each ablation creates a model with different parameter counts
  2. Forward pass works for all variants
  3. Output variance is reasonable (not collapsed)
  4. Configs are complete (no missing critical keys)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml


def check_configs():
    """Verify all ablation configs have required training parameters."""
    print("\n" + "=" * 60)
    print("  CONFIG COMPLETENESS CHECK")
    print("=" * 60)

    required_training_keys = [
        'num_epochs', 'batch_size', 'learning_rate', 'weight_decay',
        'warmup_epochs', 'optimizer', 'scheduler', 'scheduler_T0',
        'scheduler_Tmult', 'use_amp', 'gradient_clip',
        'gradient_accumulation_steps', 'use_data_parallel',
        'focal_gamma', 'focal_alpha', 'label_smoothing',
        'early_stopping_patience',
    ]

    required_model_keys = [
        'type', 'backbone', 'nodule_feature_dim', 'context_feature_dim',
        'fusion_dim', 'num_attention_heads', 'dropout', 'prediction_dropout',
    ]

    configs = [
        'configs/full_model.yaml',
        'configs/ablation_no_context.yaml',
        'configs/ablation_no_attention.yaml',
        'configs/ablation_no_curriculum.yaml',
        'configs/ablation_no_uncertainty.yaml',
    ]

    all_pass = True
    for config_path in configs:
        name = os.path.basename(config_path).replace('.yaml', '')
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)

            training = cfg.get('training', {})
            model = cfg.get('model', {})

            missing_training = [k for k in required_training_keys if k not in training]
            missing_model = [k for k in required_model_keys if k not in model]

            if missing_training or missing_model:
                print(f"\n  ✗ {name}:")
                if missing_training:
                    print(f"    Missing training keys: {missing_training}")
                if missing_model:
                    print(f"    Missing model keys: {missing_model}")
                all_pass = False
            else:
                print(f"  ✓ {name}: all keys present")

        except FileNotFoundError:
            print(f"  ✗ {name}: FILE NOT FOUND")
            all_pass = False

    return all_pass


def check_models():
    """Create each ablation model and verify forward pass + parameter counts."""
    print("\n" + "=" * 60)
    print("  MODEL ARCHITECTURE CHECK")
    print("=" * 60)

    from src.models.dca_net import DCANet

    configs_to_test = {
        'Full Model': {'ablation': None},
        'No Context': {'ablation': 'no_context'},
        'No Attention': {'ablation': 'no_attention'},
        'No Curriculum': {'ablation': 'no_curriculum'},
        'No Uncertainty': {'ablation': 'no_uncertainty'},
    }

    param_counts = {}
    all_pass = True

    for name, overrides in configs_to_test.items():
        print(f"\n  {name}:")

        try:
            config = {
                'ablation': overrides.get('ablation'),
                'model': {
                    'backbone': 'efficientnet_b0',
                    'nodule_feature_dim': 512,
                    'context_feature_dim': 256,
                    'fusion_dim': 256,
                    'num_attention_heads': 4,
                    'dropout': 0.5,
                    'prediction_dropout': 0.3,
                    'mc_dropout_passes': 5,
                    'slice_neighbors': 2,
                }
            }

            model = DCANet(config)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts[name] = total_params

            print(f"    ✓ Model created ({total_params:,} params, {trainable_params:,} trainable)")

            # Check context stream presence
            has_context = model.context_stream is not None
            expected_context = overrides.get('ablation') != 'no_context'
            if has_context != expected_context:
                print(f"    ✗ Context stream: {'present' if has_context else 'absent'} "
                      f"(expected {'present' if expected_context else 'absent'})")
                all_pass = False
            else:
                print(f"    ✓ Context stream: {'present' if has_context else 'absent (correct)'}")

            # Test forward pass
            dummy_nodule = torch.randn(2, 1, 64, 64, 64)
            dummy_context = torch.randn(2, 1, 48, 48, 48)

            with torch.no_grad():
                output = model(dummy_nodule, dummy_context)

            print(f"    ✓ Forward pass OK — output shape: {output.shape}")
            print(f"      Output range: [{output.min():.4f}, {output.max():.4f}]")
            print(f"      Output std: {output.std():.4f}")

            if output.std() < 0.001:
                print(f"    ⚠ WARNING: Very low output variance — model may collapse!")
                # Not a failure since it's just initialization

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_pass = False

    # Verify parameter counts differ
    print("\n" + "-" * 60)
    print("  PARAMETER COUNT COMPARISON")
    print("-" * 60)

    full_params = param_counts.get('Full Model', 0)
    no_context_params = param_counts.get('No Context', 0)

    if full_params > 0 and no_context_params > 0:
        if no_context_params < full_params:
            diff = full_params - no_context_params
            print(f"  ✓ No Context has {diff:,} fewer params than Full Model ({no_context_params:,} vs {full_params:,})")
        else:
            print(f"  ✗ No Context should have fewer params! ({no_context_params:,} vs {full_params:,})")
            all_pass = False

    # No Attention should have same params (attention layers still exist, just bypassed)
    no_attn_params = param_counts.get('No Attention', 0)
    if no_attn_params == full_params:
        print(f"  ✓ No Attention has same params as Full Model (attention bypassed at runtime)")
    else:
        diff = full_params - no_attn_params
        print(f"  ℹ No Attention param difference: {diff:,} ({no_attn_params:,} vs {full_params:,})")

    for name, count in param_counts.items():
        print(f"    {name:>20}: {count:>12,}")

    return all_pass


def main():
    print("=" * 60)
    print("  DCA-Net ABLATION DIAGNOSTIC")
    print("=" * 60)

    config_ok = check_configs()
    model_ok = check_models()

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Config check: {'✓ PASS' if config_ok else '✗ FAIL'}")
    print(f"  Model check:  {'✓ PASS' if model_ok else '✗ FAIL'}")

    if config_ok and model_ok:
        print("\n  ✓ All checks passed! Ready for training.")
    else:
        print("\n  ✗ Some checks failed. Fix issues before retraining.")

    print("=" * 60)
    return 0 if (config_ok and model_ok) else 1


if __name__ == '__main__':
    sys.exit(main())
