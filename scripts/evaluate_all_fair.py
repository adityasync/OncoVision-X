#!/usr/bin/env python3
"""
Fair Evaluation — Evaluate ALL Models on the Same Test Set

Loads every trained model checkpoint and runs evaluation on the
identical test set with identical metrics, ensuring a fair comparison.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_fscore_support, confusion_matrix
)

from src.data.dataset import LunaDataset
from src.models.dca_net import DCANet
from src.models.baselines import ResNet3D18, ResNet2D18SliceLevel


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def load_model(experiment_name, device):
    """Load a trained model from its experiment directory."""
    import yaml

    exp_dir = Path(f'experiments/{experiment_name}')
    config_path = exp_dir / 'config.yaml'
    ckpt_path = exp_dir / 'checkpoints' / 'best.pth'

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Create model
    model_type = config.get('model', {}).get('type', 'dca_net')

    if model_type == 'dca_net':
        model = DCANet(config)
    elif model_type == 'resnet3d18':
        model = ResNet3D18(num_classes=config.get('model', {}).get('num_classes', 1))
    elif model_type == 'resnet2d18':
        model = ResNet2D18SliceLevel(num_classes=config.get('model', {}).get('num_classes', 1))
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Load weights
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = checkpoint['model_state_dict']

    # Handle DataParallel prefix
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', '?')
    best_auc = checkpoint.get('best_val_auc', 0.0)
    print(f"  ✓ Loaded {experiment_name} (epoch {epoch}, val_auc={best_auc:.4f})")

    return model, config


@torch.no_grad()
def evaluate_model(model, test_loader, device):
    """Evaluate model on test set, return metrics dict."""
    model.eval()

    all_preds = []
    all_labels = []

    for nodule, context, labels in test_loader:
        nodule = nodule.to(device)
        context = context.to(device)

        logits = model(nodule, context)
        probs = torch.sigmoid(logits.squeeze(-1))

        all_preds.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)
    pred_binary = (preds > 0.5).astype(int)

    metrics = {}

    # AUC-ROC
    try:
        if len(np.unique(labels)) > 1:
            metrics['auc_roc'] = float(roc_auc_score(labels, preds))
        else:
            metrics['auc_roc'] = 0.0
    except Exception:
        metrics['auc_roc'] = 0.0

    # AUC-PR
    try:
        if len(np.unique(labels)) > 1:
            metrics['auc_pr'] = float(average_precision_score(labels, preds))
        else:
            metrics['auc_pr'] = 0.0
    except Exception:
        metrics['auc_pr'] = 0.0

    # Precision, Recall, F1
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='binary', zero_division=0
    )
    metrics['precision'] = float(prec)
    metrics['recall'] = float(rec)
    metrics['f1'] = float(f1)

    # Confusion matrix
    try:
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
        metrics['sensitivity'] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics['accuracy'] = float((tp + tn) / (tp + tn + fp + fn))
        metrics['tp'] = int(tp)
        metrics['fp'] = int(fp)
        metrics['tn'] = int(tn)
        metrics['fn'] = int(fn)
    except Exception:
        metrics['sensitivity'] = 0.0
        metrics['specificity'] = 0.0
        metrics['accuracy'] = 0.0

    return metrics


def main():
    print(f"\n{BOLD}{BLUE}{'=' * 60}")
    print(f"  FAIR MODEL COMPARISON — SAME TEST SET FOR ALL")
    print(f"{'=' * 60}{RESET}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Create test loader (same for all models)
    test_csv = Path('preprocessed_data/metadata/test_samples.csv')
    if not test_csv.exists():
        print(f"{RED}  ✗ Test CSV not found: {test_csv}{RESET}")
        print(f"  Run generate_metadata.py first.")
        return 1

    test_dataset = LunaDataset(str(test_csv), augment=False)
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print(f"  Test set: {len(test_dataset)} samples, {len(test_loader)} batches\n")

    # Experiments to compare
    experiments = [
        'full_model',
        'ablation_no_context',
        'ablation_no_attention',
        'ablation_no_curriculum',
        'ablation_no_uncertainty',
        'baseline_resnet3d18',
        'baseline_resnet2d18',
    ]

    all_results = {}

    for exp_name in experiments:
        print(f"\n{BOLD}{'─' * 50}")
        print(f"  Evaluating: {exp_name}")
        print(f"{'─' * 50}{RESET}")

        try:
            model, config = load_model(exp_name, device)
            metrics = evaluate_model(model, test_loader, device)

            print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
            print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
            print(f"  Specificity: {metrics['specificity']:.4f}")
            print(f"  Precision:   {metrics['precision']:.4f}")
            print(f"  F1-Score:    {metrics['f1']:.4f}")

            all_results[exp_name] = metrics

            # Free GPU memory
            del model
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"{RED}  ✗ ERROR: {e}{RESET}")
            import traceback
            traceback.print_exc()

    # Save results
    output_dir = Path('experiments/fair_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / 'fair_comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print comparison table
    print(f"\n{BOLD}{BLUE}{'=' * 80}")
    print(f"  COMPARISON TABLE")
    print(f"{'=' * 80}{RESET}\n")

    header = f"{'Method':<30} {'AUC':>8} {'Sens':>8} {'Spec':>8} {'Prec':>8} {'F1':>8}"
    print(f"  {header}")
    print(f"  {'-' * 70}")

    name_map = {
        'full_model': 'DCA-Net (Full)',
        'ablation_no_context': 'DCA-Net (No Context)',
        'ablation_no_attention': 'DCA-Net (No Attention)',
        'ablation_no_curriculum': 'DCA-Net (No Curriculum)',
        'ablation_no_uncertainty': 'DCA-Net (No Uncertainty)',
        'baseline_resnet3d18': '3D ResNet-18',
        'baseline_resnet2d18': '2D ResNet-18',
    }

    for exp_name, metrics in all_results.items():
        display = name_map.get(exp_name, exp_name)
        print(f"  {display:<30} "
              f"{metrics['auc_roc']:>8.4f} "
              f"{metrics['sensitivity']:>8.4f} "
              f"{metrics['specificity']:>8.4f} "
              f"{metrics['precision']:>8.4f} "
              f"{metrics['f1']:>8.4f}")

    print(f"\n  Results saved to: {results_path}")
    print(f"{'=' * 80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
