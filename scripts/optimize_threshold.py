#!/usr/bin/env python3
"""
Threshold Optimization for K-Fold Results
Finds optimal decision threshold using Youden's index and re-evaluates all metrics.

Usage:
    python scripts/optimize_threshold.py
    python scripts/optimize_threshold.py --fold_dir results/kfold --output_dir results/kfold/optimized
"""

import json
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
BLUE = "\033[94m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner():
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════════════╗
║                                                                    ║
║    ██████╗ ███╗   ██╗ ██████╗ ██████╗                              ║
║   ██╔═══██╗████╗  ██║██╔════╝██╔═══██╗                             ║
║   ██║   ██║██╔██╗ ██║██║     ██║   ██║                             ║
║   ██║   ██║██║╚██╗██║██║     ██║   ██║                             ║
║   ╚██████╔╝██║ ╚████║╚██████╗╚██████╔╝                             ║
║    ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝ ╚═════╝                              ║
║   ██╗   ██╗██╗███████╗██╗ ██████╗ ███╗   ██╗    ██╗  ██╗           ║
║   ██║   ██║██║██╔════╝██║██╔═══██╗████╗  ██║    ╚██╗██╔╝           ║
║   ██║   ██║██║███████╗██║██║   ██║██╔██╗ ██║     ╚███╔╝            ║
║   ╚██╗ ██╔╝██║╚════██║██║██║   ██║██║╚██╗██║     ██╔██╗            ║
║    ╚████╔╝ ██║███████║██║╚██████╔╝██║ ╚████║    ██╔╝ ██╗           ║
║     ╚═══╝  ╚═╝╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝    ╚═╝  ╚═╝           ║
║                                                                    ║
║   Dual-Context Attention Network                                   ║
║   K-Fold Threshold Optimization                                    ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")


def section(title):
    print(f"\n{BOLD}{BLUE}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}")


def info(label, value):
    print(f"  {DIM}{label}:{RESET} {value}")


def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")


def load_fold_predictions(fold_dir):
    """Load predictions and targets from a fold."""
    # Check metrics/ subdirectory first (new layout), then root (legacy)
    metrics_dir = Path(fold_dir) / 'metrics'
    if (metrics_dir / 'predictions.npy').exists():
        pred_dir = metrics_dir
    else:
        pred_dir = Path(fold_dir)

    predictions_path = pred_dir / 'predictions.npy'
    targets_path = pred_dir / 'targets.npy'

    if not predictions_path.exists() or not targets_path.exists():
        raise FileNotFoundError(f"predictions.npy/targets.npy not found in {fold_dir}")

    return np.load(predictions_path), np.load(targets_path)


def calculate_metrics_at_threshold(predictions, targets, threshold):
    """Calculate all metrics at a specific threshold."""
    pred_binary = (predictions > threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, pred_binary, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    youden = recall + specificity - 1

    return {
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'f1': float(f1),
        'accuracy': float(accuracy),
        'youden': float(youden),
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }


def find_optimal_threshold(predictions, targets):
    """Find optimal threshold using Youden's index."""
    thresholds = np.linspace(0, 1, 201)
    results = []
    best_threshold = 0.5
    best_youden = -np.inf

    for t in thresholds:
        m = calculate_metrics_at_threshold(predictions, targets, t)
        results.append(m)
        if m['youden'] > best_youden:
            best_youden = m['youden']
            best_threshold = t

    return best_threshold, results


def plot_threshold_analysis(results, save_path):
    """Plot comprehensive threshold analysis."""
    thresholds = [r['threshold'] for r in results]
    precisions = [r['precision'] for r in results]
    recalls = [r['recall'] for r in results]
    f1s = [r['f1'] for r in results]
    specificities = [r['specificity'] for r in results]
    youdens = [r['youden'] for r in results]

    best_f1_idx = int(np.argmax(f1s))
    best_youden_idx = int(np.argmax(youdens))
    idx_05 = 100  # threshold = 0.5

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: Precision, Recall, F1 vs Threshold
    ax = axes[0, 0]
    ax.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
    ax.plot(thresholds, recalls, 'r-', lw=2, label='Recall')
    ax.plot(thresholds, f1s, 'g-', lw=2, label='F1-Score')
    ax.axvline(x=0.5, color='gray', ls='--', alpha=0.5, label='Default (0.5)')
    ax.axvline(x=thresholds[best_f1_idx], color='purple', ls='--', lw=2,
               label=f'Best F1 ({thresholds[best_f1_idx]:.3f})')
    ax.set_xlabel('Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Metrics vs Decision Threshold', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Top-right: Operating Points
    ax = axes[0, 1]
    ax.plot([1 - s for s in specificities], recalls, 'b-', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax.plot(1 - specificities[idx_05], recalls[idx_05],
            'ro', ms=12, label=f'Default (0.5)')
    ax.plot(1 - specificities[best_youden_idx], recalls[best_youden_idx],
            'go', ms=12, label=f'Optimal ({thresholds[best_youden_idx]:.3f})')
    ax.set_xlabel('1 - Specificity (FPR)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Sensitivity (TPR)', fontsize=14, fontweight='bold')
    ax.set_title('Operating Points', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Youden's Index
    ax = axes[1, 0]
    ax.plot(thresholds, youdens, 'purple', lw=2)
    ax.axvline(x=thresholds[best_youden_idx], color='green', ls='--', lw=2,
               label=f'Optimal ({thresholds[best_youden_idx]:.3f})')
    ax.set_xlabel('Threshold', fontsize=14, fontweight='bold')
    ax.set_ylabel("Youden's Index", fontsize=14, fontweight='bold')
    ax.set_title("Youden's Index vs Threshold", fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])

    # Bottom-right: Comparison table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Metric', 'Thresh=0.5', f'Thresh={thresholds[best_youden_idx]:.3f}', 'Change'],
        ['Precision', f'{precisions[idx_05]:.3f}', f'{precisions[best_youden_idx]:.3f}',
         f'{(precisions[best_youden_idx] - precisions[idx_05])*100:+.1f}%'],
        ['Recall', f'{recalls[idx_05]:.3f}', f'{recalls[best_youden_idx]:.3f}',
         f'{(recalls[best_youden_idx] - recalls[idx_05])*100:+.1f}%'],
        ['F1-Score', f'{f1s[idx_05]:.3f}', f'{f1s[best_youden_idx]:.3f}',
         f'{(f1s[best_youden_idx] - f1s[idx_05])*100:+.1f}%'],
        ['Specificity', f'{specificities[idx_05]:.3f}', f'{specificities[best_youden_idx]:.3f}',
         f'{(specificities[best_youden_idx] - specificities[idx_05])*100:+.1f}%'],
    ]
    table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    for i in range(1, len(table_data)):
        if table_data[i][3].startswith('+'):
            table[(i, 3)].set_facecolor('#C6E0B4')
        else:
            table[(i, 3)].set_facecolor('#F4B183')
    ax.set_title('Threshold Comparison', fontsize=16, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Threshold analysis plot saved: {save_path}")


def analyze_all_folds(base_dir, output_dir):
    """Analyze all k-fold results with threshold optimization."""
    base_path = Path(base_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    fold_dirs = sorted([d for d in base_path.glob('fold_*') if d.is_dir()])

    if not fold_dirs:
        print(f"ERROR: No fold_* directories found in {base_dir}")
        print("Make sure you've run train_kfold.py with prediction saving first.")
        return

    banner()
    section("LOADING FOLDS")

    all_fold_results = {}

    for fold_dir in fold_dirs:
        fold_name = fold_dir.name
        print(f"\nProcessing {fold_name}...")

        try:
            predictions, targets = load_fold_predictions(fold_dir)
            print(f"  Loaded {len(predictions)} predictions")

            optimal_threshold, threshold_results = find_optimal_threshold(predictions, targets)
            print(f"  Optimal threshold: {optimal_threshold:.3f}")

            optimal_metrics = calculate_metrics_at_threshold(predictions, targets, optimal_threshold)
            default_metrics = calculate_metrics_at_threshold(predictions, targets, 0.5)

            all_fold_results[fold_name] = {
                'optimal_threshold': optimal_threshold,
                'optimal_metrics': optimal_metrics,
                'default_metrics': default_metrics,
            }

            print(f"\n  {'':20s} Default (0.5)   Optimal ({optimal_threshold:.3f})")
            print(f"    {'Precision':15s} {default_metrics['precision']:.4f}          {optimal_metrics['precision']:.4f}")
            print(f"    {'Recall':15s} {default_metrics['recall']:.4f}          {optimal_metrics['recall']:.4f}")
            print(f"    {'F1-Score':15s} {default_metrics['f1']:.4f}          {optimal_metrics['f1']:.4f}")
            print(f"    {'Specificity':15s} {default_metrics['specificity']:.4f}          {optimal_metrics['specificity']:.4f}")

            plot_path = output_path / f'{fold_name}_threshold_analysis.png'
            plot_threshold_analysis(threshold_results, plot_path)

        except Exception as e:
            print(f"  ERROR: {e}")
            continue

    if not all_fold_results:
        print("\nERROR: No folds successfully processed")
        return

    # Summary across all folds
    section("SUMMARY ACROSS ALL FOLDS")

    opt_thresholds = [r['optimal_threshold'] for r in all_fold_results.values()]
    summary = {
        'num_folds': len(all_fold_results),
        'optimal_threshold': {
            'mean': float(np.mean(opt_thresholds)),
            'std': float(np.std(opt_thresholds)),
        },
        'optimal_metrics': {},
        'default_metrics': {},
    }

    for metric in ['precision', 'recall', 'f1', 'sensitivity', 'specificity', 'accuracy']:
        opt_vals = [r['optimal_metrics'][metric] for r in all_fold_results.values()]
        def_vals = [r['default_metrics'][metric] for r in all_fold_results.values()]
        summary['optimal_metrics'][metric] = {
            'mean': float(np.mean(opt_vals)), 'std': float(np.std(opt_vals))
        }
        summary['default_metrics'][metric] = {
            'mean': float(np.mean(def_vals)), 'std': float(np.std(def_vals))
        }

    print(f"Optimal Threshold: {summary['optimal_threshold']['mean']:.3f} "
          f"± {summary['optimal_threshold']['std']:.3f}")
    print(f"\nMetrics at Optimal Threshold:")
    for m in ['precision', 'recall', 'f1', 'sensitivity', 'specificity']:
        v = summary['optimal_metrics'][m]
        print(f"  {m:>15s}: {v['mean']:.4f} ± {v['std']:.4f}")

    summary_path = output_path / 'optimized_kfold_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    success(f"Summary saved: {summary_path}")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize thresholds for k-fold results')
    parser.add_argument('--fold_dir', type=str, default='experiments/kfold',
                        help='Directory containing fold_* subdirectories')
    parser.add_argument('--output_dir', type=str, default='experiments/kfold/optimized',
                        help='Output directory for optimized results')
    args = parser.parse_args()

    analyze_all_folds(args.fold_dir, args.output_dir)
