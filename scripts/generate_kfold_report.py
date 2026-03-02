#!/usr/bin/env python3
"""
Generate Final K-Fold Cross-Validation Report
Combines original kfold_results.json + optimized threshold results.

Usage:
    python scripts/generate_kfold_report.py
    python scripts/generate_kfold_report.py --results_dir results/kfold
"""

import json
import argparse
import numpy as np
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
║   K-Fold Final Report Generator                                    ║
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


def generate_latex_table(original, optimized, output_path):
    """Generate LaTeX table for paper."""
    opt_thresh = optimized['optimal_threshold']['mean']

    lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{K-Fold Cross-Validation Results (5-Fold)}",
        "\\label{tab:kfold_results}",
        "\\begin{tabular}{lcc}",
        "\\toprule",
        f"\\textbf{{Metric}} & \\textbf{{Default ($t$=0.5)}} & "
        f"\\textbf{{Optimized ($t$$\\approx${opt_thresh:.2f})}} \\\\",
        "\\midrule",
    ]

    # AUC-ROC (threshold-independent, same for both)
    if 'auc_roc' in original.get('summary', {}):
        auc = original['summary']['auc_roc']
        lines.append(
            f"AUC-ROC & {auc['mean']:.4f} $\\pm$ {auc['std']:.4f} & "
            f"{auc['mean']:.4f} $\\pm$ {auc['std']:.4f} \\\\"
        )

    metric_map = [
        ('sensitivity', 'sensitivity', 'Sensitivity'),
        ('specificity', 'specificity', 'Specificity'),
        ('precision', 'precision', 'Precision'),
        ('f1', 'f1', 'F1-Score'),
    ]

    for orig_key, opt_key, display in metric_map:
        orig = original.get('summary', {}).get(orig_key, {'mean': 0, 'std': 0})
        opt = optimized.get('optimal_metrics', {}).get(opt_key, {'mean': 0, 'std': 0})
        lines.append(
            f"{display} & {orig['mean']:.4f} $\\pm$ {orig['std']:.4f} & "
            f"{opt['mean']:.4f} $\\pm$ {opt['std']:.4f} \\\\"
        )

    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    success(f"LaTeX table saved: {output_path}")


def generate_comparison_plot(original, optimized, output_path):
    """Generate comparison bar plot."""
    metrics = ['Precision', 'Recall/Sens.', 'F1-Score', 'Specificity']

    orig_summary = original.get('summary', {})
    orig_values = [
        orig_summary.get('precision', {}).get('mean', 0),
        orig_summary.get('sensitivity', {}).get('mean', 0),
        orig_summary.get('f1', {}).get('mean', 0),
        orig_summary.get('specificity', {}).get('mean', 0),
    ]

    opt_m = optimized.get('optimal_metrics', {})
    opt_values = [
        opt_m.get('precision', {}).get('mean', 0),
        opt_m.get('recall', {}).get('mean', 0),
        opt_m.get('f1', {}).get('mean', 0),
        opt_m.get('specificity', {}).get('mean', 0),
    ]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    bars1 = ax.bar(x - width / 2, orig_values, width,
                   label='Default (t=0.5)', color='#E74C3C', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width / 2, opt_values, width,
                   label=f'Optimized (t≈{optimized["optimal_threshold"]["mean"]:.2f})',
                   color='#27AE60', alpha=0.8, edgecolor='black')

    ax.set_xlabel('Metric', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('K-Fold Results: Default vs Optimized Threshold', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12)
    ax.legend(fontsize=12)
    ax.set_ylim([0, 1.1])
    ax.grid(axis='y', alpha=0.3)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., h,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    success(f"Comparison plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate k-fold report')
    parser.add_argument('--results_dir', type=str, default='experiments/kfold',
                        help='Base kfold results directory')
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Load original results
    orig_path = results_dir / 'kfold_results.json'
    if not orig_path.exists():
        print(f"ERROR: Original results not found at {orig_path}")
        return

    with open(orig_path) as f:
        original = json.load(f)

    # Load optimized results
    opt_path = results_dir / 'optimized' / 'optimized_kfold_summary.json'
    if not opt_path.exists():
        print(f"ERROR: Optimized results not found at {opt_path}")
        print("Run scripts/optimize_threshold.py first!")
        return

    with open(opt_path) as f:
        optimized = json.load(f)

    # Output directory
    report_dir = results_dir / 'final_report'
    report_dir.mkdir(parents=True, exist_ok=True)

    banner()
    
    section("LOADING DATA")

    section("GENERATING OUTPUTS")

    generate_latex_table(original, optimized, report_dir / 'kfold_table.tex')
    generate_comparison_plot(original, optimized, report_dir / 'kfold_comparison.png')

    # Combined JSON report
    combined = {
        'num_folds': optimized.get('num_folds', 5),
        'optimal_threshold': optimized['optimal_threshold'],
        'default_threshold': {'value': 0.5},
        'comparison': {
            'default': original.get('summary', {}),
            'optimized': optimized.get('optimal_metrics', {}),
        },
    }
    with open(report_dir / 'final_kfold_report.json', 'w') as f:
        json.dump(combined, f, indent=2)

    section("REPORT COMPLETE")
    success(f"Saved to: {report_dir}")
    info("kfold_table.tex", "LaTeX table for paper")
    info("kfold_comparison.png", "Comparison figure")
    info("final_kfold_report.json", "Complete results")
    print()


if __name__ == '__main__':
    main()
