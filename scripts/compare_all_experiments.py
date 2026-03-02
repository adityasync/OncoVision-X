#!/usr/bin/env python3
"""
Compare All Experiments - Generate Comparison Tables and Plots
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_manager import ExperimentManager


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
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
║   Experiment Comparison & Analysis                                 ║
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


def load_all_experiment_results(base_dir='experiments'):
    """Load results from all experiments"""
    
    experiments_data = []
    
    exp_names = [
        'full_model',
        'ablation_no_context',
        'ablation_no_attention',
        'ablation_no_curriculum',
        'ablation_no_uncertainty',
        'baseline_resnet3d18',
        'baseline_resnet2d18'
    ]
    
    for exp_name in exp_names:
        exp_manager = ExperimentManager(exp_name, base_dir)
        results = exp_manager.load_results()
        
        if results is not None:
            test_results = results.get('test_results', results)
            experiments_data.append({
                'name': exp_name,
                'display_name': format_experiment_name(exp_name),
                'results': test_results
            })
            success(f"Loaded: {exp_name}")
        else:
            print(f"  {RED}✗ Missing: {exp_name}{RESET}")
    
    return experiments_data


def format_experiment_name(name):
    """Format experiment name for display"""
    name_map = {
        'full_model': 'DCA-Net (Full)',
        'ablation_no_context': 'DCA-Net (No Context)',
        'ablation_no_attention': 'DCA-Net (No Attention)',
        'ablation_no_curriculum': 'DCA-Net (No Curriculum)',
        'ablation_no_uncertainty': 'DCA-Net (No Uncertainty)',
        'baseline_resnet3d18': '3D ResNet-18',
        'baseline_resnet2d18': '2D ResNet-18'
    }
    return name_map.get(name, name)


def create_comparison_table(experiments_data, output_dir):
    """Create comparison table"""
    
    # Create DataFrame
    data = []
    for exp in experiments_data:
        res = exp['results']
        data.append({
            'Method': exp['display_name'],
            'AUC': f"{res.get('auc_roc', 0):.4f}",
            'Sensitivity': f"{res.get('sensitivity', 0)*100:.1f}%",
            'Specificity': f"{res.get('specificity', 0)*100:.1f}%",
            'Precision': f"{res.get('precision', 0)*100:.1f}%",
            'F1-Score': f"{res.get('f1_score', res.get('f1', 0)):.4f}",
            'ECE': f"{res.get('ece', 0):.4f}",
            'FP/Scan': f"{res.get('fp_per_scan', 0):.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = Path(output_dir) / 'all_experiments_comparison.csv'
    df.to_csv(csv_path, index=False)
    success(f"Comparison table saved: {csv_path}")
    
    # Save as LaTeX
    latex_path = Path(output_dir) / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    success(f"LaTeX table saved: {latex_path}")
    
    # Print to console
    section("COMPARISON TABLE")
    print()
    print(df.to_string(index=False))
    print()
    
    return df


def create_comparison_plots(experiments_data, output_dir):
    """Create comparison plots"""
    
    # Extract metrics
    names = [exp['display_name'] for exp in experiments_data]
    aucs = [exp['results'].get('auc_roc', 0) for exp in experiments_data]
    sensitivities = [exp['results'].get('sensitivity', 0) * 100 for exp in experiments_data]
    specificities = [exp['results'].get('specificity', 0) * 100 for exp in experiments_data]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. AUC Comparison
    ax = axes[0, 0]
    colors = ['#2E86AB' if 'DCA-Net (Full)' in name else '#A23B72' 
              if 'ablation' in experiments_data[i]['name'] else '#F18F01'
              for i, name in enumerate(names)]
    bars = ax.barh(names, aucs, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('AUC-ROC Comparison', fontsize=14, fontweight='bold')
    min_auc = min(aucs) if aucs else 0
    ax.set_xlim([max(0, min_auc - 0.05), 1.0])
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Sensitivity Comparison
    ax = axes[0, 1]
    bars = ax.barh(names, sensitivities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Sensitivity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Comparison', fontsize=14, fontweight='bold')
    min_sens = min(sensitivities) if sensitivities else 0
    ax.set_xlim([max(0, min_sens - 10), 100])
    for i, (bar, sens) in enumerate(zip(bars, sensitivities)):
        ax.text(sens + 0.5, bar.get_y() + bar.get_height()/2,
                f'{sens:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Specificity Comparison
    ax = axes[1, 0]
    bars = ax.barh(names, specificities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Specificity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Specificity Comparison', fontsize=14, fontweight='bold')
    min_spec = min(specificities) if specificities else 0
    ax.set_xlim([max(0, min_spec - 10), 100])
    for i, (bar, spec) in enumerate(zip(bars, specificities)):
        ax.text(spec + 0.5, bar.get_y() + bar.get_height()/2,
                f'{spec:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Combined Radar Chart
    ax = axes[1, 1]
    categories = ['AUC', 'Sensitivity', 'Specificity']
    
    # Use first experiment for radar (whichever is available)
    radar_exp = experiments_data[0]
    radar_label = radar_exp['display_name']
    
    # Normalize metrics to 0-100 scale for radar
    full_model_data = [
        radar_exp['results'].get('auc_roc', 0) * 100,
        radar_exp['results'].get('sensitivity', 0) * 100,
        radar_exp['results'].get('specificity', 0) * 100
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    full_model_data += full_model_data[:1]
    angles += angles[:1]
    
    ax = plt.subplot(224, projection='polar')
    ax.plot(angles, full_model_data, 'o-', linewidth=2, label=radar_label, color='#2E86AB')
    ax.fill(angles, full_model_data, alpha=0.25, color='#2E86AB')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(85, 100)
    ax.set_title('Performance Profile', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(output_dir) / 'comparison_plots.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    success(f"Comparison plots saved: {plot_path}")


def create_ablation_analysis(experiments_data, output_dir):
    """Create ablation study specific analysis"""
    
    # Get full model as baseline
    full_model = next((exp for exp in experiments_data if exp['name'] == 'full_model'), None)
    if full_model is None:
        print(f"  {DIM}Skipping ablation analysis (full_model not found){RESET}")
        return
    full_auc = full_model['results'].get('auc_roc', 0)
    
    # Calculate differences for ablations
    ablation_data = []
    for exp in experiments_data:
        if 'ablation' in exp['name']:
            delta_auc = (exp['results'].get('auc_roc', 0) - full_auc) * 100
            ablation_data.append({
                'Component': exp['display_name'].replace('DCA-Net (No ', '').replace(')', ''),
                'AUC': f"{exp['results'].get('auc_roc', 0):.4f}",
                'Δ AUC': f"{delta_auc:+.2f}%",
                'Impact': 'High' if abs(delta_auc) > 2 else 'Medium' if abs(delta_auc) > 1 else 'Low'
            })
    
    df = pd.DataFrame(ablation_data)
    df = df.sort_values('Δ AUC')
    
    # Save
    csv_path = Path(output_dir) / 'ablation_analysis.csv'
    df.to_csv(csv_path, index=False)
    success(f"Ablation analysis saved: {csv_path}")
    
    section("ABLATION STUDY ANALYSIS")
    info("Full Model AUC", f"{full_auc:.4f}")
    print()
    print(df.to_string(index=False))
    print()


def main():
    """Main comparison function"""
    
    banner()
    
    section("LOADING EXPERIMENTS")
    
    # Load all experiments
    experiments_data = load_all_experiment_results()
    
    if len(experiments_data) == 0:
        print(f"  {RED}✗ No experiment results found!{RESET}")
        return
    
    info("Experiments loaded", str(len(experiments_data)))
    
    # Create output directory
    output_dir = Path('experiments/comparison_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(experiments_data, output_dir)
    
    # Create comparison plots
    create_comparison_plots(experiments_data, output_dir)
    
    # Ablation analysis
    create_ablation_analysis(experiments_data, output_dir)
    
    section("COMPARISON COMPLETE")
    success(f"Results saved to: {output_dir}")
    print()


if __name__ == '__main__':
    main()
