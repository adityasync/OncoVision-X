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
            experiments_data.append({
                'name': exp_name,
                'display_name': format_experiment_name(exp_name),
                'results': results['test_results']
            })
            print(f"✓ Loaded: {exp_name}")
        else:
            print(f"✗ Missing: {exp_name}")
    
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
            'AUC': f"{res['auc_roc']:.4f}",
            'Sensitivity': f"{res['sensitivity']*100:.1f}%",
            'Specificity': f"{res['specificity']*100:.1f}%",
            'Precision': f"{res['precision']*100:.1f}%",
            'F1-Score': f"{res['f1_score']:.4f}",
            'ECE': f"{res['ece']:.4f}",
            'FP/Scan': f"{res['fp_per_scan']:.2f}"
        })
    
    df = pd.DataFrame(data)
    
    # Save as CSV
    csv_path = Path(output_dir) / 'all_experiments_comparison.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Comparison table saved: {csv_path}")
    
    # Save as LaTeX
    latex_path = Path(output_dir) / 'comparison_table.tex'
    with open(latex_path, 'w') as f:
        f.write(df.to_latex(index=False, escape=False))
    print(f"✓ LaTeX table saved: {latex_path}")
    
    # Print to console
    print("\n" + "="*80)
    print("COMPARISON TABLE")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def create_comparison_plots(experiments_data, output_dir):
    """Create comparison plots"""
    
    # Extract metrics
    names = [exp['display_name'] for exp in experiments_data]
    aucs = [exp['results']['auc_roc'] for exp in experiments_data]
    sensitivities = [exp['results']['sensitivity'] * 100 for exp in experiments_data]
    specificities = [exp['results']['specificity'] * 100 for exp in experiments_data]
    
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
    ax.set_xlim([0.85, 1.0])
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        ax.text(auc + 0.002, bar.get_y() + bar.get_height()/2, 
                f'{auc:.4f}', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Sensitivity Comparison
    ax = axes[0, 1]
    bars = ax.barh(names, sensitivities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Sensitivity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Sensitivity Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([75, 100])
    for i, (bar, sens) in enumerate(zip(bars, sensitivities)):
        ax.text(sens + 0.5, bar.get_y() + bar.get_height()/2,
                f'{sens:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Specificity Comparison
    ax = axes[1, 0]
    bars = ax.barh(names, specificities, color=colors, alpha=0.8, edgecolor='black')
    ax.set_xlabel('Specificity (%)', fontsize=12, fontweight='bold')
    ax.set_title('Specificity Comparison', fontsize=14, fontweight='bold')
    ax.set_xlim([75, 95])
    for i, (bar, spec) in enumerate(zip(bars, specificities)):
        ax.text(spec + 0.5, bar.get_y() + bar.get_height()/2,
                f'{spec:.1f}%', va='center', fontsize=10, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Combined Radar Chart
    ax = axes[1, 1]
    categories = ['AUC', 'Sensitivity', 'Specificity']
    
    # Normalize metrics to 0-100 scale for radar
    full_model_data = [
        experiments_data[0]['results']['auc_roc'] * 100,
        experiments_data[0]['results']['sensitivity'] * 100,
        experiments_data[0]['results']['specificity'] * 100
    ]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    full_model_data += full_model_data[:1]
    angles += angles[:1]
    
    ax = plt.subplot(224, projection='polar')
    ax.plot(angles, full_model_data, 'o-', linewidth=2, label='DCA-Net (Full)', color='#2E86AB')
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
    print(f"✓ Comparison plots saved: {plot_path}")


def create_ablation_analysis(experiments_data, output_dir):
    """Create ablation study specific analysis"""
    
    # Get full model as baseline
    full_model = next(exp for exp in experiments_data if exp['name'] == 'full_model')
    full_auc = full_model['results']['auc_roc']
    
    # Calculate differences for ablations
    ablation_data = []
    for exp in experiments_data:
        if 'ablation' in exp['name']:
            delta_auc = (exp['results']['auc_roc'] - full_auc) * 100
            ablation_data.append({
                'Component': exp['display_name'].replace('DCA-Net (No ', '').replace(')', ''),
                'AUC': f"{exp['results']['auc_roc']:.4f}",
                'Δ AUC': f"{delta_auc:+.2f}%",
                'Impact': 'High' if abs(delta_auc) > 2 else 'Medium' if abs(delta_auc) > 1 else 'Low'
            })
    
    df = pd.DataFrame(ablation_data)
    df = df.sort_values('Δ AUC')
    
    # Save
    csv_path = Path(output_dir) / 'ablation_analysis.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Ablation analysis saved: {csv_path}")
    
    print("\n" + "="*60)
    print("ABLATION STUDY ANALYSIS")
    print("="*60)
    print(f"Full Model AUC: {full_auc:.4f}\n")
    print(df.to_string(index=False))
    print("="*60 + "\n")


def main():
    """Main comparison function"""
    
    print("\n" + "="*60)
    print("COMPARING ALL EXPERIMENTS")
    print("="*60 + "\n")
    
    # Load all experiments
    experiments_data = load_all_experiment_results()
    
    if len(experiments_data) == 0:
        print("No experiment results found!")
        return
    
    print(f"\nLoaded {len(experiments_data)} experiments\n")
    
    # Create output directory
    output_dir = Path('experiments/comparison_results')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comparison table
    df = create_comparison_table(experiments_data, output_dir)
    
    # Create comparison plots
    create_comparison_plots(experiments_data, output_dir)
    
    # Ablation analysis
    create_ablation_analysis(experiments_data, output_dir)
    
    print("\n" + "="*60)
    print("COMPARISON COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()
