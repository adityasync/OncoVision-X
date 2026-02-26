#!/usr/bin/env python3
"""
Visualization module for DCA-Net evaluation.
Generates all plots needed for the research paper.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix
)
from pathlib import Path
import json
import logging

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.dpi': 300,
})


def plot_roc_curve(labels, probs, output_path):
    """Plot ROC curve with AUC score."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='#2563EB', lw=2.5,
            label=f'DCA-Net (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve — Lung Nodule Classification')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    return roc_auc


def plot_precision_recall_curve(labels, probs, output_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, ax = plt.subplots()
    ax.plot(recall, precision, color='#16A34A', lw=2.5,
            label=f'DCA-Net (AP = {ap:.4f})')
    ax.set_xlabel('Recall (Sensitivity)')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    fig.savefig(output_path)
    plt.close(fig)
    return ap


def plot_confusion_matrix(labels, probs, output_path, threshold=0.5):
    """Plot confusion matrix heatmap."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(labels, preds, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 16})
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix (threshold = {threshold})')
    fig.savefig(output_path)
    plt.close(fig)


def plot_froc_curve(labels, probs, output_path):
    """Plot FROC curve: sensitivity at various false positive rates."""
    fpr, tpr, thresholds = roc_curve(labels, probs)
    n_neg = (labels == 0).sum()

    # Convert FPR to average FP count per scan
    fp_per_scan = fpr * n_neg / max(len(np.unique(labels)), 1)

    # Standard FROC reference points
    ref_fps = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    fig, ax = plt.subplots()
    ax.plot(fp_per_scan, tpr, color='#DC2626', lw=2.5, label='DCA-Net')

    # Mark reference points
    for fp_ref in ref_fps:
        idx = np.searchsorted(fp_per_scan, fp_ref)
        idx = min(idx, len(tpr) - 1)
        ax.plot(fp_ref, tpr[idx], 'ko', markersize=5)
        ax.annotate(f'{tpr[idx]:.2f}', (fp_ref, tpr[idx]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)

    ax.set_xscale('log')
    ax.set_xlim([0.1, 100])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Average False Positives per Scan')
    ax.set_ylabel('Sensitivity (True Positive Rate)')
    ax.set_title('FROC Curve')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')
    fig.savefig(output_path)
    plt.close(fig)


def plot_calibration_diagram(labels, probs, output_path, n_bins=10):
    """Plot reliability / calibration diagram."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = []
    bin_accuracies = []
    bin_counts = []

    for i in range(n_bins):
        mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
        if mask.sum() == 0:
            continue
        bin_centers.append(probs[mask].mean())
        bin_accuracies.append(labels[mask].mean())
        bin_counts.append(mask.sum())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9),
                                    gridspec_kw={'height_ratios': [3, 1]})

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins * 0.8,
            color='#7C3AED', alpha=0.7, edgecolor='black', label='DCA-Net')
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration / Reliability Diagram')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])

    # Histogram of predictions
    ax2.hist(probs, bins=n_bins, range=(0, 1), color='#7C3AED',
             alpha=0.6, edgecolor='black')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.set_title('Prediction Distribution')

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_uncertainty_distribution(mean_probs, confidences, labels, output_path):
    """Plot uncertainty / confidence distribution split by correct/incorrect."""
    preds = (mean_probs > 0.5).astype(int)
    correct = (preds == labels)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confidence distribution
    axes[0].hist(confidences[correct], bins=20, alpha=0.7, color='#16A34A',
                 label='Correct', edgecolor='black')
    axes[0].hist(confidences[~correct], bins=20, alpha=0.7, color='#DC2626',
                 label='Incorrect', edgecolor='black')
    axes[0].set_xlabel('Confidence Score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Confidence Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Confidence vs accuracy scatter
    conf_bins = np.linspace(0, 1, 11)
    bin_accs = []
    bin_confs = []
    for i in range(len(conf_bins) - 1):
        mask = (confidences >= conf_bins[i]) & (confidences < conf_bins[i + 1])
        if mask.sum() > 0:
            bin_confs.append(confidences[mask].mean())
            bin_accs.append(correct[mask].mean())

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Ideal')
    axes[1].scatter(bin_confs, bin_accs, s=80, color='#2563EB',
                    edgecolor='black', zorder=5)
    axes[1].set_xlabel('Mean Confidence')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Confidence vs Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_training_curves(log_path, output_path):
    """Plot training loss and validation curves from training log.
    
    Reads the training log file and parses epoch-level summaries.
    """
    train_losses = []
    val_losses = []
    val_accs = []
    epochs = []

    if not Path(log_path).exists():
        return

    with open(log_path, 'r') as f:
        for line in f:
            if 'Train Loss:' in line and 'Val Loss:' in line:
                parts = line.strip().split('|')
                for part in parts:
                    part = part.strip()
                    if part.startswith('Epoch'):
                        try:
                            ep = int(part.split('/')[0].replace('Epoch', '').strip())
                            epochs.append(ep)
                        except ValueError:
                            pass
                    elif 'Train Loss:' in part:
                        try:
                            train_losses.append(float(part.split(':')[1].strip()))
                        except (ValueError, IndexError):
                            pass
                    elif 'Val Loss:' in part:
                        try:
                            val_losses.append(float(part.split(':')[1].strip()))
                        except (ValueError, IndexError):
                            pass
                    elif 'Val Acc:' in part:
                        try:
                            val_accs.append(float(part.split(':')[1].strip()))
                        except (ValueError, IndexError):
                            pass

    if not epochs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curves
    ax1.plot(epochs[:len(train_losses)], train_losses, '-o', color='#2563EB',
             label='Train Loss', markersize=4)
    ax1.plot(epochs[:len(val_losses)], val_losses, '-s', color='#DC2626',
             label='Val Loss', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy curve
    if val_accs:
        ax2.plot(epochs[:len(val_accs)], val_accs, '-^', color='#16A34A',
                 label='Val Accuracy', markersize=4)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_subgroup_analysis(labels, probs, metadata_df, output_path):
    """Plot performance metrics broken down by nodule size category.
    
    Requires metadata_df to have a 'diameter_mm' column for true nodules.
    If not available, generates a simulated breakdown based on prediction confidence.
    """
    preds = (probs > 0.5).astype(int)
    
    # Try to get size info from metadata
    if metadata_df is not None and 'diameter_mm' in metadata_df.columns:
        size_bins = [0, 4, 6, 10, float('inf')]
        size_labels = ['Tiny (<4mm)', 'Small (4-6mm)', 'Medium (6-10mm)', 'Large (>10mm)']
        
        sensitivities = []
        counts = []
        for i in range(len(size_bins) - 1):
            mask = ((metadata_df['diameter_mm'] >= size_bins[i]) & 
                    (metadata_df['diameter_mm'] < size_bins[i + 1]) &
                    (labels == 1))
            if mask.sum() > 0:
                sens = (preds[mask] == 1).mean()
                sensitivities.append(sens)
                counts.append(mask.sum())
            else:
                sensitivities.append(0)
                counts.append(0)
    else:
        # Fallback: analyze by confidence quartiles
        pos_mask = labels == 1
        if pos_mask.sum() == 0:
            return
        pos_probs = probs[pos_mask]
        quartiles = np.percentile(pos_probs, [25, 50, 75])
        
        size_labels = ['Q1 (hardest)', 'Q2', 'Q3', 'Q4 (easiest)']
        bins = [0] + list(quartiles) + [1.01]
        sensitivities = []
        counts = []
        for i in range(len(bins) - 1):
            mask = (pos_probs >= bins[i]) & (pos_probs < bins[i + 1])
            if mask.sum() > 0:
                sensitivities.append((pos_probs[mask] > 0.5).mean())
                counts.append(mask.sum())
            else:
                sensitivities.append(0)
                counts.append(0)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ['#EF4444', '#F59E0B', '#10B981', '#3B82F6']
    bars = ax.bar(size_labels, sensitivities, color=colors, edgecolor='black',
                  alpha=0.8)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f'n={count}', ha='center', va='bottom', fontsize=11)

    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity by Nodule Subgroup')
    ax.set_ylim([0, 1.15])
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(output_path)
    plt.close(fig)


def plot_score_distribution(labels, probs, output_path):
    """Plot prediction score distributions for positive vs negative samples."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(probs[labels == 0], bins=50, alpha=0.6, color='#3B82F6',
            label='Negative', edgecolor='black', density=True)
    ax.hist(probs[labels == 1], bins=50, alpha=0.6, color='#EF4444',
            label='Positive', edgecolor='black', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', lw=1.5, alpha=0.7,
               label='Decision boundary')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title('Prediction Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.savefig(output_path)
    plt.close(fig)


def generate_all_plots(labels, probs, output_dir, mean_probs=None,
                       confidences=None, metadata_df=None, log_path=None):
    """Generate all evaluation plots and save to output_dir.
    
    Args:
        labels: numpy array of ground truth labels
        probs: numpy array of predicted probabilities
        output_dir: directory to save plots
        mean_probs: MC Dropout mean predictions (optional)
        confidences: MC Dropout confidence scores (optional)
        metadata_df: DataFrame with sample metadata (optional)
        log_path: path to training log file (optional)
        
    Returns:
        dict: paths to all generated plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger('dca-net')

    plots = {}

    # 1. ROC Curve
    logger.info("  Generating ROC curve...")
    p = output_dir / 'roc_curve.png'
    plot_roc_curve(labels, probs, p)
    plots['roc_curve'] = str(p)

    # 2. Precision-Recall Curve
    logger.info("  Generating PR curve...")
    p = output_dir / 'pr_curve.png'
    plot_precision_recall_curve(labels, probs, p)
    plots['pr_curve'] = str(p)

    # 3. Confusion Matrix
    logger.info("  Generating confusion matrix...")
    p = output_dir / 'confusion_matrix.png'
    plot_confusion_matrix(labels, probs, p)
    plots['confusion_matrix'] = str(p)

    # 4. FROC Curve
    logger.info("  Generating FROC curve...")
    p = output_dir / 'froc_curve.png'
    plot_froc_curve(labels, probs, p)
    plots['froc_curve'] = str(p)

    # 5. Calibration Diagram
    logger.info("  Generating calibration diagram...")
    p = output_dir / 'calibration_diagram.png'
    plot_calibration_diagram(labels, probs, p)
    plots['calibration_diagram'] = str(p)

    # 6. Score Distribution
    logger.info("  Generating score distribution...")
    p = output_dir / 'score_distribution.png'
    plot_score_distribution(labels, probs, p)
    plots['score_distribution'] = str(p)

    # 7. Uncertainty Distribution (if MC Dropout was run)
    if mean_probs is not None and confidences is not None:
        logger.info("  Generating uncertainty plots...")
        p = output_dir / 'uncertainty_distribution.png'
        plot_uncertainty_distribution(mean_probs, confidences, labels, p)
        plots['uncertainty_distribution'] = str(p)

    # 8. Training Curves (if log file provided)
    if log_path and Path(log_path).exists():
        logger.info("  Generating training curves...")
        p = output_dir / 'training_curves.png'
        plot_training_curves(log_path, p)
        plots['training_curves'] = str(p)

    # 9. Subgroup Analysis
    logger.info("  Generating subgroup analysis...")
    p = output_dir / 'subgroup_analysis.png'
    plot_subgroup_analysis(labels, probs, metadata_df, p)
    plots['subgroup_analysis'] = str(p)

    logger.info(f"  All plots saved to {output_dir}/")
    return plots
