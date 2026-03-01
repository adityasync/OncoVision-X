#!/usr/bin/env python3
"""
Universal Evaluation Script - Generates all metrics and plots
"""

import argparse
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.experiment_manager import ExperimentManager


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate experiment')
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default='best',
                       choices=['best', 'last'])
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'])
    return parser.parse_args()


def evaluate_model(model, dataloader, device, exp_manager, split='test'):
    """
    Comprehensive model evaluation
    
    Returns:
        dict with all metrics
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_confidences = []
    
    print(f"\nRunning inference on {split} set...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            nodule, context, targets = batch
            nodule = nodule.to(device)
            context = context.to(device)
            targets = targets.to(device).float().unsqueeze(1)
            
            # Forward pass
            outputs = model(nodule, context)
            
            # Uncertainty estimation (MC Dropout with 5 passes)
            predictions_mc = []
            model.train()  # Enable dropout
            for _ in range(5):
                pred = model(nodule, context)
                predictions_mc.append(pred.cpu())
            model.eval()
            
            # Calculate mean and confidence
            predictions_mc = torch.stack(predictions_mc)
            predictions_mean = predictions_mc.mean(dim=0)
            predictions_var = predictions_mc.var(dim=0)
            confidence = 1 - (predictions_var / (predictions_var.max() + 1e-7))
            
            all_predictions.append(predictions_mean)
            all_targets.append(targets.cpu())
            all_confidences.append(confidence)
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} batches")
    
    # Concatenate all batches
    predictions = torch.cat(all_predictions).numpy().flatten()
    targets = torch.cat(all_targets).numpy().flatten()
    confidences = torch.cat(all_confidences).numpy().flatten()
    
    print(f"✓ Inference complete: {len(predictions)} samples")
    
    # Calculate all metrics
    results = calculate_all_metrics(
        predictions, targets, confidences
    )
    
    # Generate all plots
    generate_all_plots(
        predictions, targets, confidences, exp_manager, split
    )
    
    # Save detailed results
    save_detailed_results(
        predictions, targets, confidences, results, exp_manager, split
    )
    
    return results


def calculate_all_metrics(predictions, targets, confidences, threshold=0.5):
    """Calculate comprehensive metrics"""
    
    results = {}
    
    # Binary predictions
    pred_binary = (predictions > threshold).astype(int)
    
    # 1. AUC-ROC
    results['auc_roc'] = roc_auc_score(targets, predictions)
    
    # 2. Average Precision (PR-AUC)
    results['average_precision'] = average_precision_score(targets, predictions)
    
    # 3. Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(targets, pred_binary).ravel()
    results['true_negatives'] = int(tn)
    results['false_positives'] = int(fp)
    results['false_negatives'] = int(fn)
    results['true_positives'] = int(tp)
    
    # 4. Sensitivity (Recall) and Specificity
    results['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    results['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # 5. Precision and F1
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results['f1_score'] = 2 * (results['precision'] * results['sensitivity']) / \
                         (results['precision'] + results['sensitivity']) \
                         if (results['precision'] + results['sensitivity']) > 0 else 0.0
    
    # 6. Accuracy
    results['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # 7. Expected Calibration Error
    results['ece'] = calculate_ece(predictions, targets)
    
    # 8. Confidence statistics
    correct = (pred_binary == targets)
    results['avg_confidence_correct'] = confidences[correct].mean()
    results['avg_confidence_incorrect'] = confidences[~correct].mean()
    
    # 9. False Positives per Scan (assuming ~30 candidates per scan)
    total_scans = len(targets) / 30
    results['fp_per_scan'] = fp / total_scans
    
    return results


def calculate_ece(predictions, targets, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def generate_all_plots(predictions, targets, confidences, exp_manager, split):
    """Generate all evaluation plots"""
    
    print("\nGenerating plots...")
    
    # 1. ROC Curve
    plot_roc_curve(predictions, targets, exp_manager, split)
    
    # 2. Precision-Recall Curve
    plot_pr_curve(predictions, targets, exp_manager, split)
    
    # 3. Confusion Matrix
    plot_confusion_matrix(predictions, targets, exp_manager, split)
    
    # 4. Calibration Diagram
    plot_calibration(predictions, targets, exp_manager, split)
    
    # 5. Confidence Distribution
    plot_confidence_distribution(predictions, targets, confidences, exp_manager, split)
    
    # 6. Prediction Distribution
    plot_prediction_distribution(predictions, targets, exp_manager, split)
    
    print("✓ All plots generated")


def plot_roc_curve(predictions, targets, exp_manager, split):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(targets, predictions)
    auc = roc_auc_score(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'DCA-Net (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title(f'ROC Curve — Lung Nodule Classification ({split.capitalize()})', fontsize=16)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = exp_manager.get_plot_path(f'{split}_roc_curve')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC curve saved: {save_path}")


def plot_pr_curve(predictions, targets, exp_manager, split):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(targets, predictions)
    ap = average_precision_score(targets, predictions)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'DCA-Net (AP = {ap:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve ({split.capitalize()})', fontsize=16)
    plt.legend(loc='upper right', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = exp_manager.get_plot_path(f'{split}_pr_curve')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ PR curve saved: {save_path}")


def plot_confusion_matrix(predictions, targets, exp_manager, split, threshold=0.5):
    """Plot confusion matrix"""
    pred_binary = (predictions > threshold).astype(int)
    cm = confusion_matrix(targets, pred_binary)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                cbar_kws={'label': 'Count'},
                annot_kws={'fontsize': 20})
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.title(f'Confusion Matrix (threshold = {threshold}) - {split.capitalize()}', fontsize=16)
    plt.tight_layout()
    
    save_path = exp_manager.get_plot_path(f'{split}_confusion_matrix')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confusion matrix saved: {save_path}")


def plot_calibration(predictions, targets, exp_manager, split, n_bins=10):
    """Plot calibration diagram"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Top: Reliability diagram
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences_binned = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
        prop_in_bin = in_bin.sum()
        
        if prop_in_bin > 0:
            accuracy_in_bin = targets[in_bin].mean()
            avg_confidence_in_bin = predictions[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences_binned.append(avg_confidence_in_bin)
    
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
    ax1.bar(confidences_binned, accuracies, width=0.1, alpha=0.7, 
            color='purple', edgecolor='black', label='DCA-Net')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=14)
    ax1.set_ylabel('Fraction of Positives', fontsize=14)
    ax1.set_title('Calibration / Reliability Diagram', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Bottom: Prediction distribution
    ax2.hist(predictions[targets == 0], bins=50, alpha=0.6, color='blue', label='Negative')
    ax2.hist(predictions[targets == 1], bins=50, alpha=0.6, color='red', label='Positive')
    ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    ax2.set_xlabel('Predicted Probability', fontsize=14)
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title('Prediction Distribution', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = exp_manager.get_plot_path(f'{split}_calibration')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Calibration plot saved: {save_path}")


def plot_confidence_distribution(predictions, targets, confidences, exp_manager, split):
    """Plot confidence distribution"""
    pred_binary = (predictions > 0.5).astype(int)
    correct = (pred_binary == targets)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Confidence distribution by correctness
    ax1.hist(confidences[correct], bins=50, alpha=0.7, color='green', label='Correct')
    ax1.hist(confidences[~correct], bins=50, alpha=0.7, color='red', label='Incorrect')
    ax1.set_xlabel('Confidence Score', fontsize=14)
    ax1.set_ylabel('Count', fontsize=14)
    ax1.set_title('Confidence Distribution', fontsize=16)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Right: Confidence vs Accuracy
    conf_bins = np.linspace(0, 1, 11)
    bin_centers = (conf_bins[:-1] + conf_bins[1:]) / 2
    bin_accuracies = []
    
    for i in range(len(conf_bins) - 1):
        in_bin = (confidences >= conf_bins[i]) & (confidences < conf_bins[i+1])
        if in_bin.sum() > 0:
            acc = correct[in_bin].mean()
            bin_accuracies.append(acc)
        else:
            bin_accuracies.append(np.nan)
    
    ax2.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=2)
    ax2.plot(bin_centers, bin_accuracies, 'bo-', markersize=10, linewidth=2, label='DCA-Net')
    ax2.set_xlabel('Mean Confidence', fontsize=14)
    ax2.set_ylabel('Accuracy', fontsize=14)
    ax2.set_title('Confidence vs Accuracy', fontsize=16)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = exp_manager.get_plot_path(f'{split}_confidence_analysis')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Confidence analysis saved: {save_path}")


def plot_prediction_distribution(predictions, targets, exp_manager, split):
    """Plot prediction score distribution"""
    plt.figure(figsize=(12, 6))
    
    plt.hist(predictions[targets == 0], bins=50, alpha=0.6, color='blue', 
             label=f'Negative (n={(targets==0).sum()})')
    plt.hist(predictions[targets == 1], bins=50, alpha=0.6, color='red',
             label=f'Positive (n={(targets==1).sum()})')
    plt.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Decision boundary')
    plt.xlabel('Predicted Probability', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'Prediction Score Distribution - {split.capitalize()}', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = exp_manager.get_plot_path(f'{split}_prediction_distribution')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Prediction distribution saved: {save_path}")


def save_detailed_results(predictions, targets, confidences, results, exp_manager, split):
    """Save detailed results to JSON"""
    
    # Add additional statistics
    results['predictions_stats'] = {
        'mean': float(predictions.mean()),
        'std': float(predictions.std()),
        'min': float(predictions.min()),
        'max': float(predictions.max())
    }
    
    results['targets_stats'] = {
        'total': int(len(targets)),
        'positives': int(targets.sum()),
        'negatives': int((targets == 0).sum()),
        'class_ratio': f"1:{int((targets == 0).sum() / targets.sum())}"
    }
    
    # Save to JSON
    results_path = exp_manager.dirs['metrics'] / f'{split}_detailed_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Detailed results saved: {results_path}")


def print_results_summary(results):
    """Print formatted results summary"""
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS SUMMARY")
    print(f"{'='*60}")
    
    print(f"\nClassification Metrics:")
    print(f"  AUC-ROC:           {results['auc_roc']:.4f}")
    print(f"  Average Precision: {results['average_precision']:.4f}")
    print(f"  Accuracy:          {results['accuracy']:.4f}")
    print(f"  Sensitivity:       {results['sensitivity']:.4f}")
    print(f"  Specificity:       {results['specificity']:.4f}")
    print(f"  Precision:         {results['precision']:.4f}")
    print(f"  F1-Score:          {results['f1_score']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:    {results['true_negatives']}")
    print(f"  False Positives:   {results['false_positives']}")
    print(f"  False Negatives:   {results['false_negatives']}")
    print(f"  True Positives:    {results['true_positives']}")
    
    print(f"\nClinical Metrics:")
    print(f"  FP per Scan:       {results['fp_per_scan']:.2f}")
    print(f"  ECE (Calibration): {results['ece']:.4f}")
    
    print(f"\nConfidence Analysis:")
    print(f"  Avg Conf (Correct):   {results['avg_confidence_correct']:.4f}")
    print(f"  Avg Conf (Incorrect): {results['avg_confidence_incorrect']:.4f}")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    args = parse_args()
    
    # Load experiment
    exp_manager = ExperimentManager(args.experiment)
    
    # Load model
    checkpoint_path = exp_manager.get_checkpoint_path(args.checkpoint)
    # Load your model here
    
    # Load dataloader
    # test_loader = create_dataloader(config, args.split)
    
    # Evaluate
    # results = evaluate_model(model, test_loader, device, exp_manager, args.split)
    
    # Print summary
    # print_results_summary(results)
