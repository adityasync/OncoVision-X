#!/usr/bin/env python3
"""
Evaluation module for DCA-Net.
Computes classification metrics, FROC, calibration, uncertainty, and subgroup analysis.
Generates all publication-quality plots.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, f1_score, accuracy_score,
    confusion_matrix, classification_report
)
import torch
import torch.nn as nn
import logging
from tqdm import tqdm
from pathlib import Path
import json

from src.evaluation.visualizations import generate_all_plots


class Evaluator:
    """Comprehensive evaluation for lung nodule classification.
    
    Computes:
      - AUC-ROC, AUC-PR
      - Sensitivity, Specificity, F1
      - FROC (sensitivity at different FP rates)
      - Expected Calibration Error (ECE)
      - MC Dropout uncertainty metrics
      - Subgroup analysis
    
    Generates all plots via visualizations module.
    
    Args:
        model: DCANet model (or DataParallel wrapped)
        test_loader: DataLoader for test set
        device: torch device
        logger: Logger instance
    """

    def __init__(self, model, test_loader, device=None, logger=None):
        self.model = model
        self.test_loader = test_loader
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.logger = logger or logging.getLogger('dca-net')

    @torch.no_grad()
    def collect_predictions(self):
        """Run model on test set, collect predictions and labels."""
        self.model.eval()
        all_probs = []
        all_labels = []

        for nodule, context, labels in tqdm(self.test_loader, desc="Evaluating"):
            nodule = nodule.to(self.device)
            context = context.to(self.device)

            logits = self.model(nodule, context)
            probs = torch.sigmoid(logits.squeeze(-1))

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

        return np.array(all_probs), np.array(all_labels)

    def collect_uncertainty(self, mc_passes=5):
        """Run MC Dropout uncertainty estimation on test set.
        
        Args:
            mc_passes: Number of stochastic forward passes
            
        Returns:
            mean_probs, confidences, labels: numpy arrays
        """
        # Get the raw model (unwrap DataParallel)
        raw_model = self.model
        if isinstance(raw_model, nn.DataParallel):
            raw_model = raw_model.module

        self.logger.info(f"  Running MC Dropout ({mc_passes} passes)...")
        
        all_mean_probs = []
        all_confidences = []
        all_labels = []

        for nodule, context, labels in tqdm(self.test_loader, desc="MC Dropout"):
            nodule = nodule.to(self.device)
            context = context.to(self.device)

            mean_prob, confidence = raw_model.predict_with_uncertainty(
                nodule, context
            )

            all_mean_probs.extend(mean_prob.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_labels.extend(labels.numpy())

        return (np.array(all_mean_probs),
                np.array(all_confidences),
                np.array(all_labels))

    def compute_metrics(self, probs, labels, threshold=0.5):
        """Compute all classification metrics."""
        preds = (probs >= threshold).astype(int)

        # Handle edge case: single class
        has_both = len(np.unique(labels)) > 1

        auc_roc = roc_auc_score(labels, probs) if has_both else 0.0
        auc_pr = average_precision_score(labels, probs) if has_both else 0.0
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)

        tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        metrics = {
            'auc_roc': float(auc_roc),
            'auc_pr': float(auc_pr),
            'f1_score': float(f1),
            'accuracy': float(acc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'threshold': float(threshold),
        }
        return metrics

    def compute_froc(self, probs, labels, fp_rates=[0.5, 1, 2, 4, 8]):
        """Compute FROC — sensitivity at specific false positive rates."""
        if len(np.unique(labels)) < 2:
            return {f'sensitivity_at_{fp}fp': 0.0 for fp in fp_rates}
        
        fpr, tpr, _ = roc_curve(labels, probs)
        n_neg = (labels == 0).sum()

        froc = {}
        for fp_rate in fp_rates:
            target_fpr = min(fp_rate / max(n_neg, 1), 1.0)
            idx = np.searchsorted(fpr, target_fpr)
            idx = min(idx, len(tpr) - 1)
            froc[f'sensitivity_at_{fp_rate}fp'] = float(tpr[idx])

        return froc

    def compute_ece(self, probs, labels, n_bins=10):
        """Compute Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0

        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            if mask.sum() == 0:
                continue
            bin_conf = probs[mask].mean()
            bin_acc = labels[mask].mean()
            ece += mask.sum() * abs(bin_conf - bin_acc)

        ece /= len(probs)
        return float(ece)

    def compute_uncertainty_metrics(self, mean_probs, confidences, labels):
        """Compute uncertainty-specific metrics."""
        preds = (mean_probs > 0.5).astype(int)
        correct = (preds == labels)

        metrics = {
            'mean_confidence': float(confidences.mean()),
            'mean_confidence_correct': float(
                confidences[correct].mean() if correct.sum() > 0 else 0
            ),
            'mean_confidence_incorrect': float(
                confidences[~correct].mean() if (~correct).sum() > 0 else 0
            ),
            'uncertain_cases_ratio': float(
                (confidences < 0.7).sum() / len(confidences)
            ),
            'uncertain_cases_count': int((confidences < 0.7).sum()),
            'misclassified_flagged_by_uncertainty': 0.0,
        }

        # Key metric: what fraction of misclassified cases had low confidence?
        if (~correct).sum() > 0:
            flagged = ((~correct) & (confidences < 0.7)).sum()
            metrics['misclassified_flagged_by_uncertainty'] = float(
                flagged / (~correct).sum()
            )

        return metrics

    def evaluate(self, output_dir=None, run_uncertainty=True,
                 metadata_csv=None, training_log=None):
        """Run full evaluation pipeline with all metrics and plots.
        
        Args:
            output_dir: Directory to save results and plots
            run_uncertainty: Whether to run MC Dropout uncertainty estimation
            metadata_csv: Path to metadata CSV with diameter info
            training_log: Path to training log for training curves
            
        Returns:
            dict: All computed metrics
        """
        self.logger.info("\n" + "=" * 60)
        self.logger.info("COMPREHENSIVE EVALUATION")
        self.logger.info("=" * 60)

        # ── 1. Collect predictions ──
        self.logger.info("\n1. Collecting predictions...")
        probs, labels = self.collect_predictions()

        self.logger.info(f"   Samples: {len(probs)}")
        self.logger.info(f"   Positives: {(labels == 1).sum()}")
        self.logger.info(f"   Negatives: {(labels == 0).sum()}")

        # ── 2. Classification metrics ──
        self.logger.info("\n2. Computing classification metrics...")
        metrics = self.compute_metrics(probs, labels)

        # ── 3. FROC ──
        self.logger.info("3. Computing FROC...")
        froc = self.compute_froc(probs, labels)
        metrics.update(froc)

        # ── 4. Calibration ──
        self.logger.info("4. Computing calibration (ECE)...")
        metrics['ece'] = self.compute_ece(probs, labels)

        # ── 5. Uncertainty ──
        mean_probs = None
        confidences = None
        if run_uncertainty:
            self.logger.info("5. Running uncertainty estimation...")
            mean_probs, confidences, unc_labels = self.collect_uncertainty()
            uncertainty_metrics = self.compute_uncertainty_metrics(
                mean_probs, confidences, unc_labels
            )
            metrics['uncertainty'] = uncertainty_metrics
        else:
            self.logger.info("5. Skipping uncertainty estimation")

        # ── 6. Log all results ──
        self.logger.info("\n" + "=" * 60)
        self.logger.info("RESULTS SUMMARY")
        self.logger.info("=" * 60)

        result_lines = [
            f"  AUC-ROC:      {metrics['auc_roc']:.4f}",
            f"  AUC-PR:       {metrics['auc_pr']:.4f}",
            f"  Sensitivity:  {metrics['sensitivity']:.4f}",
            f"  Specificity:  {metrics['specificity']:.4f}",
            f"  Precision:    {metrics['precision']:.4f}",
            f"  F1-Score:     {metrics['f1_score']:.4f}",
            f"  Accuracy:     {metrics['accuracy']:.4f}",
            f"  ECE:          {metrics['ece']:.4f}",
        ]
        if 'sensitivity_at_1fp' in metrics:
            result_lines.append(f"  Sens@1FP:     {metrics['sensitivity_at_1fp']:.4f}")
            result_lines.append(f"  Sens@4FP:     {metrics['sensitivity_at_4fp']:.4f}")
        if 'uncertainty' in metrics:
            um = metrics['uncertainty']
            result_lines.extend([
                f"  Mean Conf (correct):     {um['mean_confidence_correct']:.4f}",
                f"  Mean Conf (incorrect):   {um['mean_confidence_incorrect']:.4f}",
                f"  Misclassified flagged:   {um['misclassified_flagged_by_uncertainty']:.2%}",
            ])

        for line in result_lines:
            self.logger.info(line)

        # ── 7. Save results and generate plots ──
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            figures_dir = output_dir / 'figures'
            figures_dir.mkdir(exist_ok=True)

            # Save metrics JSON
            with open(output_dir / 'evaluation_results.json', 'w') as f:
                json.dump(metrics, f, indent=2)

            # Save raw predictions
            np.savez(
                output_dir / 'predictions.npz',
                probs=probs, labels=labels,
                mean_probs=mean_probs if mean_probs is not None else [],
                confidences=confidences if confidences is not None else []
            )

            # Load metadata for subgroup analysis
            metadata_df = None
            if metadata_csv and Path(metadata_csv).exists():
                metadata_df = pd.read_csv(metadata_csv)

            # Generate all plots
            self.logger.info("\n6. Generating plots...")
            plot_paths = generate_all_plots(
                labels=labels,
                probs=probs,
                output_dir=figures_dir,
                mean_probs=mean_probs,
                confidences=confidences,
                metadata_df=metadata_df,
                log_path=training_log,
            )

            # Save plot index
            with open(output_dir / 'plot_index.json', 'w') as f:
                json.dump(plot_paths, f, indent=2)

            # Generate text report
            self._generate_text_report(metrics, output_dir / 'evaluation_report.txt')

            self.logger.info(f"\n✅ All results saved to {output_dir}/")
            self.logger.info(f"   - evaluation_results.json")
            self.logger.info(f"   - predictions.npz")
            self.logger.info(f"   - evaluation_report.txt")
            self.logger.info(f"   - figures/ ({len(plot_paths)} plots)")

        return metrics

    def _generate_text_report(self, metrics, output_path):
        """Generate a human-readable text report."""
        lines = [
            "=" * 60,
            "DCA-NET EVALUATION REPORT",
            "Lung Nodule Classification — LUNA16 Dataset",
            "=" * 60,
            "",
            "CLASSIFICATION METRICS",
            "-" * 40,
            f"  AUC-ROC:         {metrics['auc_roc']:.4f}",
            f"  AUC-PR:          {metrics['auc_pr']:.4f}",
            f"  Sensitivity:     {metrics['sensitivity']:.4f}",
            f"  Specificity:     {metrics['specificity']:.4f}",
            f"  Precision:       {metrics['precision']:.4f}",
            f"  F1-Score:        {metrics['f1_score']:.4f}",
            f"  Accuracy:        {metrics['accuracy']:.4f}",
            "",
            "CONFUSION MATRIX",
            "-" * 40,
            f"  True Positives:  {metrics['true_positives']}",
            f"  False Positives: {metrics['false_positives']}",
            f"  True Negatives:  {metrics['true_negatives']}",
            f"  False Negatives: {metrics['false_negatives']}",
            "",
            "FROC ANALYSIS",
            "-" * 40,
        ]

        for fp in [0.5, 1, 2, 4, 8]:
            key = f'sensitivity_at_{fp}fp'
            if key in metrics:
                lines.append(f"  Sensitivity @ {fp} FP/scan: {metrics[key]:.4f}")

        lines.extend([
            "",
            "CALIBRATION",
            "-" * 40,
            f"  ECE:             {metrics['ece']:.4f}",
        ])

        if 'uncertainty' in metrics:
            um = metrics['uncertainty']
            lines.extend([
                "",
                "UNCERTAINTY ANALYSIS",
                "-" * 40,
                f"  Mean Confidence (all):       {um['mean_confidence']:.4f}",
                f"  Mean Confidence (correct):   {um['mean_confidence_correct']:.4f}",
                f"  Mean Confidence (incorrect): {um['mean_confidence_incorrect']:.4f}",
                f"  Uncertain cases (<0.7):      {um['uncertain_cases_count']} "
                f"({um['uncertain_cases_ratio']:.1%})",
                f"  Misclassified flagged:       "
                f"{um['misclassified_flagged_by_uncertainty']:.1%}",
            ])

        lines.extend(["", "=" * 60])

        with open(output_path, 'w') as f:
            f.write('\n'.join(lines))
