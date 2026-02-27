#!/usr/bin/env python3
"""
DCA-Net Comprehensive Evaluation Script.

Run after training to generate all metrics, plots, and reports.
Loads the best (or specified) checkpoint and evaluates on the test set.

Usage:
    python evaluate.py                        # Evaluate best model
    python evaluate.py --checkpoint path.pth  # Specific checkpoint
    python evaluate.py --no-uncertainty       # Skip MC Dropout (faster)
    python evaluate.py --output results/eval  # Custom output dir

Output files generated:
    results/evaluation/
    ├── evaluation_results.json     # All metrics as JSON
    ├── evaluation_report.txt       # Human-readable report
    ├── predictions.npz             # Raw predictions (probs, labels)
    └── figures/
        ├── roc_curve.png           # ROC curve with AUC
        ├── pr_curve.png            # Precision-Recall curve
        ├── confusion_matrix.png    # Confusion matrix heatmap
        ├── froc_curve.png          # FROC curve
        ├── calibration_diagram.png # Reliability diagram + histogram
        ├── score_distribution.png  # Prediction distributions (pos/neg)
        ├── uncertainty_distribution.png  # MC Dropout confidence
        ├── training_curves.png     # Loss & accuracy over epochs
        └── subgroup_analysis.png   # Performance by subgroup
"""

import argparse
import sys
import os
import glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.models.dca_net import DCANet
from src.data.dataset import create_data_loaders
from src.evaluation.evaluator import Evaluator


def find_latest_training_log(log_dir='logs'):
    """Find the most recent training log file."""
    pattern = os.path.join(log_dir, 'training_*.log')
    logs = glob.glob(pattern)
    if not logs:
        return None
    return max(logs, key=os.path.getmtime)


def parse_args():
    parser = argparse.ArgumentParser(
        description="DCA-Net Evaluation — Generate all metrics and plots"
    )
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to config YAML')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.pth). Default: results/checkpoints/best.pth')
    parser.add_argument('--output', type=str, default='results/evaluation',
                        help='Output directory for results and plots')
    parser.add_argument('--no-uncertainty', action='store_true',
                        help='Skip MC Dropout uncertainty estimation (faster)')
    parser.add_argument('--metadata-csv', type=str, default=None,
                        help='Path to metadata CSV with diameter info for subgroup analysis')
    parser.add_argument('--training-log', type=str, default=None,
                        help='Path to training log file for training curves')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size for evaluation')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size

    # Setup logging
    logger = setup_logging(config, name='dca-net-eval')

    logger.info("=" * 60)
    logger.info("DCA-NET COMPREHENSIVE EVALUATION")
    logger.info("=" * 60)

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"\nDevice: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # ── Load model ──
    logger.info("\nLoading model...")
    model = DCANet(config)

    # Find checkpoint
    ckpt_path = args.checkpoint
    if ckpt_path is None:
        candidates = [
            'results/checkpoints/best.pth',
            'results/checkpoints/last.pth',
        ]
        for c in candidates:
            if os.path.exists(c):
                ckpt_path = c
                break

    if ckpt_path and os.path.exists(ckpt_path):
        logger.info(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"  Trained for {ckpt.get('epoch', '?')+1} epochs")
        logger.info(f"  Best val loss: {ckpt.get('val_loss', '?')}")
    else:
        logger.warning("⚠️  No checkpoint found! Evaluating with random weights.")
        logger.warning("   Train the model first with: python train.py")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  Model parameters: {total_params:,}")

    # ── Create data loaders ──
    logger.info("\nCreating data loaders...")
    _, _, test_loader = create_data_loaders(config)
    logger.info(f"  Test batches: {len(test_loader)}")

    # ── Find metadata and training log ──
    metadata_csv = args.metadata_csv
    if metadata_csv is None:
        # Try to find test metadata
        default_meta = 'preprocessed_data/metadata/test_samples.csv'
        if os.path.exists(default_meta):
            metadata_csv = default_meta

    training_log = args.training_log
    if training_log is None:
        training_log = find_latest_training_log()
        if training_log:
            logger.info(f"  Found training log: {training_log}")

    # ── Run evaluation ──
    evaluator = Evaluator(model, test_loader, device=device, logger=logger)
    metrics = evaluator.evaluate(
        output_dir=args.output,
        run_uncertainty=not args.no_uncertainty,
        metadata_csv=metadata_csv,
        training_log=training_log,
    )

    # ── GradCAM Explainability ──
    logger.info("\n7. Generating GradCAM visualizations...")
    try:
        from src.explainability.gradcam import generate_gradcam_report
        gradcam_dir = os.path.join(args.output, 'figures', 'gradcam')
        gradcam_paths = generate_gradcam_report(
            model, test_loader, device, gradcam_dir,
            num_samples=10, stream='nodule_stream'
        )
        logger.info(f"   Generated {len(gradcam_paths)} GradCAM visualizations")
    except Exception as e:
        logger.warning(f"   GradCAM generation failed: {e}")
        logger.warning("   Continuing without GradCAM (non-critical)")

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nOutput directory: {args.output}/")
    logger.info("Files generated:")
    logger.info("  evaluation_results.json  — all metrics")
    logger.info("  evaluation_report.txt    — readable report")
    logger.info("  predictions.npz          — raw predictions")
    logger.info("  figures/                 — all plots")
    logger.info("  figures/gradcam/         — GradCAM heatmaps")

    # Print summary of key metrics
    logger.info(f"\nKEY METRICS:")
    logger.info(f"   AUC-ROC:     {metrics['auc_roc']:.4f}")
    logger.info(f"   Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"   Specificity: {metrics['specificity']:.4f}")
    logger.info(f"   F1-Score:    {metrics['f1_score']:.4f}")

    # Trained model output explanation
    logger.info(f"\nTRAINED MODEL OUTPUT:")
    if ckpt_path:
        logger.info(f"   Checkpoint: {ckpt_path}")
    logger.info(f"   Format: .pth (PyTorch state dict)")
    logger.info(f"   Contains: model weights, optimizer state, epoch, config")
    logger.info(f"   Load with: torch.load('results/checkpoints/best.pth')")


if __name__ == "__main__":
    main()
