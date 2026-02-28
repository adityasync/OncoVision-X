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
║   AI-Powered Lung Cancer Detection — Evaluation Pipeline           ║
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


def warn(msg):
    print(f"  {YELLOW}! {msg}{RESET}")


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

    banner()

    section("DCA-NET COMPREHENSIVE EVALUATION")

    # ── Device ──
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info("Device", str(device))
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            info(f"GPU {i}", f"{name} ({mem:.1f} GB)")

    # ── Load model ──
    section("MODEL INITIALIZATION")
    logger.info("Loading model...")
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
        success(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        info("Trained for", f"{ckpt.get('epoch', '?')+1} epochs")
        info("Best val loss", f"{ckpt.get('val_loss', '?')}")
    else:
        warn("No checkpoint found! Evaluating with random weights.")
        warn("Train the model first with: python train.py")

    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    info("Model parameters", f"{total_params:,}")

    # ── Create data loaders ──
    section("DATA LOADERS")
    logger.info("Creating data loaders...")
    _, _, test_loader = create_data_loaders(config)
    info("Test batches", f"{len(test_loader)}")

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
            info("Found training log", training_log)

    # ── Run evaluation ──
    section("EVALUATION")
    evaluator = Evaluator(model, test_loader, device=device, logger=logger)
    metrics = evaluator.evaluate(
        output_dir=args.output,
        run_uncertainty=not args.no_uncertainty,
        metadata_csv=metadata_csv,
        training_log=training_log,
    )

    # ── GradCAM Explainability ──
    section("GRADCAM EXPLAINABILITY")
    logger.info("Generating GradCAM visualizations...")
    try:
        from src.explainability.gradcam import generate_gradcam_report
        gradcam_dir = os.path.join(args.output, 'figures', 'gradcam')
        gradcam_paths = generate_gradcam_report(
            model, test_loader, device, gradcam_dir,
            num_samples=10, stream='nodule_stream'
        )
        success(f"Generated {len(gradcam_paths)} GradCAM visualizations")
    except Exception as e:
        warn(f"GradCAM generation failed: {e}")
        warn("Continuing without GradCAM (non-critical)")

    section("EVALUATION COMPLETE")
    print(f"\n  {DIM}Output directory:{RESET} {args.output}/")
    print(f"  {DIM}Files generated:{RESET}")
    print("    evaluation_results.json  — all metrics")
    print("    evaluation_report.txt    — readable report")
    print("    predictions.npz          — raw predictions")
    print("    figures/                 — all plots")
    print("    figures/gradcam/         — GradCAM heatmaps")

    # Print summary of key metrics
    print(f"\n  {BOLD}KEY METRICS:{RESET}")
    print(f"    AUC-ROC:     {BOLD}{GREEN}{metrics['auc_roc']:.4f}{RESET}")
    print(f"    Sensitivity: {metrics['sensitivity']:.4f}")
    print(f"    Specificity: {metrics['specificity']:.4f}")
    print(f"    F1-Score:    {metrics['f1_score']:.4f}")

    # Trained model output explanation
    if ckpt_path:
        print(f"\n  {BOLD}TRAINED MODEL OUTPUT:{RESET}")
        print(f"    Checkpoint: {ckpt_path}")
        print(f"    Format:     .pth (PyTorch state dict)")
        print(f"    Load with:  torch.load('results/checkpoints/best.pth')")


if __name__ == "__main__":
    main()
