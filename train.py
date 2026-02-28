#!/usr/bin/env python3
"""
OncoVision-X Training Pipeline
================================
AI-Powered Lung Cancer Detection using Dual-Context Attention Network.
Optimized for Dual Quadro RTX 6000 (25GB x2).

Usage:
    python train.py                         # Train with default config
    python train.py --preprocess            # Run preprocessing first
    python train.py --preprocess-only       # Preprocessing only (no training)
    python train.py --config path/to.yaml   # Custom config
    python train.py --resume                # Resume from last checkpoint
    python train.py --dry-run               # Verify 2 batches only
    python train.py --evaluate              # Quick evaluation
"""

import argparse
import sys
import os
import io
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────────────────────────────────────
# Tee — capture ALL stdout/stderr to log file
# ─────────────────────────────────────────────────────────────
class TeeWriter:
    """Write to both a file and the original stream (stdout/stderr).
    
    Captures everything: print(), tqdm, logger, etc.
    """
    def __init__(self, stream, log_file):
        self.stream = stream
        self.log_file = open(log_file, 'a', buffering=1)  # Line-buffered

    def write(self, data):
        self.stream.write(data)
        self.stream.flush()
        # Strip ANSI color codes for the log file
        import re
        clean = re.sub(r'\033\[[0-9;]*m', '', data)
        self.log_file.write(clean)
        self.log_file.flush()

    def flush(self):
        self.stream.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()

    def fileno(self):
        return self.stream.fileno()

    def isatty(self):
        return self.stream.isatty()


# ─────────────────────────────────────────────────────────────
# Display
# ─────────────────────────────────────────────────────────────
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
GREEN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"


def banner():
    print(f"""
{BOLD}{GREEN}╔════════════════════════════════════════════════════════════════════╗
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
║   AI-Powered Lung Cancer Detection — Training Pipeline             ║
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


# ─────────────────────────────────────────────────────────────
# Core functions
# ─────────────────────────────────────────────────────────────
def run_preprocessing(logger):
    """Run the full preprocessing pipeline + metadata generation."""
    import subprocess
    project_root = os.path.dirname(os.path.abspath(__file__))

    preprocess_script = os.path.join(project_root, 'src', 'data', 'preprocess_dataset.py')
    metadata_dir = os.path.join(project_root, 'preprocessed_data', 'metadata')

    patches_exist = (
        os.path.exists(os.path.join(project_root, 'preprocessed_data', 'nodule_patches')) and
        len(os.listdir(os.path.join(project_root, 'preprocessed_data', 'nodule_patches'))) > 0
    )

    if not patches_exist:
        section("PREPROCESSING — Extracting patches from CT scans")
        logger.info("Extracting patches from CT scans...")
        result = subprocess.run(
            [sys.executable, preprocess_script],
            cwd=project_root, capture_output=False
        )
        if result.returncode != 0:
            logger.error("Preprocessing failed!")
            sys.exit(1)
    else:
        success("Patches already exist, skipping extraction")

    metadata_complete = (
        os.path.exists(metadata_dir) and
        os.path.exists(os.path.join(metadata_dir, 'train_samples.csv')) and
        os.path.exists(os.path.join(metadata_dir, 'val_samples.csv')) and
        os.path.exists(os.path.join(metadata_dir, 'test_samples.csv'))
    )

    if not metadata_complete:
        section("METADATA — Generating Train/Val/Test splits")
        gen_script = os.path.join(project_root, 'generate_metadata.py')
        result = subprocess.run(
            [sys.executable, gen_script],
            cwd=project_root, capture_output=False
        )
        if result.returncode != 0:
            logger.error("Metadata generation failed!")
            sys.exit(1)
    else:
        success("Metadata CSVs already exist, skipping generation")

    success("Preprocessing complete! Ready to train.")


def parse_args():
    parser = argparse.ArgumentParser(
        description='OncoVision-X Training Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                        help='Path to training config YAML')
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing before training')
    parser.add_argument('--preprocess-only', action='store_true',
                        help='Run preprocessing only (no training)')
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Specific checkpoint path to resume from')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 2 training batches to verify everything works')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate model on test set')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override config batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override config epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override config learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Setup logging with full stdout/stderr capture ──
    from src.utils.config import load_config
    from src.utils.logging_utils import setup_logging
    from src.models.dca_net import DCANet
    from src.data.dataset import create_data_loaders
    from src.training.trainer import Trainer
    from src.evaluation.evaluator import Evaluator

    config = load_config(args.config)

    # Override config with CLI args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Create log file and tee stdout/stderr to it
    log_dir = Path(config.get('paths', {}).get('log_dir', 'logs'))
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    full_log_file = log_dir / f"full_output_{timestamp}.log"

    tee_out = TeeWriter(sys.stdout, str(full_log_file))
    tee_err = TeeWriter(sys.stderr, str(full_log_file))
    sys.stdout = tee_out
    sys.stderr = tee_err

    # Setup structured logger (also writes to its own log file)
    logger = setup_logging(config)

    # ── Banner ──
    banner()
    print(f"  {DIM}Full output log: {full_log_file}{RESET}")
    print(f"  {DIM}Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{RESET}")

    # ── Preprocessing ──
    if args.preprocess or args.preprocess_only:
        run_preprocessing(logger)
        if args.preprocess_only:
            return

    # ── System Info ──
    section("SYSTEM INFORMATION")

    import torch
    info("PyTorch", torch.__version__)
    info("CUDA", f"{'Available' if torch.cuda.is_available() else 'Not available'}")

    if torch.cuda.is_available():
        info("GPU Count", torch.cuda.device_count())
        total_vram = 0
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            total_vram += mem
            info(f"  GPU {i}", f"{name} ({mem:.1f} GB)")
        info("Total VRAM", f"{total_vram:.1f} GB")

    use_dp = config.get('training', {}).get('use_data_parallel', False)
    info("DataParallel", f"{'Enabled' if use_dp else 'Disabled'}")
    info("AMP", f"{'Enabled' if config.get('training', {}).get('use_amp', False) else 'Disabled (float32)'}")

    # ── Model ──
    section("MODEL")

    model = DCANet(config)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_cfg = config.get('model', {})
    info("Architecture", "OncoVision-X (Dual-Context Attention Network)")
    info("Backbone", model_cfg.get('backbone', 'efficientnet_b3'))
    info("Nodule stream", f"{model_cfg.get('nodule_feature_dim', 768)}D features")
    info("Context stream", f"{model_cfg.get('context_feature_dim', 512)}D features")
    info("Fusion", f"{model_cfg.get('fusion_dim', 512)}D, {model_cfg.get('num_attention_heads', 8)} attention heads")
    info("Total parameters", f"{total_params:,}")
    info("Trainable", f"{trainable_params:,}")

    # ── Data ──
    section("DATA")

    train_loader, val_loader, test_loader = create_data_loaders(config)

    train_total = len(train_loader.dataset)
    val_total = len(val_loader.dataset)
    test_total = len(test_loader.dataset)

    try:
        train_pos = int(train_loader.dataset.metadata['label'].sum())
        info("Train samples", f"{train_total} ({train_pos} positive, {train_total - train_pos} negative)")
    except Exception:
        info("Train samples", f"{train_total}")
    info("Val samples", f"{val_total}")
    info("Test samples", f"{test_total}")
    info("Train batches", f"{len(train_loader)}")
    info("Batch size", config.get('training', {}).get('batch_size', 64))

    # ── Training Config ──
    section("TRAINING CONFIGURATION")

    train_cfg = config.get('training', {})
    info("Epochs", train_cfg.get('num_epochs', 150))
    info("Learning rate", f"{train_cfg.get('learning_rate', 1e-4):.1e}")
    info("Optimizer", train_cfg.get('optimizer', 'AdamW'))
    info("Scheduler", train_cfg.get('scheduler', 'CosineAnnealingWarmRestarts'))
    info("Gradient clip", train_cfg.get('gradient_clip', 1.0))
    info("Label smoothing", train_cfg.get('label_smoothing', 0.1))
    info("Early stopping", f"patience={train_cfg.get('early_stopping_patience', 25)}")

    loss_cfg = train_cfg.get('loss_weights', {})
    info("Loss weights", f"BCE={loss_cfg.get('bce', 0.4)}, "
         f"Focal={loss_cfg.get('focal', 0.4)}, "
         f"Unc={loss_cfg.get('uncertainty', 0.2)}")

    # ── Evaluate mode ──
    if args.evaluate:
        section("EVALUATION")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = args.checkpoint or 'results/checkpoints/best.pth'
        if not os.path.exists(ckpt_path):
            print(f"  {RED}ERROR: Checkpoint not found: {ckpt_path}{RESET}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)

        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        results = evaluator.evaluate(output_dir='results/evaluation', run_uncertainty=True)

        info("AUC-ROC", f"{results['auc_roc']:.4f}")
        info("Sensitivity", f"{results['sensitivity']:.4f}")
        info("Specificity", f"{results['specificity']:.4f}")
        info("F1-Score", f"{results['f1_score']:.4f}")
        print(f"\n  {DIM}For full evaluation: python evaluate.py{RESET}")
        return

    # ── Train ──
    section("TRAINING")

    trainer = Trainer(model, config, train_loader, val_loader, logger=logger)

    # Resume
    if args.resume:
        ckpt_path = args.checkpoint or 'results/checkpoints/last.pth'
        if os.path.exists(ckpt_path):
            trainer.load_checkpoint(ckpt_path)
            success(f"Resumed from: {ckpt_path}")
        else:
            warn(f"Checkpoint not found: {ckpt_path}, starting from scratch")

    start_time = time.time()
    trainer.train(num_epochs=args.epochs, dry_run=args.dry_run)
    elapsed = time.time() - start_time

    if not args.dry_run:
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)

        section("TRAINING COMPLETE")
        print(f"""
  {BOLD}OncoVision-X training finished successfully!{RESET}

  {DIM}Results:{RESET}
    Best AUC-ROC:     {BOLD}{GREEN}{trainer.best_val_auc:.4f}{RESET}
    Best val loss:    {trainer.best_val_loss:.4f}
    Training time:    {hours}h {mins}m
    Epochs completed: {trainer.start_epoch + (args.epochs or train_cfg.get('num_epochs', 150))}

  {DIM}Output files:{RESET}
    Model checkpoint: results/checkpoints/best.pth
    Full output log:  {full_log_file}
    Training log:     logs/

  {DIM}Next steps:{RESET}
    1. Run evaluation:    {GREEN}python evaluate.py{RESET}
    2. Run 5-fold CV:     {GREEN}python train_kfold.py{RESET}
    3. Run demo:          {GREEN}python demo.py{RESET}
    4. Run prediction:    {GREEN}python predict.py --help{RESET}
""")

    # Cleanup tee
    sys.stdout = tee_out.stream
    sys.stderr = tee_err.stream
    tee_out.close()
    tee_err.close()

    print(f"Full output saved to: {full_log_file}")


if __name__ == "__main__":
    main()
