#!/usr/bin/env python3
"""
DCA-Net Training Entry Point.
Optimized for Dual Quadro RTX 6000 (24GB x2).

Usage:
    python train.py                         # Train with default config
    python train.py --preprocess            # Run preprocessing first, then train
    python train.py --preprocess-only       # Run preprocessing only (no training)
    python train.py --config path/to.yaml   # Custom config
    python train.py --resume                # Resume from last checkpoint
    python train.py --dry-run               # Verify forward/backward pass (2 batches)
    python train.py --evaluate              # Evaluate best model on test set
"""

import argparse
import sys
import os
import subprocess

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.utils.logging_utils import setup_logging
from src.models.dca_net import DCANet
from src.data.dataset import create_data_loaders
from src.training.trainer import Trainer
from src.evaluation.evaluator import Evaluator


def run_preprocessing(logger):
    """Run the full preprocessing pipeline + metadata generation."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Step 1: Preprocess dataset (extract patches)
    preprocess_script = os.path.join(project_root, 'src', 'data', 'preprocess_dataset.py')
    metadata_dir = os.path.join(project_root, 'preprocessed_data', 'metadata')
    
    # Check if preprocessing is already done
    patches_exist = (
        os.path.exists(os.path.join(project_root, 'preprocessed_data', 'nodule_patches')) and
        len(os.listdir(os.path.join(project_root, 'preprocessed_data', 'nodule_patches'))) > 0
    )
    
    if not patches_exist:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 1: PREPROCESSING — Extracting patches from CT scans")
        logger.info("=" * 60)
        logger.info("This may take a while depending on your CPU...")
        
        result = subprocess.run(
            [sys.executable, preprocess_script],
            cwd=project_root,
            capture_output=False
        )
        if result.returncode != 0:
            logger.error("Preprocessing failed!")
            sys.exit(1)
    else:
        logger.info("✓ Patches already exist, skipping extraction")
    
    # Step 2: Generate metadata CSVs
    metadata_complete = (
        os.path.exists(metadata_dir) and
        os.path.exists(os.path.join(metadata_dir, 'train_samples.csv')) and
        os.path.exists(os.path.join(metadata_dir, 'val_samples.csv')) and
        os.path.exists(os.path.join(metadata_dir, 'test_samples.csv'))
    )
    
    if not metadata_complete:
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: GENERATING METADATA — Train/Val/Test splits")
        logger.info("=" * 60)
        
        gen_script = os.path.join(project_root, 'generate_metadata.py')
        result = subprocess.run(
            [sys.executable, gen_script],
            cwd=project_root,
            capture_output=False
        )
        if result.returncode != 0:
            logger.error("Metadata generation failed!")
            sys.exit(1)
    else:
        logger.info("✓ Metadata CSVs already exist, skipping generation")
    
    logger.info("\n✅ Preprocessing complete! Ready to train.\n")


def parse_args():
    parser = argparse.ArgumentParser(description="DCA-Net Training Pipeline")
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
                        help='Evaluate model on test set (use evaluate.py for full eval)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override config batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override config epochs')
    parser.add_argument('--lr', type=float, default=None,
                        help='Override config learning rate')
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    config = load_config(args.config)

    # Override config with CLI args
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr

    # Setup logging
    logger = setup_logging(config)

    logger.info("=" * 60)
    logger.info("DCA-NET — Lung Cancer Detection Pipeline")
    logger.info("=" * 60)

    # ── Preprocessing ──
    if args.preprocess or args.preprocess_only:
        run_preprocessing(logger)
        if args.preprocess_only:
            return

    # Print GPU info
    import torch
    logger.info(f"\nPyTorch Version: {torch.__version__}")
    logger.info(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.info(f"  GPU {i}: {name} ({mem:.1f} GB)")

    # Create model
    logger.info("\nInitializing DCA-Net model...")
    model = DCANet(config)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")

    # Create data loaders
    logger.info("\nCreating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(config)
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")
    logger.info(f"  Test batches: {len(test_loader)}")

    if args.evaluate:
        # Quick evaluation mode (use evaluate.py for full plots)
        logger.info("\nRunning quick evaluation...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = args.checkpoint or 'results/checkpoints/best.pth'
        if not os.path.exists(ckpt_path):
            logger.error(f"Checkpoint not found: {ckpt_path}")
            sys.exit(1)

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        model = model.to(device)

        evaluator = Evaluator(model, test_loader, device=device, logger=logger)
        results = evaluator.evaluate(
            output_dir='results/evaluation',
            run_uncertainty=True,
        )
        logger.info("\n💡 For full evaluation with all plots, run: python evaluate.py")
        return

    # Training mode
    trainer = Trainer(model, config, train_loader, val_loader, logger=logger)

    # Resume
    if args.resume:
        ckpt_path = args.checkpoint or 'results/checkpoints/last.pth'
        if os.path.exists(ckpt_path):
            trainer.load_checkpoint(ckpt_path)
        else:
            logger.warning(f"Checkpoint not found: {ckpt_path}, starting from scratch")

    # Train
    trainer.train(
        num_epochs=args.epochs,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE!")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("  1. Run full evaluation:  python evaluate.py")
        logger.info("  2. Check results in:     results/evaluation/")
        logger.info("  3. Trained model saved:   results/checkpoints/best.pth")


if __name__ == "__main__":
    main()
