#!/usr/bin/env python3
"""
Malignancy Classifier Training Script

Train the 3D ResNet malignancy classifier on LIDC-IDRI annotations.
This is a standalone training script for the demo classification layer.

Usage:
  python train_classifier.py                              # Train with default config
  python train_classifier.py --config configs/custom.yaml # Custom config
  python train_classifier.py --dry-run                    # Verify pipeline without full training

NOT part of the research paper — demo feature only.
"""

import argparse
import logging
import time
import sys
from pathlib import Path

import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from sklearn.metrics import roc_auc_score, accuracy_score

from src.models.malignancy_classifier import MalignancyClassifier
from src.data.malignancy_dataset import create_malignancy_loaders


def setup_logging(experiment_dir):
    """Configure logging to file and console."""
    experiment_dir = Path(experiment_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger('malignancy-classifier')
    logger.setLevel(logging.INFO)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(experiment_dir / 'training.log')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
    logger.addHandler(fh)

    return logger


def compute_class_weights(train_loader):
    """Compute inverse-frequency class weights for imbalanced data."""
    all_labels = []
    for _, labels in train_loader:
        all_labels.extend(labels.numpy())
    all_labels = np.array(all_labels)

    counts = np.bincount(all_labels, minlength=2)
    total = len(all_labels)
    weights = total / (2 * counts + 1e-8)
    return torch.FloatTensor(weights)


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, use_amp, logger, epoch):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (patches, labels) in enumerate(train_loader):
        patches = patches.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast('cuda', enabled=use_amp):
            logits = model(patches)
            loss = criterion(logits, labels)

        if torch.isnan(loss):
            logger.warning(f"NaN loss at batch {batch_idx}, skipping")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = running_loss / max(len(train_loader), 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Run validation and compute metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []

    for patches, labels in val_loader:
        patches = patches.to(device)
        labels = labels.to(device)

        logits = model(patches)
        loss = criterion(logits, labels)

        running_loss += loss.item()
        probs = torch.softmax(logits, dim=1)
        _, predicted = logits.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # Malignant probability
        all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / max(len(val_loader), 1)
    accuracy = accuracy_score(all_labels, all_preds) * 100

    try:
        if len(np.unique(all_labels)) > 1:
            auc = roc_auc_score(all_labels, all_probs)
        else:
            auc = 0.0
    except Exception:
        auc = 0.0

    return avg_loss, accuracy, auc


def main():
    parser = argparse.ArgumentParser(description='Train Malignancy Classifier')
    parser.add_argument('--config', type=str, default='configs/malignancy_classifier.yaml',
                        help='Path to config YAML')
    parser.add_argument('--dry-run', action='store_true',
                        help='Run 2 batches only to verify pipeline')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

    training_cfg = config.get('training', {})
    log_cfg = config.get('logging', {})

    # Setup
    experiment_dir = Path(log_cfg.get('experiment_dir', 'experiments/malignancy_classifier'))
    ckpt_dir = Path(log_cfg.get('checkpoint_dir', experiment_dir / 'checkpoints'))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(experiment_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Data
    logger.info("Loading data...")
    train_loader, val_loader = create_malignancy_loaders(config)
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # Model
    model = MalignancyClassifier(config).to(device)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {params:,}")

    # Loss with class weights
    class_weights = compute_class_weights(train_loader).to(device)
    logger.info(f"Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    lr = training_cfg.get('learning_rate', 1e-4)
    wd = training_cfg.get('weight_decay', 1e-5)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    num_epochs = training_cfg.get('num_epochs', 50)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_cfg.get('scheduler_T_max', num_epochs)
    )

    # AMP
    use_amp = training_cfg.get('use_amp', True) and torch.cuda.is_available()
    scaler = GradScaler('cuda', enabled=use_amp)

    # Resume
    start_epoch = 0
    best_val_auc = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_auc = ckpt.get('best_val_auc', 0.0)
        logger.info(f"Resumed from epoch {start_epoch} (best AUC: {best_val_auc:.4f})")

    # Dry run
    if args.dry_run:
        logger.info("DRY RUN: verifying pipeline with 2 batches...")
        model.train()
        for i, (patches, labels) in enumerate(train_loader):
            patches, labels = patches.to(device), labels.to(device)
            with autocast('cuda', enabled=use_amp):
                logits = model(patches)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            logger.info(f"  Batch {i + 1}/2 | Loss: {loss.item():.4f} | Output shape: {logits.shape}")
            if i >= 1:
                break
        logger.info("DRY RUN COMPLETE — pipeline works!")
        return

    # Training loop
    patience = training_cfg.get('early_stopping_patience', 15)
    epochs_no_improve = 0

    logger.info(f"\n{'=' * 60}")
    logger.info(f"Starting training: {num_epochs} epochs, lr={lr}, AMP={use_amp}")
    logger.info(f"Early stopping: patience={patience}, metric=AUC")
    logger.info(f"{'=' * 60}\n")

    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler,
            device, use_amp, logger, epoch
        )
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.1f}% | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.1f}%, AUC: {val_auc:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best
        is_best = val_auc > best_val_auc
        if is_best:
            best_val_auc = val_auc
            epochs_no_improve = 0
            best_path = ckpt_dir / 'malignancy_classifier_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_auc': best_val_auc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config,
            }, best_path)
            logger.info(f"  ★ New best model saved (AUC: {val_auc:.4f})")
        else:
            epochs_no_improve += 1

        # Save last
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_auc': best_val_auc,
            'val_loss': val_loss,
            'config': config,
        }, ckpt_dir / 'last.pth')

        # Early stopping
        if epochs_no_improve >= patience:
            logger.info(f"\nEarly stopping at epoch {epoch + 1} (no AUC improvement for {patience} epochs)")
            break

    logger.info(f"\nTraining complete! Best validation AUC: {best_val_auc:.4f}")


if __name__ == '__main__':
    main()
