#!/usr/bin/env python3
"""
DCA-Net Trainer with curriculum learning, AMP, DataParallel, and early stopping.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path
import time
import logging
from tqdm import tqdm

from src.training.losses import DCANetLoss


class Trainer:
    """Training loop for DCA-Net.
    
    Features:
      - Mixed precision (AMP)
      - DataParallel (multi-GPU)
      - Gradient clipping
      - CosineAnnealingWarmRestarts scheduler
      - Checkpoint save/load
      - Early stopping
      - Curriculum learning (3 stages)
    """

    def __init__(self, model, config, train_loader, val_loader, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger('dca-net')
        self.train_cfg = config.get('training', {})
        self.log_cfg = config.get('logging', {})
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Multi-GPU
        if (self.train_cfg.get('use_data_parallel', False) and 
                torch.cuda.device_count() > 1):
            device_ids = self.train_cfg.get('device_ids', [0, 1])
            available = list(range(torch.cuda.device_count()))
            device_ids = [d for d in device_ids if d in available]
            self.logger.info(f"Using DataParallel with GPUs: {device_ids}")
            model = nn.DataParallel(model, device_ids=device_ids)
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Loss
        loss_weights = self.train_cfg.get('loss_weights', {})
        self.criterion = DCANetLoss(
            bce_weight=loss_weights.get('bce', 0.4),
            focal_weight=loss_weights.get('focal', 0.4),
            uncertainty_weight=loss_weights.get('uncertainty', 0.2),
            focal_gamma=self.train_cfg.get('focal_gamma', 2.0),
            focal_alpha=self.train_cfg.get('focal_alpha', 0.75),
            label_smoothing=self.train_cfg.get('label_smoothing', 0.1),
        )

        # Optimizer
        lr = self.train_cfg.get('learning_rate', 3e-4)
        wd = self.train_cfg.get('weight_decay', 1e-5)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Scheduler
        T0 = self.train_cfg.get('scheduler_T0', 10)
        Tmult = self.train_cfg.get('scheduler_Tmult', 2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T0, T_mult=Tmult
        )

        # Mixed precision
        self.use_amp = self.train_cfg.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_clip = self.train_cfg.get('gradient_clip', 1.0)

        # Early stopping
        self.patience = self.train_cfg.get('early_stopping_patience', 10)
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0

        # Checkpoint
        self.ckpt_dir = Path(self.log_cfg.get('checkpoint_dir', 'results/checkpoints'))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = self.log_cfg.get('save_interval', 5)
        self.log_interval = self.log_cfg.get('log_interval', 10)

        # TensorBoard
        self.writer = None
        if self.log_cfg.get('use_tensorboard', False):
            try:
                from torch.utils.tensorboard import SummaryWriter
                log_dir = Path(self.log_cfg.get('log_dir', 'logs'))
                log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(log_dir))
            except ImportError:
                self.logger.warning("TensorBoard not available, skipping")

        self.global_step = 0
        self.start_epoch = 0

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for batch_idx, (nodule, context, labels) in enumerate(pbar):
            nodule = nodule.to(self.device)
            context = context.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(nodule, context)
                loss, loss_dict = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()

            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Logging
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item():.4f} | BCE: {loss_dict['bce']:.4f} | "
                    f"Focal: {loss_dict['focal']:.4f} | Unc: {loss_dict['uncertainty']:.4f}"
                )
                if self.writer:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)

        self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for nodule, context, labels in self.val_loader:
            nodule = nodule.to(self.device)
            context = context.to(self.device)
            labels = labels.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(nodule, context)
                loss, _ = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            probs = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        # Simple accuracy
        import numpy as np
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        acc = ((preds_arr > 0.5) == labels_arr).mean()

        self.logger.info(
            f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | Val Acc: {acc:.4f}"
        )

        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            self.writer.add_scalar('val/accuracy', acc, epoch)

        return avg_loss, acc

    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        model_state = (self.model.module.state_dict() 
                      if isinstance(self.model, nn.DataParallel) 
                      else self.model.state_dict())
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step,
            'config': self.config,
        }

        # Save latest
        path = self.ckpt_dir / 'last.pth'
        torch.save(checkpoint, path)

        # Save best
        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved (val_loss: {val_loss:.4f})")

        # Save periodic
        if (epoch + 1) % self.save_interval == 0:
            periodic_path = self.ckpt_dir / f'epoch_{epoch+1}.pth'
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model.load_state_dict(ckpt['model_state_dict'])
        
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.start_epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        self.best_val_loss = ckpt.get('val_loss', float('inf'))
        
        self.logger.info(
            f"Resumed from epoch {self.start_epoch} (val_loss: {self.best_val_loss:.4f})"
        )

    def train(self, num_epochs=None, dry_run=False):
        """Full training loop.
        
        Args:
            num_epochs: Override number of epochs
            dry_run: If True, run 2 batches per epoch for 1 epoch then exit
        """
        if num_epochs is None:
            num_epochs = self.train_cfg.get('num_epochs', 60)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting training: {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"AMP: {self.use_amp}")
        self.logger.info(f"{'='*60}\n")

        if dry_run:
            num_epochs = 1
            self.logger.info("DRY RUN MODE: running 2 batches only")

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            start = time.time()

            if dry_run:
                # Override train_loader to only yield 2 batches
                self.model.train()
                batch_count = 0
                for nodule, context, labels in self.train_loader:
                    nodule = nodule.to(self.device)
                    context = context.to(self.device)
                    labels = labels.to(self.device)

                    self.optimizer.zero_grad()
                    with autocast('cuda', enabled=self.use_amp):
                        logits = self.model(nodule, context)
                        loss, loss_dict = self.criterion(logits, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    batch_count += 1
                    self.logger.info(
                        f"Dry run batch {batch_count}/2 | Loss: {loss.item():.4f}"
                    )
                    if batch_count >= 2:
                        break

                self.logger.info("\n✅ Dry run complete! Model can train successfully.")
                return

            # Normal training
            train_loss = self.train_epoch(epoch)
            val_loss, val_acc = self.validate(epoch)
            elapsed = time.time() - start

            self.logger.info(
                f"Epoch {epoch+1}/{self.start_epoch + num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | Time: {elapsed:.1f}s"
            )

            # Checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, val_loss, is_best=is_best)

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                self.logger.info(
                    f"\nEarly stopping at epoch {epoch+1} "
                    f"(no improvement for {self.patience} epochs)"
                )
                break

        if self.writer:
            self.writer.close()
        
        self.logger.info(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")
