#!/usr/bin/env python3
"""
DCA-Net Trainer with AMP, DataParallel, and early stopping on AUC-ROC.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from pathlib import Path
import time
import logging
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix

from src.training.losses import DCANetLoss


class Trainer:
    """Training loop for DCA-Net.
    
    Features:
      - Mixed precision (AMP) for training only
      - DataParallel (multi-GPU)
      - Gradient clipping
      - CosineAnnealingWarmRestarts scheduler
      - Checkpoint save/load
      - Early stopping on AUC-ROC
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
        pos_weight = float(self.config.get('data', {}).get('positive_negative_ratio', 15.0))
        
        self.criterion = DCANetLoss(
            bce_weight=loss_weights.get('bce', 0.4),
            focal_weight=loss_weights.get('focal', 0.4),
            uncertainty_weight=loss_weights.get('uncertainty', 0.2),
            focal_gamma=self.train_cfg.get('focal_gamma', 2.0),
            focal_alpha=self.train_cfg.get('focal_alpha', 0.9375),  # 15/16 for 1:15 ratio
            label_smoothing=self.train_cfg.get('label_smoothing', 0.1),
            pos_weight=pos_weight
        )

        # Optimizer
        lr = self.train_cfg.get('learning_rate', 5e-5)
        wd = self.train_cfg.get('weight_decay', 1e-5)
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=wd
        )

        # Warmup + Scheduler
        self.warmup_epochs = self.train_cfg.get('warmup_epochs', 5)
        T0 = self.train_cfg.get('scheduler_T0', 15)
        Tmult = self.train_cfg.get('scheduler_Tmult', 2)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=T0, T_mult=Tmult
        )
        self.base_lr = lr

        # Mixed precision — ONLY used during training, NOT validation
        self.use_amp = self.train_cfg.get('use_amp', True) and torch.cuda.is_available()
        self.scaler = GradScaler('cuda', enabled=self.use_amp)
        self.grad_clip = self.train_cfg.get('gradient_clip', 0.5)
        self.accum_steps = self.train_cfg.get('gradient_accumulation_steps', 1)

        # Early stopping — based on AUC-ROC (higher = better)
        self.patience = self.train_cfg.get('early_stopping_patience', 15)
        self.best_val_auc = 0.0
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
        """Train for one epoch with gradient accumulation and warmup."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        nan_batches = 0

        # Linear warmup: scale LR from 10% to 100% over warmup_epochs
        if epoch < self.warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / self.warmup_epochs)
            for pg in self.optimizer.param_groups:
                pg['lr'] = self.base_lr * warmup_factor

        # Custom modern progress bar
        bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}", 
            leave=False,
            bar_format=bar_format,
            ascii=" █",             # Solid blocks instead of hashes
            colour='white'          # Clean white bar
        )
        self.optimizer.zero_grad()

        for batch_idx, (nodule, context, labels) in enumerate(pbar):
            nodule = nodule.to(self.device)
            context = context.to(self.device)
            labels = labels.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                logits = self.model(nodule, context)
                loss, loss_dict = self.criterion(logits, labels)
                loss = loss / self.accum_steps  # Scale loss for accumulation

            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                if nan_batches <= 3:  # Only log first few
                    self.logger.warning(f"NaN/Inf loss at train batch {batch_idx}, skipping")
                self.optimizer.zero_grad()  # Clear any stale gradients
                continue

            self.scaler.scale(loss).backward()

            # Step optimizer every accum_steps batches (or at end of epoch)
            if (batch_idx + 1) % self.accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                if self.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.accum_steps  # Unscale for logging
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({
                'loss': f"{loss.item() * self.accum_steps:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Logging
            if batch_idx % self.log_interval == 0:
                self.logger.info(
                    f"Epoch {epoch+1} | Batch {batch_idx}/{len(self.train_loader)} | "
                    f"Loss: {loss.item() * self.accum_steps:.4f} | BCE: {loss_dict['bce']:.4f} | "
                    f"Focal: {loss_dict['focal']:.4f} | Unc: {loss_dict['uncertainty']:.4f}"
                )
                if self.writer:
                    for k, v in loss_dict.items():
                        self.writer.add_scalar(f'train/{k}', v, self.global_step)

        if epoch >= self.warmup_epochs:
            self.scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)

        if nan_batches > 0:
            self.logger.warning(
                f"Epoch {epoch+1}: {nan_batches}/{len(self.train_loader)} batches had NaN loss"
            )

        return avg_loss, nan_batches

    def _compute_metrics(self, all_preds, all_labels):
        """Compute metrics for imbalanced binary classification."""
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        pred_binary = (preds_arr > 0.5).astype(int)

        metrics = {}

        # AUC-ROC
        try:
            if len(np.unique(labels_arr)) > 1:
                metrics['auc_roc'] = roc_auc_score(labels_arr, preds_arr)
            else:
                metrics['auc_roc'] = 0.0
        except Exception:
            metrics['auc_roc'] = 0.0

        # Average Precision (PR-AUC)
        try:
            if len(np.unique(labels_arr)) > 1:
                metrics['avg_precision'] = average_precision_score(labels_arr, preds_arr)
            else:
                metrics['avg_precision'] = 0.0
        except Exception:
            metrics['avg_precision'] = 0.0

        # Precision, Recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_arr, pred_binary, average='binary', zero_division=0
        )
        metrics['precision'] = precision
        metrics['recall'] = recall
        metrics['f1'] = f1

        # Confusion matrix
        try:
            tn, fp, fn, tp = confusion_matrix(labels_arr, pred_binary, labels=[0, 1]).ravel()
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
            metrics['tp'] = int(tp)
            metrics['fp'] = int(fp)
            metrics['tn'] = int(tn)
            metrics['fn'] = int(fn)
        except Exception:
            metrics['sensitivity'] = 0.0
            metrics['specificity'] = 0.0
            metrics['accuracy'] = (pred_binary == labels_arr).mean()

        return metrics

    @torch.no_grad()
    def validate(self, epoch):
        """Run validation on single GPU to avoid DataParallel eval-mode errors.
        
        DataParallel + MultiheadAttention in eval mode causes CUDA misaligned
        address errors. Solution: unwrap to model.module for validation.
        """
        # Unwrap DataParallel for validation (avoids CUDA misaligned address)
        if isinstance(self.model, nn.DataParallel):
            val_model = self.model.module
        else:
            val_model = self.model
        
        val_model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_labels = []

        for nodule, context, labels in self.val_loader:
            nodule = nodule.to(self.device)
            context = context.to(self.device)
            labels = labels.to(self.device)

            logits = val_model(nodule, context)
            loss, _ = self.criterion(logits, labels)

            if not torch.isnan(loss):
                total_loss += loss.item()
                num_batches += 1

            probs = torch.sigmoid(logits.squeeze(-1))
            probs = torch.clamp(probs, 1e-7, 1.0 - 1e-7)
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / max(num_batches, 1)

        # Compute metrics
        metrics = self._compute_metrics(all_preds, all_labels)

        # Debug: prediction distribution
        preds_arr = np.array(all_preds)
        labels_arr = np.array(all_labels)
        self.logger.info(
            f"Epoch {epoch+1} | Val preds: min={preds_arr.min():.4f} "
            f"max={preds_arr.max():.4f} mean={preds_arr.mean():.4f} | "
            f"Labels: {int(labels_arr.sum())}/{len(labels_arr)} positive"
        )

        self.logger.info(
            f"Epoch {epoch+1} | Val Loss: {avg_loss:.4f} | "
            f"AUC: {metrics['auc_roc']:.4f} | "
            f"Sens: {metrics['sensitivity']:.4f} | "
            f"Spec: {metrics['specificity']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f}"
        )

        if self.writer:
            self.writer.add_scalar('val/loss', avg_loss, epoch)
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(f'val/{k}', v, epoch)

        return avg_loss, metrics

    def save_checkpoint(self, epoch, val_loss, val_metrics=None, is_best=False):
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
            'val_metrics': val_metrics,
            'best_val_auc': self.best_val_auc,
            'global_step': self.global_step,
            'config': self.config,
        }

        # Save latest
        path = self.ckpt_dir / 'last.pth'
        torch.save(checkpoint, path)

        if is_best:
            best_path = self.ckpt_dir / 'best.pth'
            torch.save(checkpoint, best_path)
            auc_str = f"{val_metrics['auc_roc']:.4f}" if val_metrics else "N/A"
            self.logger.info(f"New best model saved (AUC: {auc_str}, val_loss: {val_loss:.4f})")

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
        self.best_val_auc = ckpt.get('best_val_auc', 0.0)
        
        self.logger.info(
            f"Resumed from epoch {self.start_epoch} "
            f"(val_loss: {self.best_val_loss:.4f}, best_auc: {self.best_val_auc:.4f})"
        )

    def train(self, num_epochs=None, dry_run=False):
        """Full training loop."""
        if num_epochs is None:
            num_epochs = self.train_cfg.get('num_epochs', 60)

        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Starting training: {num_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"AMP (training only): {self.use_amp}")
        self.logger.info(f"Early stopping: patience={self.patience}, metric=AUC-ROC")
        self.logger.info(f"{'='*60}\n")

        if dry_run:
            num_epochs = 1
            self.logger.info("DRY RUN MODE: running 2 batches only")

        for epoch in range(self.start_epoch, self.start_epoch + num_epochs):
            start = time.time()

            if dry_run:
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

                self.logger.info("\nDry run complete! Model can train successfully.")
                return

            # Normal training
            train_result = self.train_epoch(epoch)
            train_loss = train_result[0]
            nan_count = train_result[1] if len(train_result) > 1 else 0

            # NaN recovery: if >20% batches are NaN, weights are corrupted
            nan_threshold = len(self.train_loader) * 0.2
            if nan_count > nan_threshold:
                self.logger.warning(
                    f"\nWARNING: {nan_count} NaN batches detected — weights corrupted!"
                )
                recovery_ckpt = self.ckpt_dir / 'best.pth'
                if not recovery_ckpt.exists():
                    recovery_ckpt = self.ckpt_dir / 'last.pth'
                if recovery_ckpt.exists():
                    self.logger.info(f"Recovering from: {recovery_ckpt}")
                    self.load_checkpoint(recovery_ckpt)
                    # Lower learning rate to prevent recurrence
                    for pg in self.optimizer.param_groups:
                        pg['lr'] *= 0.5
                    self.logger.info(
                        f"Reduced LR to {self.optimizer.param_groups[0]['lr']:.2e}"
                    )
                    continue
                else:
                    self.logger.error("No checkpoint to recover from!")
                    break

            val_loss, val_metrics = self.validate(epoch)
            elapsed = time.time() - start

            val_auc = val_metrics.get('auc_roc', 0.0)

            self.logger.info(
                f"Epoch {epoch+1}/{self.start_epoch + num_epochs} | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val AUC: {val_auc:.4f} | Time: {elapsed:.1f}s"
            )

            # Early stopping on AUC-ROC
            is_best = val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = val_auc
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1

            self.save_checkpoint(epoch, val_loss, val_metrics=val_metrics, is_best=is_best)

            if self.epochs_no_improve >= self.patience:
                self.logger.info(
                    f"\nEarly stopping at epoch {epoch+1} "
                    f"(no AUC improvement for {self.patience} epochs)"
                )
                break

        if self.writer:
            self.writer.close()
        
        self.logger.info(
            f"\nTraining complete! Best AUC: {self.best_val_auc:.4f} | "
            f"Best val loss: {self.best_val_loss:.4f}"
        )
