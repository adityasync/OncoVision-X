#!/usr/bin/env python3
"""
OncoVision-X — 5-Fold Cross-Validation Training

Uses LUNA16 subsets 0-4 as 5 folds for rigorous model evaluation.
Each fold: 3 subsets train, 1 val, 1 test.

Usage:
    python train_kfold.py
    python train_kfold.py --folds 3      # Run only 3 folds
    python train_kfold.py --resume 2     # Resume from fold 2
"""

import argparse
import json
import logging
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import yaml

from src.models.dca_net import DCANet
from src.data.dataset import LunaDataset, create_data_loaders
from src.training.trainer import Trainer


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
║   AI-Powered Lung Cancer Detection — 5-Fold Cross-Validation       ║
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


def setup_logging(fold_id):
    """Setup logging for a specific fold."""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    ts = time.strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'kfold_fold{fold_id}_{ts}.log'
    
    logger = logging.getLogger(f'oncovision-x-fold{fold_id}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    
    fh = logging.FileHandler(log_file)
    ch = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    logger.info(f"Logging to: {log_file}")
    return logger


# 5-fold split configuration
# Each fold uses 3 subsets for training, 1 for validation, 1 for testing
FOLD_SPLITS = [
    {'train': [0, 1, 2], 'val': [3], 'test': [4]},  # Fold 0 (original)
    {'train': [1, 2, 3], 'val': [4], 'test': [0]},  # Fold 1
    {'train': [2, 3, 4], 'val': [0], 'test': [1]},  # Fold 2
    {'train': [3, 4, 0], 'val': [1], 'test': [2]},  # Fold 3
    {'train': [4, 0, 1], 'val': [2], 'test': [3]},  # Fold 4
]


def create_fold_dataloaders(config, fold_id, logger):
    """Create data loaders for a specific fold.
    
    Modifies the config to use the correct train/val/test subsets for this fold,
    then regenerates metadata and creates DataLoaders.
    """
    fold_cfg = FOLD_SPLITS[fold_id]
    
    logger.info(f"Fold {fold_id}: train={fold_cfg['train']}, "
                f"val={fold_cfg['val']}, test={fold_cfg['test']}")
    
    # Update config data splits
    fold_config = deepcopy(config)
    fold_config['data']['train_subsets'] = fold_cfg['train']
    fold_config['data']['val_subsets'] = fold_cfg['val']
    fold_config['data']['test_subsets'] = fold_cfg['test']
    
    # Regenerate metadata for this fold
    from generate_metadata import main as generate_metadata_main
    import generate_metadata as gm
    
    # Temporarily patch the module-level config
    original_subsets = None
    
    # Create fold-specific metadata by calling generate_metadata
    # with modified split config
    preprocessed_dir = Path(fold_config['data'].get('preprocessed_dir', 'preprocessed_data'))
    metadata_dir = preprocessed_dir / 'metadata'
    
    # Use existing all_samples.csv but re-split
    all_samples_csv = metadata_dir / 'all_samples.csv'
    if not all_samples_csv.exists():
        logger.error(f"all_samples.csv not found at {all_samples_csv}")
        logger.error("Run generate_metadata.py first to create the base metadata.")
        sys.exit(1)
    
    import pandas as pd
    all_df = pd.read_csv(all_samples_csv)
    
    # Extract subset from series_uid by matching against scan paths
    data_dir = Path(fold_config['data'].get('data_dir', 'data'))
    
    # Build series_uid → subset mapping
    uid_to_subset = {}
    for subset_dir in sorted(data_dir.glob('subset*/subset*')):
        subset_id = int(subset_dir.parent.name.replace('subset', ''))
        for mhd in subset_dir.glob('*.mhd'):
            uid = mhd.stem
            uid_to_subset[uid] = subset_id
    
    # Also check flat structure
    if not uid_to_subset:
        for subset_dir in sorted(data_dir.glob('subset*')):
            subset_id = int(subset_dir.name.replace('subset', ''))
            for mhd in subset_dir.glob('*.mhd'):
                uid = mhd.stem
                uid_to_subset[uid] = subset_id
    
    all_df['subset'] = all_df['series_uid'].map(uid_to_subset)
    
    # Split based on fold config
    train_df = all_df[all_df['subset'].isin(fold_cfg['train'])].copy()
    val_df = all_df[all_df['subset'].isin(fold_cfg['val'])].copy()
    test_df = all_df[all_df['subset'].isin(fold_cfg['test'])].copy()
    
    # Balance train set
    from generate_metadata import balance_samples
    train_df = balance_samples(train_df)
    
    # Save fold-specific CSVs
    fold_metadata_dir = metadata_dir / f'fold{fold_id}'
    fold_metadata_dir.mkdir(parents=True, exist_ok=True)
    
    train_csv = fold_metadata_dir / 'train_samples.csv'
    val_csv = fold_metadata_dir / 'val_samples.csv'
    test_csv = fold_metadata_dir / 'test_samples.csv'
    
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    logger.info(f"  Train: {len(train_df)} samples ({train_df['label'].sum():.0f} pos)")
    logger.info(f"  Val:   {len(val_df)} samples ({val_df['label'].sum():.0f} pos)")
    logger.info(f"  Test:  {len(test_df)} samples ({test_df['label'].sum():.0f} pos)")
    
    # Create DataLoaders
    data_cfg = fold_config.get('data', {})
    aug_config = data_cfg.get('augmentation', {})
    batch_size = fold_config.get('training', {}).get('batch_size', 64)
    
    train_dataset = LunaDataset(str(train_csv), augment=aug_config.get('enabled', True),
                                 aug_config=aug_config)
    val_dataset = LunaDataset(str(val_csv), augment=False)
    test_dataset = LunaDataset(str(test_csv), augment=False)
    
    loader_kwargs = {
        'num_workers': data_cfg.get('num_workers', 8),
        'pin_memory': data_cfg.get('pin_memory', True),
        'persistent_workers': data_cfg.get('num_workers', 8) > 0,
    }
    if data_cfg.get('num_workers', 8) > 0:
        loader_kwargs['prefetch_factor'] = data_cfg.get('prefetch_factor', 4)
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                               drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                             **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                              **loader_kwargs)
    
    return train_loader, val_loader, test_loader, fold_config


def evaluate_fold(model, test_loader, device, logger):
    """Evaluate a fold on test set and return metrics."""
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
    
    if isinstance(model, nn.DataParallel):
        eval_model = model.module
    else:
        eval_model = model
    
    eval_model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for nodule, context, labels in test_loader:
            nodule = nodule.to(device)
            context = context.to(device)
            
            logits = eval_model(nodule, context)
            probs = torch.sigmoid(logits.squeeze(-1))
            all_preds.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    pred_binary = (preds > 0.5).astype(int)
    
    metrics = {}
    try:
        metrics['auc_roc'] = roc_auc_score(labels, preds)
    except Exception:
        metrics['auc_roc'] = 0.0
    
    try:
        metrics['auc_pr'] = average_precision_score(labels, preds)
    except Exception:
        metrics['auc_pr'] = 0.0
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, pred_binary, average='binary', zero_division=0
    )
    metrics['precision'] = prec
    metrics['recall'] = rec
    metrics['f1'] = f1
    
    try:
        tn, fp, fn, tp = confusion_matrix(labels, pred_binary, labels=[0, 1]).ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    except Exception:
        metrics['sensitivity'] = 0.0
        metrics['specificity'] = 0.0
        metrics['accuracy'] = 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='OncoVision-X 5-Fold Cross-Validation')
    parser.add_argument('--config', default='configs/training_config.yaml')
    parser.add_argument('--folds', type=int, default=5, help='Number of folds (max 5)')
    parser.add_argument('--resume', type=int, default=0, help='Resume from fold N')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    num_folds = min(args.folds, 5)
    all_metrics = {}
    
    banner()
    section("CROSS-VALIDATION SETUP")
    info("Folds", str(num_folds))
    info("Config", args.config)
    info("Backbone", config['model'].get('backbone', 'efficientnet_b3'))
    info("Batch size", str(config['training'].get('batch_size', 64)))
    info("Epochs per fold", str(config['training'].get('num_epochs', 150)))
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            info(f"GPU {i}", f"{name} ({mem:.1f} GB)")
    
    for fold_id in range(args.resume, num_folds):
        section(f"FOLD {fold_id + 1}/{num_folds}")
        
        logger = setup_logging(fold_id)
        
        # Create fold data
        train_loader, val_loader, test_loader, fold_config = \
            create_fold_dataloaders(config, fold_id, logger)
        
        # Fresh model for each fold
        model = DCANet(fold_config)
        params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {params:,}")
        
        # Setup checkpoint dir for this fold
        fold_config['logging']['checkpoint_dir'] = f'results/checkpoints/fold{fold_id}'
        
        # Train
        trainer = Trainer(model, fold_config, train_loader, val_loader, logger)
        trainer.train()
        
        # Evaluate on test set
        logger.info("\nEvaluating on test set...")
        
        # Load best model for this fold
        best_ckpt = Path(f'results/checkpoints/fold{fold_id}/best.pth')
        if best_ckpt.exists():
            ckpt = torch.load(best_ckpt, map_location=trainer.device, weights_only=False)
            if isinstance(trainer.model, torch.nn.DataParallel):
                trainer.model.module.load_state_dict(ckpt['model_state_dict'])
            else:
                trainer.model.load_state_dict(ckpt['model_state_dict'])
        
        fold_metrics = evaluate_fold(trainer.model, test_loader, trainer.device, logger)
        all_metrics[f'fold_{fold_id}'] = fold_metrics
        
        logger.info(f"\nFold {fold_id} Test Results:")
        logger.info(f"  AUC-ROC:     {fold_metrics['auc_roc']:.4f}")
        logger.info(f"  AUC-PR:      {fold_metrics.get('auc_pr', 0):.4f}")
        logger.info(f"  Sensitivity: {fold_metrics['sensitivity']:.4f}")
        logger.info(f"  Specificity: {fold_metrics['specificity']:.4f}")
        logger.info(f"  F1-Score:    {fold_metrics['f1']:.4f}")
        
        # Clean up GPU memory
        del trainer, model
        torch.cuda.empty_cache()
    
    # Aggregate results
    section("CROSS-VALIDATION RESULTS")
    
    metric_names = ['auc_roc', 'auc_pr', 'sensitivity', 'specificity', 'f1', 'accuracy']
    summary = {}
    
    for metric in metric_names:
        values = [all_metrics[f'fold_{i}'][metric] 
                  for i in range(num_folds) if f'fold_{i}' in all_metrics]
        if values:
            mean = np.mean(values)
            std = np.std(values)
            summary[metric] = {'mean': mean, 'std': std, 'values': values}
            print(f"  {DIM}{metric:>15}:{RESET} {BOLD}{GREEN}{mean:.4f}{RESET} ± {std:.4f}  {values}")
    
    # Save results
    results_dir = Path('results/kfold')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / 'kfold_results.json'
    save_data = {
        'num_folds': num_folds,
        'config': config,
        'fold_metrics': all_metrics,
        'summary': {k: {'mean': v['mean'], 'std': v['std']} 
                    for k, v in summary.items()}
    }
    with open(results_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    
    print(f"\n  Results saved to: {results_file}")
    print(f"  Fold checkpoints: results/checkpoints/fold{{0-{num_folds-1}}}/")
    print()


if __name__ == '__main__':
    main()
