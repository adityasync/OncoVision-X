#!/usr/bin/env python3
"""
LUNA16 PyTorch Dataset for DCA-Net training.
Loads preprocessed .npz patches and applies on-the-fly augmentation.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class LunaDataset(Dataset):
    """
    PyTorch Dataset for LUNA16 preprocessed patches.
    
    Loads nodule (64³) and context (48³) patches from .npz files.
    Applies random augmentation during training.
    
    Args:
        csv_path: Path to metadata CSV (train_samples.csv, etc.)
        augment: Whether to apply data augmentation
        aug_config: Augmentation configuration dict
    """

    def __init__(self, csv_path, augment=False, aug_config=None):
        self.metadata = pd.read_csv(csv_path)
        self.augment = augment
        self.aug_config = aug_config or {}
        
        # Verify a sample exists
        if len(self.metadata) > 0:
            sample = self.metadata.iloc[0]
            if not Path(sample['nodule_path']).exists():
                raise FileNotFoundError(
                    f"Patch file not found: {sample['nodule_path']}. "
                    "Check that preprocessed_data/ paths are correct."
                )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Load patches
        nodule_patch = np.load(row['nodule_path'])['patch'].astype(np.float32)
        context_patch = np.load(row['context_path'])['patch'].astype(np.float32)
        label = np.float32(row['label'])

        # Apply augmentation
        if self.augment:
            nodule_patch, context_patch = self._augment(
                nodule_patch, context_patch
            )

        # Convert to tensors: add channel dim → (1, D, H, W)
        nodule_tensor = torch.from_numpy(nodule_patch).unsqueeze(0)
        context_tensor = torch.from_numpy(context_patch).unsqueeze(0)
        label_tensor = torch.tensor(label)

        return nodule_tensor, context_tensor, label_tensor

    def _augment(self, nodule, context):
        """Apply random augmentations to both patches consistently."""
        cfg = self.aug_config

        # Random rotation (90° increments along each axis)
        if cfg.get('rotation', True):
            k = np.random.randint(0, 4)
            axes = [(0, 1), (0, 2), (1, 2)]
            ax = axes[np.random.randint(0, 3)]
            nodule = np.rot90(nodule, k=k, axes=ax).copy()
            context = np.rot90(context, k=k, axes=ax).copy()

        # Random flip
        if cfg.get('flip', True):
            for axis in range(3):
                if np.random.rand() > 0.5:
                    nodule = np.flip(nodule, axis=axis).copy()
                    context = np.flip(context, axis=axis).copy()

        # Gaussian noise
        if cfg.get('noise', True):
            std = cfg.get('noise_std', 0.05)
            noise = np.random.normal(0, std, nodule.shape).astype(np.float32)
            nodule = nodule + noise
            noise_c = np.random.normal(0, std, context.shape).astype(np.float32)
            context = context + noise_c

        # Random intensity shift
        if cfg.get('intensity_shift', 0) > 0:
            shift = np.random.uniform(
                -cfg['intensity_shift'], cfg['intensity_shift']
            )
            nodule = nodule + shift
            context = context + shift

        # Clamp back to [-1, 1]
        nodule = np.clip(nodule, -1.0, 1.0)
        context = np.clip(context, -1.0, 1.0)

        return nodule, context


def create_data_loaders(config):
    """Create train, validation, and test DataLoaders from config.
    
    Args:
        config: Full training configuration dict
        
    Returns:
        train_loader, val_loader, test_loader
    """
    data_cfg = config.get('data', {})
    preprocessed_dir = Path(data_cfg.get('preprocessed_dir', 'preprocessed_data'))
    metadata_dir = preprocessed_dir / 'metadata'

    aug_config = data_cfg.get('augmentation', {})
    
    train_csv = metadata_dir / 'train_samples.csv'
    val_csv = metadata_dir / 'val_samples.csv'
    test_csv = metadata_dir / 'test_samples.csv'

    # Check files exist
    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Metadata CSV not found: {csv_path}. "
                "Run generate_metadata.py first."
            )

    train_dataset = LunaDataset(
        train_csv, augment=aug_config.get('enabled', True),
        aug_config=aug_config
    )
    val_dataset = LunaDataset(val_csv, augment=False)
    test_dataset = LunaDataset(test_csv, augment=False)

    loader_kwargs = {
        'num_workers': data_cfg.get('num_workers', 4),
        'pin_memory': data_cfg.get('pin_memory', True),
        'persistent_workers': data_cfg.get('persistent_workers', False)
            and data_cfg.get('num_workers', 4) > 0,
    }
    # prefetch_factor only valid when num_workers > 0
    if data_cfg.get('num_workers', 4) > 0:
        loader_kwargs['prefetch_factor'] = data_cfg.get('prefetch_factor', 2)

    batch_size = config.get('training', {}).get('batch_size', 16)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs
    )

    return train_loader, val_loader, test_loader
