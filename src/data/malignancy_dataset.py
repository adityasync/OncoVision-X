#!/usr/bin/env python3
"""
Malignancy Dataset — LIDC-IDRI Malignancy Classification Data Pipeline

Handles:
  - Loading LIDC-IDRI malignancy annotations (1-5 scale)
  - Binary conversion: 1-2 → benign (0), 4-5 → malignant (1), skip 3
  - PyTorch Dataset for loading 64³ nodule patches with labels
  - 3D augmentation (rotation, flip, noise)

NOT part of the research paper — demo feature only.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


def prepare_malignancy_data(csv_path):
    """Load and prepare LIDC-IDRI malignancy annotations for binary classification.

    Converts the 5-point malignancy scale to binary:
      - Benign (0):      malignancy 1-2
      - Malignant (1):   malignancy 4-5
      - Skipped:         malignancy 3 (indeterminate)

    Args:
        csv_path: Path to LIDC annotations CSV with columns:
                  - nodule_id: unique identifier
                  - malignancy: 1-5 rating

    Returns:
        pd.DataFrame with added 'label' column (0 or 1)
    """
    annotations = pd.read_csv(csv_path)

    if 'malignancy' not in annotations.columns:
        raise ValueError(
            f"CSV must have a 'malignancy' column. Found: {list(annotations.columns)}"
        )

    # Convert to binary classification
    annotations['label'] = annotations['malignancy'].apply(
        lambda x: 0 if x <= 2 else (1 if x >= 4 else -1)
    )

    # Remove indeterminate cases (malignancy == 3)
    annotations = annotations[annotations['label'] != -1].reset_index(drop=True)

    benign_count = (annotations['label'] == 0).sum()
    malignant_count = (annotations['label'] == 1).sum()
    print(f"Malignancy data prepared:")
    print(f"  Benign nodules:    {benign_count}")
    print(f"  Malignant nodules: {malignant_count}")
    print(f"  Total:             {len(annotations)}")
    print(f"  Ratio (B:M):       {benign_count / max(malignant_count, 1):.1f}:1")

    return annotations


class MalignancyDataset(Dataset):
    """PyTorch Dataset for malignancy classification.

    Loads 64³ nodule patches from .npy files and returns (patch, label) pairs.

    Args:
        annotations_df: DataFrame with 'nodule_id' and 'label' columns
        patches_dir: Directory containing {nodule_id}.npy patch files
        augment: Whether to apply data augmentation
    """

    def __init__(self, annotations_df, patches_dir, augment=False):
        self.annotations = annotations_df.reset_index(drop=True)
        self.patches_dir = Path(patches_dir)
        self.augment = augment

        # Verify directory exists
        if not self.patches_dir.exists():
            raise FileNotFoundError(
                f"Patches directory not found: {self.patches_dir}"
            )

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load nodule patch (64×64×64)
        nodule_id = row['nodule_id']
        patch_path = self.patches_dir / f"{nodule_id}.npy"

        if not patch_path.exists():
            # Fallback: try .npz format
            npz_path = self.patches_dir / f"{nodule_id}.npz"
            if npz_path.exists():
                patch = np.load(npz_path)['patch'].astype(np.float32)
            else:
                raise FileNotFoundError(
                    f"Patch file not found: {patch_path} or {npz_path}"
                )
        else:
            patch = np.load(patch_path).astype(np.float32)

        # Apply augmentation
        if self.augment:
            patch = self._augment(patch)

        # Convert to tensor: add channel dim → (1, 64, 64, 64)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        label = torch.tensor(row['label'], dtype=torch.long)

        return patch_tensor, label

    def _augment(self, patch):
        """Apply random 3D augmentations.

        Consistent with the augmentation patterns used in the main
        LunaDataset for the detection task.
        """
        # Random 90° rotation along a random axis pair
        k = np.random.randint(0, 4)
        axes = [(0, 1), (0, 2), (1, 2)]
        ax = axes[np.random.randint(0, 3)]
        patch = np.rot90(patch, k=k, axes=ax).copy()

        # Random flip along each axis
        for axis in range(3):
            if np.random.rand() > 0.5:
                patch = np.flip(patch, axis=axis).copy()

        # Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 0.05, patch.shape).astype(np.float32)
            patch = patch + noise

        # Clamp to valid range
        patch = np.clip(patch, -1.0, 1.0)

        return patch


def create_malignancy_loaders(config):
    """Create train and validation DataLoaders for malignancy classification.

    Args:
        config: Configuration dict with 'data' and 'training' sections

    Returns:
        (train_loader, val_loader)
    """
    from sklearn.model_selection import train_test_split

    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})

    annotations = prepare_malignancy_data(data_cfg['annotations_csv'])

    # Stratified train/val split
    val_ratio = data_cfg.get('val_ratio', 0.2)
    train_df, val_df = train_test_split(
        annotations, test_size=val_ratio,
        stratify=annotations['label'], random_state=42
    )

    print(f"  Train split: {len(train_df)} samples")
    print(f"  Val split:   {len(val_df)} samples")

    patches_dir = data_cfg['patches_dir']
    train_dataset = MalignancyDataset(train_df, patches_dir, augment=True)
    val_dataset = MalignancyDataset(val_df, patches_dir, augment=False)

    batch_size = training_cfg.get('batch_size', 32)
    num_workers = data_cfg.get('num_workers', 4)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
