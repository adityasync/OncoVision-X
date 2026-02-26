#!/usr/bin/env python3
"""
Generate metadata CSVs from preprocessing checkpoint.
Reads checkpoint.json and creates train/val/test split CSVs.

Handles cases where all data is from a single subset by falling back
to random patient-level splitting (60/20/20).
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("preprocessed_data")
DATA_DIR = Path("data")
HARD_NEG_RATIO = 5
RANDOM_NEG_RATIO = 2


def get_scan_paths():
    """Get all .mhd scan file paths organized by subset."""
    scan_paths = {}
    for i in range(5):
        subset_dir = DATA_DIR / f"subset{i}" / f"subset{i}"
        if subset_dir.exists():
            mhd_files = list(subset_dir.glob("*.mhd"))
            scan_paths[f"subset{i}"] = mhd_files
    return scan_paths


def create_splits(metadata_df, scan_paths):
    """Create train/val/test splits.
    
    Tries subset-based splitting first (subset0-2=train, subset3=val, subset4=test).
    Falls back to random patient-level split if that produces empty sets.
    """
    train_series = set()
    val_series = set()
    test_series = set()

    for subset_name, paths in scan_paths.items():
        for path in paths:
            series_uid = path.stem
            if subset_name in ['subset0', 'subset1', 'subset2']:
                train_series.add(series_uid)
            elif subset_name == 'subset3':
                val_series.add(series_uid)
            elif subset_name == 'subset4':
                test_series.add(series_uid)

    # Check if the subset-based split works
    data_series = set(metadata_df['series_uid'].unique())
    train_match = data_series & train_series
    val_match = data_series & val_series
    test_match = data_series & test_series

    if len(val_match) > 0 and len(test_match) > 0:
        # Subset-based split works
        train_df = metadata_df[metadata_df['series_uid'].isin(train_series)]
        val_df = metadata_df[metadata_df['series_uid'].isin(val_series)]
        test_df = metadata_df[metadata_df['series_uid'].isin(test_series)]
        print("  Using subset-based splitting")
    else:
        # Fallback: random patient-level split (60/20/20)
        print("  Subset-based split produced empty val/test sets.")
        print("  Falling back to random patient-level split (60/20/20)...")
        
        unique_series = metadata_df['series_uid'].unique()
        np.random.shuffle(unique_series)
        
        n = len(unique_series)
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))
        
        train_uids = set(unique_series[:n_train])
        val_uids = set(unique_series[n_train:n_train + n_val])
        test_uids = set(unique_series[n_train + n_val:])
        
        # If test is empty (very few patients), take from train
        if len(test_uids) == 0:
            test_uids = {unique_series[n_train - 1]}
            train_uids.discard(unique_series[n_train - 1])
        
        train_df = metadata_df[metadata_df['series_uid'].isin(train_uids)]
        val_df = metadata_df[metadata_df['series_uid'].isin(val_uids)]
        test_df = metadata_df[metadata_df['series_uid'].isin(test_uids)]

    return train_df, val_df, test_df


def balance_samples(metadata_df):
    """Balance positive and negative samples."""
    positives = metadata_df[metadata_df['label'] == 1]
    negatives = metadata_df[metadata_df['label'] == 0]
    hard_negatives = negatives[negatives['is_hard_negative'] == True]
    easy_negatives = negatives[negatives['is_hard_negative'] == False]

    n_pos = len(positives)
    
    if n_pos == 0:
        # No positives to balance against — return all samples
        return metadata_df.sample(frac=1, random_state=RANDOM_SEED)
    
    n_hard = min(len(hard_negatives), n_pos * HARD_NEG_RATIO)
    n_easy = min(len(easy_negatives), n_pos * RANDOM_NEG_RATIO)

    sampled_hard = hard_negatives.sample(n=n_hard, random_state=RANDOM_SEED) if n_hard > 0 else hard_negatives
    sampled_easy = easy_negatives.sample(n=n_easy, random_state=RANDOM_SEED) if n_easy > 0 else easy_negatives

    balanced_df = pd.concat([positives, sampled_hard, sampled_easy], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED)

    return balanced_df


def main():
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Loading checkpoint.json...")
    with open(checkpoint_path, 'r') as f:
        checkpoint = json.load(f)

    all_results = checkpoint['all_results']
    print(f"  Total patches in checkpoint: {len(all_results)}")

    metadata_df = pd.DataFrame(all_results)
    print(f"  Unique scans: {metadata_df['series_uid'].nunique()}")
    print(f"  Positives: {(metadata_df['label'] == 1).sum()}")
    print(f"  Negatives: {(metadata_df['label'] == 0).sum()}")
    print(f"  Hard negatives: {metadata_df['is_hard_negative'].sum()}")

    # Get scan paths for split assignment
    scan_paths = get_scan_paths()

    # Create splits
    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = create_splits(metadata_df, scan_paths)

    # Balance training set
    print("Balancing training set...")
    train_balanced = balance_samples(train_df)

    print(f"\nSplit statistics:")
    print(f"  Train (balanced): {len(train_balanced)} samples")
    print(f"    Positives: {(train_balanced['label'] == 1).sum()}")
    print(f"    Negatives: {(train_balanced['label'] == 0).sum()}")
    print(f"  Validation: {len(val_df)} samples")
    print(f"    Positives: {(val_df['label'] == 1).sum()}")
    print(f"    Negatives: {(val_df['label'] == 0).sum()}")
    print(f"  Test: {len(test_df)} samples")
    print(f"    Positives: {(test_df['label'] == 1).sum()}")
    print(f"    Negatives: {(test_df['label'] == 0).sum()}")

    # Save metadata
    print("\nSaving metadata CSVs...")
    train_balanced.to_csv(metadata_dir / "train_samples.csv", index=False)
    val_df.to_csv(metadata_dir / "val_samples.csv", index=False)
    test_df.to_csv(metadata_dir / "test_samples.csv", index=False)
    metadata_df.to_csv(metadata_dir / "all_samples.csv", index=False)

    # Save statistics
    stats = {
        'total_patches': len(metadata_df),
        'total_positives': int((metadata_df['label'] == 1).sum()),
        'total_negatives': int((metadata_df['label'] == 0).sum()),
        'train_samples': len(train_balanced),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
    }
    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n✅ Metadata generation complete!")


if __name__ == "__main__":
    main()
