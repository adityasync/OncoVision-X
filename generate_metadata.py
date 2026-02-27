#!/usr/bin/env python3
"""
Generate metadata CSVs from preprocessed patches.

Reconstructs metadata from:
  1. checkpoint.json (if available), OR
  2. Existing all_samples.csv (if available), OR
  3. Patch files on disk + candidates.csv

Then creates balanced train/val/test split CSVs.
"""

import json
import os
import pandas as pd
import numpy as np
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_DIR = Path("preprocessed_data")
DATA_DIR = Path("data")
HARD_NEG_RATIO = 10   # Hard negatives per positive (increased from 5)
RANDOM_NEG_RATIO = 5  # Random negatives per positive (increased from 2)


def get_scan_paths():
    """Get all .mhd scan file paths organized by subset."""
    scan_paths = {}
    for i in range(5):
        subset_dir = DATA_DIR / f"subset{i}" / f"subset{i}"
        if subset_dir.exists():
            mhd_files = list(subset_dir.glob("*.mhd"))
            scan_paths[f"subset{i}"] = mhd_files
    return scan_paths


def rebuild_metadata_from_patches():
    """Reconstruct metadata by scanning patch files + candidates.csv."""
    print("  Rebuilding metadata from patch files on disk...")

    nodule_dir = OUTPUT_DIR / "nodule_patches"
    context_dir = OUTPUT_DIR / "context_patches"

    if not nodule_dir.exists():
        raise FileNotFoundError(f"Patch directory not found: {nodule_dir}")

    # Load candidates for label lookup (index -> label)
    candidates_df = pd.read_csv(DATA_DIR / "candidates.csv")
    candidate_label = {}
    candidate_series = {}
    for idx, row in candidates_df.iterrows():
        candidate_label[idx] = int(row['class'])
        candidate_series[idx] = row['seriesuid']

    # Scan patch files
    patch_files = sorted(nodule_dir.glob("*.npz"))
    print(f"  Found {len(patch_files)} patch files")

    results = []
    for pf in patch_files:
        patch_id = pf.stem

        # Parse: series_uid + candidate_idx from filename
        last_underscore = patch_id.rfind('_')
        if last_underscore == -1:
            continue

        series_uid = patch_id[:last_underscore]
        try:
            candidate_idx = int(patch_id[last_underscore + 1:])
        except ValueError:
            continue

        # Look up label
        if candidate_idx not in candidate_label:
            continue

        label = candidate_label[candidate_idx]

        # is_hard_negative: we can't recompute voxel distances without
        # loading scans, so set False. The balance function will just use
        # all negatives equally — this is fine for val/test.
        is_hard_negative = False

        nodule_path = str(nodule_dir / f"{patch_id}.npz")
        context_path = str(context_dir / f"{patch_id}.npz")

        results.append({
            'patch_id': patch_id,
            'series_uid': series_uid,
            'label': label,
            'is_hard_negative': is_hard_negative,
            'nodule_path': nodule_path,
            'context_path': context_path,
        })

    return pd.DataFrame(results)


def create_splits(metadata_df, scan_paths):
    """Create train/val/test splits based on subsets."""
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

    data_series = set(metadata_df['series_uid'].unique())
    val_match = data_series & val_series
    test_match = data_series & test_series

    if len(val_match) > 0 and len(test_match) > 0:
        train_df = metadata_df[metadata_df['series_uid'].isin(train_series)]
        val_df = metadata_df[metadata_df['series_uid'].isin(val_series)]
        test_df = metadata_df[metadata_df['series_uid'].isin(test_series)]
        print("  Using subset-based splitting")
    else:
        print("  Falling back to random patient-level split (60/20/20)...")
        unique_series = metadata_df['series_uid'].unique()
        np.random.shuffle(unique_series)

        n = len(unique_series)
        n_train = max(1, int(n * 0.6))
        n_val = max(1, int(n * 0.2))

        train_uids = set(unique_series[:n_train])
        val_uids = set(unique_series[n_train:n_train + n_val])
        test_uids = set(unique_series[n_train + n_val:])

        if len(test_uids) == 0:
            test_uids = {unique_series[n_train - 1]}
            train_uids.discard(unique_series[n_train - 1])

        train_df = metadata_df[metadata_df['series_uid'].isin(train_uids)]
        val_df = metadata_df[metadata_df['series_uid'].isin(val_uids)]
        test_df = metadata_df[metadata_df['series_uid'].isin(test_uids)]

    return train_df, val_df, test_df


def balance_samples(metadata_df):
    """Balance positive and negative samples.
    
    Target ratio: 1 positive : 7 negatives
    Prefers hard negatives (5x) + easy negatives (2x) when available.
    Falls back to random negatives if hard_negative info is unavailable.
    """
    positives = metadata_df[metadata_df['label'] == 1]
    negatives = metadata_df[metadata_df['label'] == 0]

    n_pos = len(positives)
    if n_pos == 0:
        return metadata_df.sample(frac=1, random_state=RANDOM_SEED)

    hard_negatives = negatives[negatives['is_hard_negative'] == True]
    easy_negatives = negatives[negatives['is_hard_negative'] == False]

    total_neg_wanted = n_pos * (HARD_NEG_RATIO + RANDOM_NEG_RATIO)  # 7x

    if len(hard_negatives) > 0:
        # Have hard negative info — use preferred split
        n_hard = min(len(hard_negatives), n_pos * HARD_NEG_RATIO)
        n_easy = min(len(easy_negatives), n_pos * RANDOM_NEG_RATIO)
        sampled_hard = hard_negatives.sample(n=n_hard, random_state=RANDOM_SEED)
        sampled_easy = easy_negatives.sample(n=n_easy, random_state=RANDOM_SEED) if n_easy > 0 else easy_negatives.iloc[:0]
        sampled_neg = pd.concat([sampled_hard, sampled_easy])
    else:
        # No hard negative info — just sample 7x negatives randomly
        n_neg = min(len(negatives), total_neg_wanted)
        sampled_neg = negatives.sample(n=n_neg, random_state=RANDOM_SEED)

    balanced_df = pd.concat([positives, sampled_neg], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED)

    return balanced_df


def main():
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    all_samples_path = OUTPUT_DIR / "metadata" / "all_samples.csv"
    metadata_dir = OUTPUT_DIR / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Try sources in order: checkpoint > all_samples.csv > rebuild from patches
    if checkpoint_path.exists():
        print("Loading from checkpoint.json...")
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        metadata_df = pd.DataFrame(checkpoint['all_results'])
    elif all_samples_path.exists():
        print("Loading from existing all_samples.csv...")
        metadata_df = pd.read_csv(all_samples_path)
    else:
        print("No checkpoint or all_samples.csv found.")
        metadata_df = rebuild_metadata_from_patches()

    print(f"  Total patches: {len(metadata_df)}")
    print(f"  Unique scans: {metadata_df['series_uid'].nunique()}")
    print(f"  Positives: {(metadata_df['label'] == 1).sum()}")
    print(f"  Negatives: {(metadata_df['label'] == 0).sum()}")
    print(f"  Hard negatives: {metadata_df['is_hard_negative'].sum()}")

    scan_paths = get_scan_paths()

    print("\nCreating train/val/test splits...")
    train_df, val_df, test_df = create_splits(metadata_df, scan_paths)

    # Balance ALL splits
    print("Balancing training set...")
    train_balanced = balance_samples(train_df)
    print("Balancing validation set...")
    val_balanced = balance_samples(val_df)
    print("Balancing test set...")
    test_balanced = balance_samples(test_df)

    def _print_split(name, df):
        n_pos = (df['label'] == 1).sum()
        n_neg = (df['label'] == 0).sum()
        ratio = n_neg / max(n_pos, 1)
        print(f"  {name}: {len(df)} samples")
        print(f"    Positives: {n_pos}")
        print(f"    Negatives: {n_neg}")
        print(f"    Ratio: 1:{ratio:.1f}")

    print(f"\nSplit statistics:")
    _print_split("Train (balanced)", train_balanced)
    _print_split("Validation (balanced)", val_balanced)
    _print_split("Test (balanced)", test_balanced)

    # Save
    print("\nSaving metadata CSVs...")
    train_balanced.to_csv(metadata_dir / "train_samples.csv", index=False)
    val_balanced.to_csv(metadata_dir / "val_samples.csv", index=False)
    test_balanced.to_csv(metadata_dir / "test_samples.csv", index=False)
    metadata_df.to_csv(metadata_dir / "all_samples.csv", index=False)

    stats = {
        'total_patches': len(metadata_df),
        'total_positives': int((metadata_df['label'] == 1).sum()),
        'total_negatives': int((metadata_df['label'] == 0).sum()),
        'train_samples': len(train_balanced),
        'val_samples': len(val_balanced),
        'test_samples': len(test_balanced),
    }
    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)

    print("\nMetadata generation complete!")


if __name__ == "__main__":
    main()
