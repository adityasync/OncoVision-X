#!/usr/bin/env python3
"""
LUNA16 Preprocessing Pipeline
Phase 1, Week 2: Patch Extraction and Preprocessing

This script implements memory-efficient preprocessing for the DCA-Net model:
1. Nodule patch extraction (64x64x64)
2. Context patch extraction (96x96x96 → 48x48x48)
3. HU windowing and normalization
4. Data balancing strategy
5. Train/Val/Test split

Hardware-aware: Processes one scan at a time for 8GB RAM constraint.
"""

import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm
import json
import logging
from datetime import datetime
from scipy.ndimage import zoom
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import psutil
import gc
import signal
import sys

# ============== Configuration ==============
DATA_DIR = Path("data")
OUTPUT_DIR = Path("preprocessed_data")
NODULE_PATCH_SIZE = 64  # 64x64x64 cube
CONTEXT_PATCH_SIZE = 96  # Extract 96x96x96, then downsample
CONTEXT_DOWNSAMPLE = 48  # Downsample to 48x48x48

# HU windowing for lung tissue
HU_MIN = -1000
HU_MAX = 400

# Data balancing: 1 positive : 7 negatives (from roadmap)
HARD_NEG_RATIO = 3  # Hard negatives per positive (reduced from 5 — more generalizable)
RANDOM_NEG_RATIO = 4  # Random negatives per positive (increased from 2 — better diversity)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Parallel processing configuration
# Use 32 of available CPU cores (leave some for system)
NUM_WORKERS = min(32, max(1, multiprocessing.cpu_count() - 4))
BATCH_SIZE = 64  # Process in batches for efficient checkpointing


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
║   AI-Powered Lung Cancer Detection — Data Preprocessing            ║
║                                                                    ║
╚════════════════════════════════════════════════════════════════════╝{RESET}
""")


# ─────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────
logger = logging.getLogger('preprocessing')

def setup_logging():
    """Setup dual logging: console (colored) + file (clean text) in logs/ folder."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f"preprocessing_{timestamp}.log"
    
    logger.setLevel(logging.DEBUG)
    
    # File handler — clean text (no ANSI colors)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
    logger.addHandler(fh)
    
    # Console handler — minimal (print() handles colored console output)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.WARNING)  # Only warnings+ to avoid double-printing
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    logger.info(f"Preprocessing log started: {log_file}")
    print(f"  {DIM}Log file:{RESET} {log_file}")
    
    return log_file


def _strip_ansi(text):
    """Remove ANSI escape codes for clean log file output."""
    import re
    return re.sub(r'\033\[[0-9;]*m', '', text)


def section(title):
    print(f"\n{BOLD}{BLUE}{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}{RESET}")
    logger.info(f"{'─' * 60}")
    logger.info(f"  {title}")
    logger.info(f"{'─' * 60}")


def info(label, value):
    print(f"  {DIM}{label}:{RESET} {value}")
    logger.info(f"  {label}: {_strip_ansi(str(value))}")


def success(msg):
    print(f"  {GREEN}✓ {msg}{RESET}")
    logger.info(f"  ✓ {msg}")


def warn(msg):
    print(f"  {YELLOW}! {msg}{RESET}")
    logger.warning(f"  ! {msg}")


def get_scan_paths():
    """Get all .mhd scan file paths organized by subset (supports 0-9)."""
    scan_paths = {}
    for i in range(10):  # Changed from range(5) to support full LUNA16 (0-9)
        subset_dir = DATA_DIR / f"subset{i}" / f"subset{i}"
        if subset_dir.exists():
            mhd_files = list(subset_dir.glob("*.mhd"))
            if len(mhd_files) > 0:  # Only add if scans found
                scan_paths[f"subset{i}"] = mhd_files
    return scan_paths


def load_scan(mhd_path):
    """Load a CT scan from .mhd file."""
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    
    # Get metadata for coordinate conversion
    origin = np.array(img.GetOrigin())  # (x, y, z)
    spacing = np.array(img.GetSpacing())  # (x, y, z)
    direction = np.array(img.GetDirection()).reshape(3, 3)
    
    return arr, origin, spacing, direction


def world_to_voxel(world_coord, origin, spacing, direction=None):
    """Convert world coordinates (mm) to voxel indices."""
    # Simple conversion (assuming identity direction matrix)
    voxel_coord = (world_coord - origin) / spacing
    return np.round(voxel_coord).astype(int)


def apply_hu_windowing(arr, hu_min=HU_MIN, hu_max=HU_MAX):
    """Apply HU windowing and normalize to [-1, 1]."""
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)  # [0, 1]
    arr = arr * 2 - 1  # [-1, 1]
    return arr.astype(np.float32)


def extract_patch(arr, center, patch_size):
    """Extract a cubic patch centered at the given voxel coordinate.
    
    Args:
        arr: 3D numpy array (z, y, x)
        center: (x, y, z) voxel coordinates
        patch_size: Size of the cubic patch
    
    Returns:
        patch: Extracted patch or None if out of bounds
    """
    half = patch_size // 2
    cx, cy, cz = center
    
    # Compute bounds (note: arr is z, y, x)
    z_start, z_end = cz - half, cz + half
    y_start, y_end = cy - half, cy + half
    x_start, x_end = cx - half, cx + half
    
    # Check bounds
    if (z_start < 0 or z_end > arr.shape[0] or
        y_start < 0 or y_end > arr.shape[1] or
        x_start < 0 or x_end > arr.shape[2]):
        return None
    
    patch = arr[z_start:z_end, y_start:y_end, x_start:x_end]
    assert patch.shape == (patch_size, patch_size, patch_size), f"Invalid patch shape: {patch.shape}"
    return patch


def downsample_patch(patch, target_size):
    """Downsample a 3D patch using scipy zoom."""
    if patch is None:
        return None
    zoom_factor = target_size / patch.shape[0]
    return zoom(patch, zoom_factor, order=1)  # Linear interpolation


def process_scan(mhd_path, annotations_df, candidates_df, output_dir):
    """Process a single CT scan and extract patches for all candidates.
    
    Args:
        mhd_path: Path to .mhd file
        annotations_df: DataFrame with ground truth nodules
        candidates_df: DataFrame with all candidates
        output_dir: Directory to save patches
    
    Returns:
        results: List of dicts with patch metadata
        stats: Dict with extraction statistics
    """
    series_uid = mhd_path.stem
    
    # Get candidates for this scan
    scan_candidates = candidates_df[candidates_df['seriesuid'] == series_uid]
    
    if len(scan_candidates) == 0:
        return [], {'total': 0, 'success': 0, 'failures': {
            'nodule_out_of_bounds': 0, 'context_out_of_bounds': 0, 'load_error': 0
        }}
    
    # Initialize statistics
    stats = {
        'total': len(scan_candidates),
        'success': 0,
        'failures': {
            'nodule_out_of_bounds': 0,
            'context_out_of_bounds': 0,
            'load_error': 0
        }
    }
    
    # Load the scan
    try:
        arr, origin, spacing, direction = load_scan(mhd_path)
    except Exception as e:
        warn(f"Error loading {mhd_path}: {e}")
        stats['failures']['load_error'] = len(scan_candidates)
        return [], stats
    
    # Apply HU windowing
    arr_windowed = apply_hu_windowing(arr)
    
    # Get annotations for this scan (for hard negative identification)
    scan_annotations = annotations_df[annotations_df['seriesuid'] == series_uid]
    nodule_positions = []
    for _, row in scan_annotations.iterrows():
        world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        nodule_positions.append({
            'voxel': voxel_coord,
            'diameter_mm': row['diameter_mm']
        })
    
    results = []
    
    for idx, row in scan_candidates.iterrows():
        world_coord = np.array([row['coordX'], row['coordY'], row['coordZ']])
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        label = row['class']
        
        # Extract nodule patch (64x64x64)
        nodule_patch = extract_patch(arr_windowed, voxel_coord, NODULE_PATCH_SIZE)
        if nodule_patch is None:
            stats['failures']['nodule_out_of_bounds'] += 1
            continue
        
        # Extract context patch (96x96x96 → 48x48x48)
        # FIX: Skip sample if context is out of bounds — don't create fake zero data
        context_patch = extract_patch(arr_windowed, voxel_coord, CONTEXT_PATCH_SIZE)
        if context_patch is None:
            stats['failures']['context_out_of_bounds'] += 1
            continue  # Don't create fake data with zero-padding
        
        context_patch = downsample_patch(context_patch, CONTEXT_DOWNSAMPLE)
        
        # Determine if this is a hard negative (close to a real nodule)
        # FIX: Use 1.5× diameter with average spacing (research-backed)
        is_hard_negative = False
        if label == 0 and len(nodule_positions) > 0:
            avg_spacing = np.mean(spacing)  # Average across x,y,z
            for nodule in nodule_positions:
                dist = np.linalg.norm(voxel_coord - nodule['voxel'])
                # Consider "hard" if within 2.0x nodule diameter (in voxels) — research-backed
                threshold_voxels = (nodule['diameter_mm'] * 2.0) / avg_spacing
                if dist < threshold_voxels:
                    is_hard_negative = True
                    break
        
        # Save patches
        patch_id = f"{series_uid}_{idx}"
        nodule_path = output_dir / "nodule_patches" / f"{patch_id}.npz"
        context_path = output_dir / "context_patches" / f"{patch_id}.npz"
        
        np.savez_compressed(nodule_path, patch=nodule_patch)
        np.savez_compressed(context_path, patch=context_patch)
        
        stats['success'] += 1
        results.append({
            'patch_id': patch_id,
            'series_uid': series_uid,
            'label': label,
            'is_hard_negative': is_hard_negative,
            'voxel_x': int(voxel_coord[0]),
            'voxel_y': int(voxel_coord[1]),
            'voxel_z': int(voxel_coord[2]),
            'nodule_path': str(nodule_path),
            'context_path': str(context_path)
        })
    
    # Log statistics if significant failures
    total_failures = stats['failures']['nodule_out_of_bounds'] + stats['failures']['context_out_of_bounds']
    if total_failures > 0:
        failure_rate = total_failures / stats['total'] * 100
        if failure_rate > 10:  # Warn if >10% failures
            warn(f"{series_uid}: {failure_rate:.1f}% extraction failures ({total_failures}/{stats['total']})")
    
    return results, stats


def balance_samples(metadata_df, pos_to_neg_ratio=7):
    """Balance positive and negative samples according to strategy.
    
    Strategy from roadmap:
    - Keep all positive samples
    - Hard negatives: 5x positives
    - Random negatives: 2x positives
    - Total ratio: 1:7
    """
    positives = metadata_df[metadata_df['label'] == 1]
    negatives = metadata_df[metadata_df['label'] == 0]
    hard_negatives = negatives[negatives['is_hard_negative'] == True]
    easy_negatives = negatives[negatives['is_hard_negative'] == False]
    
    n_pos = len(positives)
    n_hard = min(len(hard_negatives), n_pos * HARD_NEG_RATIO)
    n_easy = min(len(easy_negatives), n_pos * RANDOM_NEG_RATIO)
    
    # Sample negatives
    sampled_hard = hard_negatives.sample(n=n_hard, random_state=RANDOM_SEED) if n_hard > 0 else hard_negatives
    sampled_easy = easy_negatives.sample(n=n_easy, random_state=RANDOM_SEED) if n_easy > 0 else easy_negatives
    
    # Combine
    balanced_df = pd.concat([positives, sampled_hard, sampled_easy], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_SEED)  # Shuffle
    
    return balanced_df


def create_splits(metadata_df, scan_paths):
    """Create train/val/test splits based on available subsets.
    
    Split strategy (adaptive):
    - 10 subsets (full LUNA16):  Train subset0-7 (80%), Val subset8 (10%), Test subset9 (10%)
    - 5 subsets (partial):       Train subset0-2 (60%), Val subset3 (20%), Test subset4 (20%)
    - Other counts:             Train 60%, Val 20%, Test 20% by index
    """
    train_series = set()
    val_series = set()
    test_series = set()
    
    available_subsets = sorted(scan_paths.keys())  # e.g. ['subset0', 'subset1', ...]
    num_subsets = len(available_subsets)
    
    for subset_name, paths in scan_paths.items():
        for path in paths:
            series_uid = path.stem
            
            if num_subsets >= 10:
                # Full LUNA16 (10 subsets) — standard split
                if subset_name in ['subset0', 'subset1', 'subset2', 'subset3',
                                   'subset4', 'subset5', 'subset6', 'subset7']:
                    train_series.add(series_uid)
                elif subset_name == 'subset8':
                    val_series.add(series_uid)
                elif subset_name == 'subset9':
                    test_series.add(series_uid)
            elif num_subsets == 5:
                # Partial LUNA16 (5 subsets)
                if subset_name in ['subset0', 'subset1', 'subset2']:
                    train_series.add(series_uid)
                elif subset_name == 'subset3':
                    val_series.add(series_uid)
                elif subset_name == 'subset4':
                    test_series.add(series_uid)
            else:
                # Generic: 60/20/20 by index
                subset_idx = int(subset_name.replace('subset', ''))
                train_cutoff = int(num_subsets * 0.6)
                val_cutoff = int(num_subsets * 0.8)
                if subset_idx < train_cutoff:
                    train_series.add(series_uid)
                elif subset_idx < val_cutoff:
                    val_series.add(series_uid)
                else:
                    test_series.add(series_uid)
    
    train_df = metadata_df[metadata_df['series_uid'].isin(train_series)]
    val_df = metadata_df[metadata_df['series_uid'].isin(val_series)]
    test_df = metadata_df[metadata_df['series_uid'].isin(test_series)]
    
    return train_df, val_df, test_df


def load_checkpoint(checkpoint_path):
    """Load checkpoint file to resume processing."""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return {'processed_scans': [], 'all_results': []}


def save_checkpoint(checkpoint_path, processed_scans, all_results):
    """Save progress checkpoint."""
    checkpoint = {
        'processed_scans': list(processed_scans),
        'all_results': all_results
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully."""
    global shutdown_requested
    print(f"\n\n  {BOLD}{RED}⚠️  Shutdown requested. Saving checkpoint and exiting...{RESET}")
    shutdown_requested = True


def check_memory():
    """Return True if at least 10 GB of RAM is available."""
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    return available_gb >= 10


def process_scan_wrapper(args):
    """Top-level wrapper so ProcessPoolExecutor can pickle the call."""
    mhd_path, annotations_df, candidates_df, output_dir = args
    try:
        results, stats = process_scan(mhd_path, annotations_df, candidates_df, output_dir)
        return (mhd_path.stem, results, stats, None)
    except Exception as e:
        return (mhd_path.stem, [], {
            'total': 0,
            'success': 0,
            'failures': {
                'nodule_out_of_bounds': 0,
                'context_out_of_bounds': 0,
                'load_error': 1
            }
        }, str(e))


def main():
    """Main preprocessing pipeline with checkpoint/resume support."""
    global shutdown_requested
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    banner()
    log_file = setup_logging()
    print(f"  {DIM}Press Ctrl+C to safely stop and save progress{RESET}\n")
    
    # Create output directories
    (OUTPUT_DIR / "nodule_patches").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "context_patches").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = OUTPUT_DIR / "checkpoint.json"
    
    # Load checkpoint if exists
    checkpoint = load_checkpoint(checkpoint_path)
    processed_scans = set(checkpoint['processed_scans'])
    all_results = checkpoint['all_results']
    
    if processed_scans:
        info("Status", f"Resumed from {len(processed_scans)} scans processed")
        info("Status", f"{len(all_results)} patches already extracted\n")
    
    # Load annotations and candidates
    section("DATA PREPARATION")
    info("Status", "Loading annotations and candidates...")
    annotations_df = pd.read_csv(DATA_DIR / "annotations.csv")
    candidates_df = pd.read_csv(DATA_DIR / "candidates.csv")
    
    info("Annotations", f"{len(annotations_df)} nodules")
    info("Candidates", f"{len(candidates_df)} total")
    
    # Get scan paths
    scan_paths = get_scan_paths()
    total_scans = sum(len(paths) for paths in scan_paths.values())
    remaining = total_scans - len(processed_scans)
    info("Total scans", f"{total_scans} | Remaining: {remaining}")
    
    # Process each scan
    section("PATCH EXTRACTION")
    scans_processed_this_session = 0
    
    # Aggregate extraction statistics
    aggregate_stats = {
        'total_candidates': 0,
        'total_success': 0,
        'total_nodule_oob': 0,
        'total_context_oob': 0,
        'total_load_error': 0
    }
    
    # Check memory and auto-scale workers
    if not check_memory():
        warn(f"Low memory detected — reducing workers from {NUM_WORKERS} to {max(8, NUM_WORKERS // 2)}")
        n_workers = max(8, NUM_WORKERS // 2)
    else:
        n_workers = NUM_WORKERS
    info("Parallel workers", n_workers)

    for subset_name, paths in scan_paths.items():
        if shutdown_requested:
            break

        print(f"\nProcessing {subset_name}...")

        # Filter out already processed scans
        remaining_paths = [p for p in paths if p.stem not in processed_scans]
        skipped = len(paths) - len(remaining_paths)
        if skipped > 0:
            print(f"  (Skipping {skipped} already processed scans)")

        if len(remaining_paths) == 0:
            print(f"  All scans in {subset_name} already processed")
            continue

        # Build task args for parallel execution
        task_args = [(p, annotations_df, candidates_df, OUTPUT_DIR) for p in remaining_paths]
        print(f"  Processing {len(remaining_paths)} scans with {n_workers} CPU cores...")

        batch_results = []
        batch_stats = []
        completed = 0

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {executor.submit(process_scan_wrapper, a): a[0] for a in task_args}

            with tqdm(total=len(futures), desc=subset_name,
                      bar_format='{l_bar}%s{bar}%s{r_bar}' % (GREEN, RESET)) as pbar:

                for future in as_completed(futures):
                    if shutdown_requested:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        series_uid, results, scan_stats, error = future.result()
                        if error:
                            warn(f"Error processing {series_uid}: {error}")

                        batch_results.extend(results)
                        batch_stats.append(scan_stats)
                        processed_scans.add(series_uid)
                        scans_processed_this_session += 1
                        completed += 1

                        # Checkpoint every ~1,000 patches in the batch
                        if len(batch_results) >= 1000:
                            all_results.extend(batch_results)
                            for s in batch_stats:
                                aggregate_stats['total_candidates'] += s['total']
                                aggregate_stats['total_success'] += s['success']
                                aggregate_stats['total_nodule_oob'] += s['failures']['nodule_out_of_bounds']
                                aggregate_stats['total_context_oob'] += s['failures']['context_out_of_bounds']
                                aggregate_stats['total_load_error'] += s['failures']['load_error']
                            save_checkpoint(checkpoint_path, processed_scans, all_results)
                            batch_results = []
                            batch_stats = []
                            gc.collect()

                    except Exception as e:
                        warn(f"Error getting result: {e}")
                    finally:
                        pbar.update(1)

        # Flush remaining batch results
        if batch_results:
            all_results.extend(batch_results)
            for s in batch_stats:
                aggregate_stats['total_candidates'] += s['total']
                aggregate_stats['total_success'] += s['success']
                aggregate_stats['total_nodule_oob'] += s['failures']['nodule_out_of_bounds']
                aggregate_stats['total_context_oob'] += s['failures']['context_out_of_bounds']
                aggregate_stats['total_load_error'] += s['failures']['load_error']

        gc.collect()
        print(f"  Completed {completed}/{len(remaining_paths)} scans in {subset_name}")
    
    # Final checkpoint save
    save_checkpoint(checkpoint_path, processed_scans, all_results)
    
    # Print extraction statistics
    section("EXTRACTION STATISTICS")
    info("Total candidates", f"{aggregate_stats['total_candidates']:,}")
    info("Successfully extracted", f"{aggregate_stats['total_success']:,}")
    info("Nodule out-of-bounds", f"{aggregate_stats['total_nodule_oob']:,}")
    info("Context out-of-bounds", f"{aggregate_stats['total_context_oob']:,}")
    info("Load errors", f"{aggregate_stats['total_load_error']:,}")
    if aggregate_stats['total_candidates'] > 0:
        total_rejected = (aggregate_stats['total_candidates'] - aggregate_stats['total_success'])
        rejection_rate = total_rejected / aggregate_stats['total_candidates'] * 100
        info("Rejection rate", f"{rejection_rate:.1f}%")
    
    if shutdown_requested:
        section("PREPROCESSING PAUSED")
        info("Progress", f"{len(processed_scans)}/{total_scans} scans processed")
        info("Patches extracted", f"{len(all_results)}")
        print(f"\n  {DIM}Run the script again to resume from checkpoint{RESET}")
        return
    
    # All scans processed - create final metadata
    section("METADATA GENERATION")
    
    metadata_df = pd.DataFrame(all_results)
    info("Total patches", f"{len(metadata_df)}")
    info("Positives", f"{(metadata_df['label'] == 1).sum()}")
    info("Negatives", f"{(metadata_df['label'] == 0).sum()}")
    info("Hard negatives", f"{metadata_df['is_hard_negative'].sum()}")
    
    # Create splits
    section("DATA SPLITS")
    info("Status", "Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(metadata_df, scan_paths)
    
    # FIX: Only balance TRAINING set — val/test must maintain natural distribution
    print(f"\n{BOLD}{BLUE}DATA BALANCING{RESET}")
    info("Strategy", "Balancing TRAINING ONLY (1:7 ratio)")
    info("Val/Test", "Keeping NATURAL distribution (~1:1350)")
    
    train_balanced = balance_samples(train_df, pos_to_neg_ratio=7)
    val_final = val_df    # NO balancing - natural distribution
    test_final = test_df  # NO balancing - natural distribution
    
    def _print_split(name, df, is_balanced=False):
        n_pos = (df['label'] == 1).sum()
        n_neg = (df['label'] == 0).sum()
        ratio = n_neg / max(n_pos, 1)
        status = "(balanced 1:7)" if is_balanced else "(natural ~1:1350)"
        
        print(f"\n  {BOLD}{name} {status}:{RESET} {len(df)} samples")
        print(f"    {GREEN}Positives:{RESET} {n_pos:,}")
        print(f"    {RED}Negatives:{RESET} {n_neg:,}")
        print(f"    Ratio: 1:{ratio:.0f}")
    
    print(f"\n  {BOLD}{BLUE}FINAL SPLIT STATISTICS{RESET}")
    _print_split("Train", train_balanced, is_balanced=True)
    _print_split("Validation", val_final, is_balanced=False)
    _print_split("Test", test_final, is_balanced=False)
    
    # Save metadata
    section("SAVING")
    info("Status", "Saving metadata CSVs...")
    train_balanced.to_csv(OUTPUT_DIR / "metadata" / "train_samples.csv", index=False)
    val_final.to_csv(OUTPUT_DIR / "metadata" / "val_samples.csv", index=False)
    test_final.to_csv(OUTPUT_DIR / "metadata" / "test_samples.csv", index=False)
    metadata_df.to_csv(OUTPUT_DIR / "metadata" / "all_samples.csv", index=False)
    
    # Save statistics
    stats = {
        'total_patches': len(metadata_df),
        'total_positives': int((metadata_df['label'] == 1).sum()),
        'total_negatives': int((metadata_df['label'] == 0).sum()),
        'train_samples': len(train_balanced),
        'val_samples': len(val_final),
        'test_samples': len(test_final),
        'extraction_stats': aggregate_stats,
        'patch_sizes': {
            'nodule': NODULE_PATCH_SIZE,
            'context': CONTEXT_DOWNSAMPLE
        },
        'hu_window': {
            'min': HU_MIN,
            'max': HU_MAX
        }
    }
    
    with open(OUTPUT_DIR / "statistics.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Remove checkpoint file on successful completion
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    
    section("PREPROCESSING COMPLETE")
    print(f"\n  {DIM}Output directory:{RESET} {OUTPUT_DIR}/")
    print(f"    nodule_patches/:  {len(metadata_df)} .npz files")
    print(f"    context_patches/: {len(metadata_df)} .npz files")
    print(f"    metadata/:        train_samples.csv, val_samples.csv, test_samples.csv")
    print(f"    statistics.json")


if __name__ == "__main__":
    main()

