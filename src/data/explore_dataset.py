#!/usr/bin/env python3
"""
LUNA16 Dataset Exploration Script
Phase 1, Week 1: Data Understanding

This script analyzes the LUNA16 dataset structure and generates statistics.
"""

import os
import pandas as pd
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from collections import defaultdict

# Configure paths
DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")
FIGURES_DIR = REPORTS_DIR / "figures"

# Create output directories
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_annotations():
    """Load and analyze annotations.csv (ground truth nodules)."""
    annotations_path = DATA_DIR / "annotations.csv"
    df = pd.read_csv(annotations_path)
    print("\n" + "="*60)
    print("ANNOTATIONS.CSV ANALYSIS (Ground Truth Nodules)")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    print(f"\nUnique series (scans): {df['seriesuid'].nunique()}")
    print(f"Total nodules: {len(df)}")
    
    return df


def load_candidates():
    """Load and analyze candidates.csv (all candidate nodules)."""
    candidates_path = DATA_DIR / "candidates.csv"
    df = pd.read_csv(candidates_path)
    print("\n" + "="*60)
    print("CANDIDATES.CSV ANALYSIS (All Candidates)")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:\n{df.head()}")
    print(f"\nClass distribution:")
    print(df['class'].value_counts())
    print(f"\nPositive/Negative ratio: 1:{len(df[df['class']==0]) / max(len(df[df['class']==1]), 1):.0f}")
    print(f"\nUnique series (scans): {df['seriesuid'].nunique()}")
    
    return df


def analyze_class_imbalance(candidates_df):
    """Analyze and visualize class imbalance."""
    print("\n" + "="*60)
    print("CLASS IMBALANCE ANALYSIS")
    print("="*60)
    
    pos = len(candidates_df[candidates_df['class'] == 1])
    neg = len(candidates_df[candidates_df['class'] == 0])
    
    print(f"\nPositive candidates (true nodules): {pos}")
    print(f"Negative candidates (false positives): {neg}")
    print(f"Total candidates: {pos + neg}")
    print(f"\nClass imbalance ratio: 1:{neg/pos:.0f}")
    print(f"Positive percentage: {100*pos/(pos+neg):.3f}%")
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(['Positive\n(True Nodules)', 'Negative\n(Non-nodules)'], 
                  [pos, neg], color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('LUNA16 Class Distribution\n(Severe Imbalance: ~1:1350)')
    ax.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, [pos, neg]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{val:,}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'class_imbalance.png', dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'class_imbalance.png'}")
    
    return {'positive': pos, 'negative': neg, 'ratio': neg/pos}


def analyze_nodule_sizes(annotations_df):
    """Analyze nodule size distribution."""
    print("\n" + "="*60)
    print("NODULE SIZE DISTRIBUTION")
    print("="*60)
    
    diameters = annotations_df['diameter_mm']
    
    print(f"\nDiameter statistics (mm):")
    print(f"  Min: {diameters.min():.2f}")
    print(f"  Max: {diameters.max():.2f}")
    print(f"  Mean: {diameters.mean():.2f}")
    print(f"  Median: {diameters.median():.2f}")
    print(f"  Std: {diameters.std():.2f}")
    
    # Size categories as per roadmap
    tiny = len(diameters[diameters < 4])
    small = len(diameters[(diameters >= 4) & (diameters < 6)])
    medium = len(diameters[(diameters >= 6) & (diameters < 10)])
    large = len(diameters[diameters >= 10])
    
    print(f"\nSize categories:")
    print(f"  Tiny (<4mm): {tiny} ({100*tiny/len(diameters):.1f}%)")
    print(f"  Small (4-6mm): {small} ({100*small/len(diameters):.1f}%)")
    print(f"  Medium (6-10mm): {medium} ({100*medium/len(diameters):.1f}%)")
    print(f"  Large (>10mm): {large} ({100*large/len(diameters):.1f}%)")
    
    # Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    axes[0].hist(diameters, bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(4, color='red', linestyle='--', label='Easy/Hard boundary (4mm)')
    axes[0].axvline(8, color='orange', linestyle='--', label='Medium/Large boundary (8mm)')
    axes[0].set_xlabel('Diameter (mm)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Nodule Diameter Distribution')
    axes[0].legend()
    
    # Category pie chart
    sizes = [tiny, small, medium, large]
    labels = [f'Tiny\n(<4mm)\n{tiny}', f'Small\n(4-6mm)\n{small}', 
              f'Medium\n(6-10mm)\n{medium}', f'Large\n(>10mm)\n{large}']
    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
    axes[1].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Nodule Size Categories\n(For Curriculum Learning)')
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'nodule_sizes.png', dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'nodule_sizes.png'}")
    
    return {
        'tiny': tiny, 'small': small, 'medium': medium, 'large': large,
        'min_mm': float(diameters.min()), 'max_mm': float(diameters.max()),
        'mean_mm': float(diameters.mean()), 'median_mm': float(diameters.median())
    }


def analyze_subsets():
    """Analyze the subset structure and scan counts."""
    print("\n" + "="*60)
    print("SUBSET ANALYSIS")
    print("="*60)
    
    subset_stats = {}
    total_scans = 0
    
    for i in range(5):  # subset0 to subset4
        subset_dir = DATA_DIR / f"subset{i}"
        # Handle nested structure (subset0/subset0/)
        nested_dir = subset_dir / f"subset{i}"
        
        search_dir = nested_dir if nested_dir.exists() else subset_dir
        
        if search_dir.exists():
            # Count .mhd files (each represents a CT scan)
            mhd_files = list(search_dir.glob("*.mhd"))
            subset_stats[f"subset{i}"] = len(mhd_files)
            total_scans += len(mhd_files)
            print(f"  subset{i}: {len(mhd_files)} scans")
    
    print(f"\nTotal scans across all subsets: {total_scans}")
    
    if total_scans == 0:
        print("  WARNING: No scans found!")
        return subset_stats
    
    # Train/Val/Test split as per roadmap
    train_scans = subset_stats.get('subset0', 0) + subset_stats.get('subset1', 0) + subset_stats.get('subset2', 0)
    val_scans = subset_stats.get('subset3', 0)
    test_scans = subset_stats.get('subset4', 0)
    
    print(f"\nRecommended Split (per roadmap):")
    print(f"  Train (subset0-2): {train_scans} scans ({100*train_scans/total_scans:.0f}%)")
    print(f"  Validation (subset3): {val_scans} scans ({100*val_scans/total_scans:.0f}%)")
    print(f"  Test (subset4): {test_scans} scans ({100*test_scans/total_scans:.0f}%)")
    
    return subset_stats


def analyze_sample_scan():
    """Load and analyze a sample CT scan to understand the data format."""
    print("\n" + "="*60)
    print("SAMPLE SCAN ANALYSIS")
    print("="*60)
    
    # Find a sample .mhd file (handle nested directories)
    sample_mhd = None
    for i in range(5):
        subset_dir = DATA_DIR / f"subset{i}"
        nested_dir = subset_dir / f"subset{i}"
        search_dir = nested_dir if nested_dir.exists() else subset_dir
        
        if search_dir.exists():
            mhd_files = list(search_dir.glob("*.mhd"))
            if mhd_files:
                sample_mhd = mhd_files[0]
                break
    
    if sample_mhd is None:
        print("No .mhd files found!")
        return None
    
    print(f"\nLoading sample scan: {sample_mhd.name}")
    
    # Load with SimpleITK
    img = sitk.ReadImage(str(sample_mhd))
    arr = sitk.GetArrayFromImage(img)
    
    print(f"\nImage Properties:")
    print(f"  Size (SimpleITK): {img.GetSize()}")  # (x, y, z)
    print(f"  Array shape (numpy): {arr.shape}")   # (z, y, x)
    print(f"  Spacing (mm): {img.GetSpacing()}")
    print(f"  Origin: {img.GetOrigin()}")
    print(f"  Direction: {img.GetDirection()}")
    
    print(f"\nHU Value Statistics:")
    print(f"  Min HU: {arr.min()}")
    print(f"  Max HU: {arr.max()}")
    print(f"  Mean HU: {arr.mean():.2f}")
    
    print(f"\nMemory Usage:")
    print(f"  Data type: {arr.dtype}")
    print(f"  Size in MB: {arr.nbytes / (1024**2):.2f}")
    
    # Visualize middle slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (z-axis)
    z_mid = arr.shape[0] // 2
    axes[0].imshow(arr[z_mid], cmap='gray', vmin=-1000, vmax=400)
    axes[0].set_title(f'Axial View (Slice {z_mid}/{arr.shape[0]})')
    axes[0].axis('off')
    
    # Coronal (y-axis)
    y_mid = arr.shape[1] // 2
    axes[1].imshow(arr[:, y_mid, :], cmap='gray', vmin=-1000, vmax=400)
    axes[1].set_title(f'Coronal View (Slice {y_mid}/{arr.shape[1]})')
    axes[1].axis('off')
    
    # Sagittal (x-axis)
    x_mid = arr.shape[2] // 2
    axes[2].imshow(arr[:, :, x_mid], cmap='gray', vmin=-1000, vmax=400)
    axes[2].set_title(f'Sagittal View (Slice {x_mid}/{arr.shape[2]})')
    axes[2].axis('off')
    
    plt.suptitle(f'Sample CT Scan: {sample_mhd.stem}\n(HU windowed: -1000 to 400)')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sample_scan_views.png', dpi=150)
    plt.close()
    print(f"\nSaved: {FIGURES_DIR / 'sample_scan_views.png'}")
    
    return {
        'shape': list(arr.shape),
        'spacing': list(img.GetSpacing()),
        'origin': list(img.GetOrigin()),
        'hu_min': int(arr.min()),
        'hu_max': int(arr.max()),
        'size_mb': float(arr.nbytes / (1024**2))
    }


def analyze_lung_segmentations():
    """Analyze the pre-computed lung segmentation masks."""
    print("\n" + "="*60)
    print("LUNG SEGMENTATION MASKS")
    print("="*60)
    
    seg_dir = DATA_DIR / "seg-lungs-LUNA16"
    
    if not seg_dir.exists():
        print(f"Segmentation directory not found: {seg_dir}")
        return None
    
    mhd_files = list(seg_dir.glob("*.mhd"))
    print(f"\nTotal segmentation masks: {len(mhd_files)}")
    
    if mhd_files:
        # Load a sample segmentation
        sample_seg = mhd_files[0]
        print(f"\nSample segmentation: {sample_seg.name}")
        
        img = sitk.ReadImage(str(sample_seg))
        arr = sitk.GetArrayFromImage(img)
        
        print(f"  Shape: {arr.shape}")
        print(f"  Unique values: {np.unique(arr)}")
        print(f"  Lung voxels: {(arr > 0).sum()}")
    
    return {'count': len(mhd_files)}


def generate_report(stats):
    """Generate a markdown exploration report."""
    report_path = REPORTS_DIR / "data_exploration_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# LUNA16 Data Exploration Report\n\n")
        f.write("## Dataset Overview\n\n")
        f.write(f"- **Total CT Scans**: {sum(stats['subsets'].values())}\n")
        f.write(f"- **Total True Nodules**: {stats['class_balance']['positive']}\n")
        f.write(f"- **Total Candidates**: {stats['class_balance']['positive'] + stats['class_balance']['negative']}\n")
        f.write(f"- **Class Imbalance**: 1:{stats['class_balance']['ratio']:.0f}\n\n")
        
        f.write("## Subset Distribution\n\n")
        f.write("| Subset | Scans | Purpose |\n")
        f.write("|--------|-------|--------|\n")
        for i in range(5):
            purpose = "Train" if i < 3 else ("Validation" if i == 3 else "Test")
            f.write(f"| subset{i} | {stats['subsets'].get(f'subset{i}', 0)} | {purpose} |\n")
        
        f.write("\n## Nodule Size Distribution\n\n")
        f.write(f"- **Tiny (<4mm)**: {stats['nodule_sizes']['tiny']}\n")
        f.write(f"- **Small (4-6mm)**: {stats['nodule_sizes']['small']}\n")
        f.write(f"- **Medium (6-10mm)**: {stats['nodule_sizes']['medium']}\n")
        f.write(f"- **Large (>10mm)**: {stats['nodule_sizes']['large']}\n\n")
        
        f.write("## Key Figures\n\n")
        f.write("![Class Imbalance](figures/class_imbalance.png)\n\n")
        f.write("![Nodule Sizes](figures/nodule_sizes.png)\n\n")
        f.write("![Sample Scan](figures/sample_scan_views.png)\n\n")
        
        f.write("## Hardware Considerations\n\n")
        if stats.get('sample_scan'):
            f.write(f"- **Single scan size**: ~{stats['sample_scan']['size_mb']:.0f} MB\n")
            f.write(f"- **Scan shape**: {stats['sample_scan']['shape']}\n")
        f.write("- **GPU VRAM**: 3.5GB (RTX 3050)\n")
        f.write("- **Strategy**: Extract patches, don't load full scans\n")
    
    print(f"\nSaved report: {report_path}")


def main():
    """Main exploration pipeline."""
    print("\n" + "="*60)
    print("LUNA16 DATASET EXPLORATION")
    print("="*60)
    
    stats = {}
    
    # 1. Load and analyze annotations
    annotations_df = load_annotations()
    
    # 2. Load and analyze candidates
    candidates_df = load_candidates()
    
    # 3. Analyze class imbalance
    stats['class_balance'] = analyze_class_imbalance(candidates_df)
    
    # 4. Analyze nodule sizes
    stats['nodule_sizes'] = analyze_nodule_sizes(annotations_df)
    
    # 5. Analyze subsets
    stats['subsets'] = analyze_subsets()
    
    # 6. Analyze sample scan
    stats['sample_scan'] = analyze_sample_scan()
    
    # 7. Analyze lung segmentations
    stats['segmentations'] = analyze_lung_segmentations()
    
    # Save statistics as JSON
    stats_path = REPORTS_DIR / "statistics.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved statistics: {stats_path}")
    
    # Generate markdown report
    generate_report(stats)
    
    print("\n" + "="*60)
    print("EXPLORATION COMPLETE!")
    print("="*60)
    print(f"\nOutput files:")
    print(f"  - {REPORTS_DIR / 'statistics.json'}")
    print(f"  - {REPORTS_DIR / 'data_exploration_report.md'}")
    print(f"  - {FIGURES_DIR / 'class_imbalance.png'}")
    print(f"  - {FIGURES_DIR / 'nodule_sizes.png'}")
    print(f"  - {FIGURES_DIR / 'sample_scan_views.png'}")


if __name__ == "__main__":
    main()
