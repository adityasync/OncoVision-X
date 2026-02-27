#!/usr/bin/env python3
"""
DCA-Net Lung Cancer Prediction Script
======================================
Classify lung nodule candidates using the trained DCA-Net model.

Usage:
  # Mode 1: CT scan + single coordinate (world coords in mm)
  python predict.py --scan path/to/scan.mhd --coords "100.5,200.3,-150.7"

  # Mode 2: CT scan + CSV of candidate coordinates
  python predict.py --scan path/to/scan.mhd --candidates candidates.csv

  # Mode 3: Pre-extracted patches
  python predict.py --nodule-patch path/to/nodule.npz --context-patch path/to/context.npz

  # Options
  python predict.py --scan scan.mhd --coords "x,y,z" --checkpoint path/to/model.pth --output results.json
"""

import argparse
import json
import sys
import os
import warnings
from pathlib import Path

import numpy as np
import yaml

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Constants (match training preprocessing)
# ─────────────────────────────────────────────────────────────
HU_MIN = -1000
HU_MAX = 400
NODULE_PATCH_SIZE = 64
CONTEXT_PATCH_SIZE = 96       # Extract 96³, downsample to 48³
CONTEXT_TARGET_SIZE = 48


def load_config(config_path='configs/training_config.yaml'):
    """Load training configuration."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_scan(mhd_path):
    """Load a CT scan from .mhd file using SimpleITK."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(mhd_path))
    arr = sitk.GetArrayFromImage(img)   # (z, y, x)
    origin = np.array(img.GetOrigin())   # (x, y, z)
    spacing = np.array(img.GetSpacing()) # (x, y, z)
    direction = np.array(img.GetDirection()).reshape(3, 3)
    return arr, origin, spacing, direction


def world_to_voxel(world_coord, origin, spacing):
    """Convert world coordinates (mm) to voxel indices."""
    voxel_coord = (world_coord - origin) / spacing
    return np.round(voxel_coord).astype(int)


def apply_hu_windowing(arr, hu_min=HU_MIN, hu_max=HU_MAX):
    """Apply HU windowing and normalize to [-1, 1]."""
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)   # [0, 1]
    arr = arr * 2 - 1                           # [-1, 1]
    return arr.astype(np.float32)


def extract_patch(arr, center_xyz, patch_size):
    """Extract a cubic patch centered at the given voxel coordinate.

    Args:
        arr: 3D numpy array (z, y, x)
        center_xyz: (x, y, z) voxel coordinates
        patch_size: Side length of the cubic patch

    Returns:
        patch or None if out of bounds
    """
    half = patch_size // 2
    cx, cy, cz = center_xyz

    z_start, z_end = cz - half, cz + half
    y_start, y_end = cy - half, cy + half
    x_start, x_end = cx - half, cx + half

    # Pad if out of bounds
    pad_needed = False
    pads = [(0, 0)] * 3
    if z_start < 0 or z_end > arr.shape[0]:
        pads[0] = (max(0, -z_start), max(0, z_end - arr.shape[0]))
        pad_needed = True
    if y_start < 0 or y_end > arr.shape[1]:
        pads[1] = (max(0, -y_start), max(0, y_end - arr.shape[1]))
        pad_needed = True
    if x_start < 0 or x_end > arr.shape[2]:
        pads[2] = (max(0, -x_start), max(0, x_end - arr.shape[2]))
        pad_needed = True

    if pad_needed:
        arr = np.pad(arr, pads, mode='constant', constant_values=HU_MIN)
        # Adjust coordinates for padding
        cz += pads[0][0]
        cy += pads[1][0]
        cx += pads[2][0]
        z_start, z_end = cz - half, cz + half
        y_start, y_end = cy - half, cy + half
        x_start, x_end = cx - half, cx + half

    patch = arr[z_start:z_end, y_start:y_end, x_start:x_end]
    if patch.shape != (patch_size, patch_size, patch_size):
        return None
    return patch


def downsample_patch(patch, target_size):
    """Downsample a 3D patch using scipy zoom."""
    from scipy.ndimage import zoom
    if patch is None:
        return None
    zoom_factor = target_size / patch.shape[0]
    return zoom(patch, zoom_factor, order=1)


def extract_candidate_patches(scan_arr, center_xyz):
    """Extract nodule and context patches for a single candidate.

    Args:
        scan_arr: Full CT scan array (z, y, x), raw HU values
        center_xyz: (x, y, z) voxel coordinates

    Returns:
        (nodule_patch, context_patch) or (None, None) if extraction fails
    """
    # Extract nodule patch (64³)
    nodule = extract_patch(scan_arr, center_xyz, NODULE_PATCH_SIZE)
    if nodule is None:
        return None, None

    # Extract context patch (96³ → downsample to 48³)
    context = extract_patch(scan_arr, center_xyz, CONTEXT_PATCH_SIZE)
    if context is None:
        return None, None
    context = downsample_patch(context, CONTEXT_TARGET_SIZE)

    # Apply HU windowing
    nodule = apply_hu_windowing(nodule)
    context = apply_hu_windowing(context)

    return nodule, context


def load_model(checkpoint_path, config, device):
    """Load trained DCA-Net model from checkpoint."""
    import torch
    sys.path.insert(0, str(Path(__file__).parent))
    from src.models.dca_net import DCANet

    model = DCANet(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state_dict = checkpoint['model_state_dict']
    # Remove DataParallel prefix if present
    clean_state = {}
    for k, v in state_dict.items():
        clean_state[k.replace('module.', '')] = v

    model.load_state_dict(clean_state)
    model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'unknown')
    val_loss = checkpoint.get('best_val_loss', 0)
    print(f"  Model loaded from epoch {epoch} (val_loss: {val_loss:.4f})")
    return model


def predict_single(model, nodule_patch, context_patch, device):
    """Run model inference on a single candidate.

    Returns:
        probability (float), classification (str), confidence (str)
    """
    import torch

    # Convert to tensors: (1, 1, D, H, W)
    nodule_t = torch.from_numpy(nodule_patch).unsqueeze(0).unsqueeze(0).to(device)
    context_t = torch.from_numpy(context_patch).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(nodule_t, context_t)
        prob = torch.sigmoid(logits.squeeze()).item()

    # Classification
    label = "MALIGNANT" if prob > 0.5 else "BENIGN"

    # Confidence level
    confidence = abs(prob - 0.5) * 2  # 0-1 scale
    if confidence > 0.8:
        conf_str = "Very High"
    elif confidence > 0.6:
        conf_str = "High"
    elif confidence > 0.4:
        conf_str = "Moderate"
    elif confidence > 0.2:
        conf_str = "Low"
    else:
        conf_str = "Very Low"

    return prob, label, conf_str


def mc_dropout_predict(model, nodule_patch, context_patch, device, num_passes=10):
    """Run MC Dropout inference for uncertainty estimation.

    Returns:
        mean_prob, std_prob, predictions list
    """
    import torch

    model.train()  # Enable dropout
    nodule_t = torch.from_numpy(nodule_patch).unsqueeze(0).unsqueeze(0).to(device)
    context_t = torch.from_numpy(context_patch).unsqueeze(0).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(num_passes):
            logits = model(nodule_t, context_t)
            prob = torch.sigmoid(logits.squeeze()).item()
            predictions.append(prob)

    model.eval()

    mean_prob = np.mean(predictions)
    std_prob = np.std(predictions)
    return mean_prob, std_prob, predictions


# ─────────────────────────────────────────────────────────────
# Display functions
# ─────────────────────────────────────────────────────────────
def print_header():
    print("\n" + "=" * 60)
    print("  DCA-NET — Lung Nodule Classification")
    print("  Dual-Context Attention Network")
    print("=" * 60)


def print_result(idx, coords, prob, label, confidence, uncertainty=None):
    """Print a single prediction result."""
    # Color coding
    if label == "MALIGNANT":
        color = "\033[91m"  # Red
        icon = "!!!"
    else:
        color = "\033[92m"  # Green
        icon = "OK "
    reset = "\033[0m"
    bold = "\033[1m"

    print(f"\n  [{icon}] Candidate {idx}")
    if coords is not None:
        print(f"       Coordinates: ({coords[0]:.1f}, {coords[1]:.1f}, {coords[2]:.1f}) mm")
    print(f"       Probability: {bold}{color}{prob:.4f}{reset}")
    print(f"       Classification: {bold}{color}{label}{reset}")
    print(f"       Confidence: {confidence}")
    if uncertainty is not None:
        print(f"       Uncertainty (std): {uncertainty:.4f}")


def print_summary(results):
    """Print batch prediction summary."""
    total = len(results)
    malignant = sum(1 for r in results if r['classification'] == 'MALIGNANT')
    benign = total - malignant

    print("\n" + "-" * 60)
    print(f"  SUMMARY: {total} candidates analyzed")
    print(f"    Malignant: \033[91m{malignant}\033[0m")
    print(f"    Benign:    \033[92m{benign}\033[0m")

    if malignant > 0:
        print(f"\n  \033[1m\033[91mATTENTION: {malignant} suspicious nodule(s) detected.\033[0m")
        print("  Consult a radiologist for clinical interpretation.")
    else:
        print(f"\n  \033[92mNo suspicious nodules detected.\033[0m")
        print("  Note: This is an AI screening tool, not a clinical diagnosis.")

    print("-" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# Main entry points
# ─────────────────────────────────────────────────────────────
def predict_from_scan(args, model, device):
    """Predict from a CT scan with coordinates."""
    print(f"\n  Loading CT scan: {args.scan}")
    scan_arr, origin, spacing, direction = load_scan(args.scan)
    print(f"  Scan shape: {scan_arr.shape}")
    print(f"  Origin: {origin}")
    print(f"  Spacing: {spacing}")

    # Collect candidates
    candidates = []
    if args.coords:
        # Single coordinate
        coords = [float(x.strip()) for x in args.coords.split(',')]
        if len(coords) != 3:
            print("ERROR: --coords must be x,y,z (3 values)")
            sys.exit(1)
        candidates.append(np.array(coords))
    elif args.candidates:
        # CSV file with coordinates
        import pandas as pd
        df = pd.read_csv(args.candidates)
        for col_set in [('coordX', 'coordY', 'coordZ'), ('x', 'y', 'z')]:
            if all(c in df.columns for c in col_set):
                for _, row in df.iterrows():
                    candidates.append(np.array([row[col_set[0]], row[col_set[1]], row[col_set[2]]]))
                break
        else:
            print(f"ERROR: CSV must have columns (coordX,coordY,coordZ) or (x,y,z)")
            print(f"  Found columns: {list(df.columns)}")
            sys.exit(1)
    else:
        print("ERROR: Provide --coords or --candidates with --scan")
        sys.exit(1)

    print(f"\n  Processing {len(candidates)} candidate(s)...")

    results = []
    for i, world_coord in enumerate(candidates):
        # Convert world → voxel
        voxel_coord = world_to_voxel(world_coord, origin, spacing)
        print(f"\n  Candidate {i+1}: world=({world_coord[0]:.1f}, {world_coord[1]:.1f}, {world_coord[2]:.1f}) → voxel=({voxel_coord[0]}, {voxel_coord[1]}, {voxel_coord[2]})")

        # Extract patches
        nodule, context = extract_candidate_patches(scan_arr, voxel_coord)
        if nodule is None:
            print(f"  WARNING: Could not extract patches for candidate {i+1} (near scan boundary)")
            continue

        # Predict
        prob, label, confidence = predict_single(model, nodule, context, device)

        # Optional: MC Dropout uncertainty
        uncertainty = None
        if args.uncertainty:
            mean_prob, std_prob, _ = mc_dropout_predict(
                model, nodule, context, device, num_passes=args.mc_passes
            )
            uncertainty = std_prob
            prob = mean_prob
            label = "MALIGNANT" if prob > 0.5 else "BENIGN"

        print_result(i + 1, world_coord, prob, label, confidence, uncertainty)

        results.append({
            'candidate': i + 1,
            'world_coords': world_coord.tolist(),
            'voxel_coords': voxel_coord.tolist(),
            'probability': float(prob),
            'classification': label,
            'confidence': confidence,
            'uncertainty': float(uncertainty) if uncertainty else None
        })

    return results


def predict_from_patches(args, model, device):
    """Predict from pre-extracted patch files."""
    print(f"\n  Loading patches...")
    nodule = np.load(args.nodule_patch)['patch'].astype(np.float32)
    context = np.load(args.context_patch)['patch'].astype(np.float32)
    print(f"  Nodule patch shape:  {nodule.shape}")
    print(f"  Context patch shape: {context.shape}")

    prob, label, confidence = predict_single(model, nodule, context, device)

    uncertainty = None
    if args.uncertainty:
        mean_prob, std_prob, _ = mc_dropout_predict(
            model, nodule, context, device, num_passes=args.mc_passes
        )
        uncertainty = std_prob
        prob = mean_prob
        label = "MALIGNANT" if prob > 0.5 else "BENIGN"

    print_result(1, None, prob, label, confidence, uncertainty)

    results = [{
        'candidate': 1,
        'probability': float(prob),
        'classification': label,
        'confidence': confidence,
        'uncertainty': float(uncertainty) if uncertainty else None
    }]
    return results


def main():
    parser = argparse.ArgumentParser(
        description='DCA-Net Lung Nodule Classification',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a nodule at specific coordinates (world coords in mm)
  python predict.py --scan data/subset0/1.3.6.1.4.1.14519.mhd --coords "100.5,200.3,-150.7"

  # Batch classify from a CSV of candidates
  python predict.py --scan data/subset0/1.3.6.1.4.1.14519.mhd --candidates candidates.csv

  # Classify from pre-extracted patches
  python predict.py --nodule-patch preprocessed_data/patches/nodule_001.npz \\
                     --context-patch preprocessed_data/patches/context_001.npz

  # With uncertainty estimation (MC Dropout)
  python predict.py --scan scan.mhd --coords "x,y,z" --uncertainty --mc-passes 20
        """
    )

    # Input modes
    scan_group = parser.add_argument_group('CT Scan Input')
    scan_group.add_argument('--scan', type=str, help='Path to CT scan (.mhd file)')
    scan_group.add_argument('--coords', type=str,
                           help='Nodule coordinates in mm: "x,y,z"')
    scan_group.add_argument('--candidates', type=str,
                           help='CSV file with candidate coordinates (coordX,coordY,coordZ)')

    patch_group = parser.add_argument_group('Pre-extracted Patch Input')
    patch_group.add_argument('--nodule-patch', type=str,
                            help='Path to nodule patch .npz file')
    patch_group.add_argument('--context-patch', type=str,
                            help='Path to context patch .npz file')

    # Model options
    model_group = parser.add_argument_group('Model Options')
    model_group.add_argument('--checkpoint', type=str,
                            default='results/checkpoints/best.pth',
                            help='Path to model checkpoint (default: results/checkpoints/best.pth)')
    model_group.add_argument('--config', type=str,
                            default='configs/training_config.yaml',
                            help='Path to config file')
    model_group.add_argument('--device', type=str, default=None,
                            help='Device: cuda or cpu (auto-detected)')

    # Uncertainty
    unc_group = parser.add_argument_group('Uncertainty Estimation')
    unc_group.add_argument('--uncertainty', action='store_true',
                          help='Enable MC Dropout uncertainty estimation')
    unc_group.add_argument('--mc-passes', type=int, default=10,
                          help='Number of MC Dropout forward passes (default: 10)')

    # Output
    out_group = parser.add_argument_group('Output')
    out_group.add_argument('--output', type=str, default=None,
                          help='Save results to JSON file')
    out_group.add_argument('--quiet', action='store_true',
                          help='Minimal output (just probability and label)')

    args = parser.parse_args()

    # Validate inputs
    has_scan = args.scan is not None
    has_patches = args.nodule_patch is not None and args.context_patch is not None

    if not has_scan and not has_patches:
        parser.error("Provide either --scan (+ --coords/--candidates) or --nodule-patch + --context-patch")

    if has_scan and not args.coords and not args.candidates:
        parser.error("--scan requires --coords or --candidates")

    # Setup
    import torch
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not args.quiet:
        print_header()
        print(f"\n  Device: {device}")
        if device.type == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"\n  ERROR: Checkpoint not found: {args.checkpoint}")
        print("  Make sure you have the trained model at results/checkpoints/best.pth")
        sys.exit(1)

    # Load model
    if not args.quiet:
        print(f"\n  Loading model checkpoint: {args.checkpoint}")
    config = load_config(args.config)
    model = load_model(args.checkpoint, config, device)

    # Run prediction
    if has_scan:
        results = predict_from_scan(args, model, device)
    else:
        results = predict_from_patches(args, model, device)

    # Summary
    if not args.quiet and len(results) > 0:
        print_summary(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump({
                'model': str(args.checkpoint),
                'device': str(device),
                'num_candidates': len(results),
                'results': results
            }, f, indent=2)
        print(f"  Results saved to: {args.output}")

    return results


if __name__ == '__main__':
    main()
