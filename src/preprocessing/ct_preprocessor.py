#!/usr/bin/env python3
"""
CT Scan Preprocessor for Inference Pipeline

Handles loading raw CT scans (.mhd, .nii.gz, .nii), preprocessing,
candidate extraction via blob detection, and patch extraction for
both DCA-Net detection and malignancy classification.

Optimized for RTX 3050 (4GB VRAM) or CPU inference.
"""

import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom


# ── Constants (match training preprocessing) ──
HU_MIN = -1000
HU_MAX = 400
NODULE_PATCH_SIZE = 64          # DCA-Net nodule input
CONTEXT_PATCH_SIZE = 96         # Extract 96³, downsample to 48³
CONTEXT_TARGET_SIZE = 48
CLASSIFIER_PATCH_SIZE = 32      # Malignancy classifier input


def load_ct_scan(scan_path):
    """Load a CT scan from .mhd, .nii, or .nii.gz file.

    Args:
        scan_path: Path to CT scan file

    Returns:
        ct_array: numpy (Z, Y, X) in HU
        origin: (X, Y, Z) world origin
        spacing: (X, Y, Z) voxel spacing in mm
        direction: 3×3 direction cosines
    """
    scan_path = str(scan_path)
    image = sitk.ReadImage(scan_path)
    ct_array = sitk.GetArrayFromImage(image)    # (Z, Y, X)
    origin = np.array(image.GetOrigin())        # (X, Y, Z)
    spacing = np.array(image.GetSpacing())      # (X, Y, Z)
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return ct_array, origin, spacing, direction


def normalize_hu(ct_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip HU values and normalize to [0, 1].

    Args:
        ct_array: Raw HU numpy array
        hu_min: Lower HU bound (default -1000)
        hu_max: Upper HU bound (default 400)

    Returns:
        Normalized array in [0, 1]
    """
    clipped = np.clip(ct_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def normalize_hu_signed(ct_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip HU values and normalize to [-1, 1] (for DCA-Net).

    Args:
        ct_array: Raw HU numpy array

    Returns:
        Normalized array in [-1, 1]
    """
    clipped = np.clip(ct_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)  # [0, 1]
    signed = normalized * 2 - 1                           # [-1, 1]
    return signed.astype(np.float32)


def find_candidates_blob(ct_normalized, min_sigma=3, max_sigma=15,
                         num_sigma=8, threshold=0.08, max_candidates=50):
    """Find nodule candidate locations using blob detection.

    Uses Laplacian of Gaussian (LoG) blob detection on the normalized
    CT volume to find round bright structures (potential nodules).

    Args:
        ct_normalized: Normalized CT array [0, 1], shape (Z, Y, X)
        min_sigma: Minimum blob scale
        max_sigma: Maximum blob scale
        num_sigma: Number of sigma values between min and max
        threshold: Detection threshold (lower = more candidates)
        max_candidates: Maximum candidates to return

    Returns:
        List of dicts: [{'location': (z, y, x), 'radius': float}, ...]
    """
    from skimage.feature import blob_log

    # Run on 2D slices at intervals to keep it fast
    # Then merge detections across slices
    all_blobs = []
    step = max(1, ct_normalized.shape[0] // 40)  # Sample ~40 slices

    for z_idx in range(0, ct_normalized.shape[0], step):
        slice_2d = ct_normalized[z_idx]
        blobs_2d = blob_log(
            slice_2d, min_sigma=min_sigma, max_sigma=max_sigma,
            num_sigma=num_sigma, threshold=threshold
        )
        for y, x, sigma in blobs_2d:
            all_blobs.append({
                'location': (z_idx, int(y), int(x)),
                'radius': sigma * np.sqrt(2),
            })

    # Deduplicate: merge blobs within 10 voxels of each other
    if len(all_blobs) == 0:
        return []

    merged = []
    used = set()
    for i, b1 in enumerate(all_blobs):
        if i in used:
            continue
        cluster = [b1]
        for j, b2 in enumerate(all_blobs[i + 1:], start=i + 1):
            if j in used:
                continue
            dist = np.sqrt(sum((a - b) ** 2 for a, b in
                               zip(b1['location'], b2['location'])))
            if dist < 10:
                cluster.append(b2)
                used.add(j)
        used.add(i)
        # Average the cluster
        avg_loc = tuple(int(np.mean([c['location'][d] for c in cluster]))
                        for d in range(3))
        avg_rad = np.mean([c['radius'] for c in cluster])
        merged.append({'location': avg_loc, 'radius': avg_rad})

    # Sort by radius (larger = more suspicious) and limit
    merged.sort(key=lambda b: b['radius'], reverse=True)
    return merged[:max_candidates]


def extract_patch(ct_array, center_zyx, patch_size, pad_value=0):
    """Extract a cubic patch centered at the given voxel coordinate.

    Handles boundary padding automatically.

    Args:
        ct_array: 3D numpy array (Z, Y, X)
        center_zyx: (z, y, x) voxel coordinates
        patch_size: Side length of the cubic patch
        pad_value: Value to use for padding

    Returns:
        patch: numpy array of shape (patch_size, patch_size, patch_size)
    """
    half = patch_size // 2
    cz, cy, cx = center_zyx
    Z, Y, X = ct_array.shape

    # Compute slice bounds
    z0, z1 = cz - half, cz + half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    # Check if padding is needed
    pads = [
        (max(0, -z0), max(0, z1 - Z)),
        (max(0, -y0), max(0, y1 - Y)),
        (max(0, -x0), max(0, x1 - X)),
    ]
    needs_pad = any(p[0] > 0 or p[1] > 0 for p in pads)

    if needs_pad:
        ct_array = np.pad(ct_array, pads, mode='constant',
                          constant_values=pad_value)
        cz += pads[0][0]
        cy += pads[1][0]
        cx += pads[2][0]
        z0, z1 = cz - half, cz + half
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half

    patch = ct_array[z0:z1, y0:y1, x0:x1]

    # Final size check
    if patch.shape != (patch_size, patch_size, patch_size):
        # Fallback: pad to correct size
        result = np.full((patch_size, patch_size, patch_size),
                         pad_value, dtype=ct_array.dtype)
        sz = min(patch.shape[0], patch_size)
        sy = min(patch.shape[1], patch_size)
        sx = min(patch.shape[2], patch_size)
        result[:sz, :sy, :sx] = patch[:sz, :sy, :sx]
        return result

    return patch


def downsample_patch(patch, target_size):
    """Downsample a 3D patch using trilinear interpolation."""
    if patch is None:
        return None
    factor = target_size / patch.shape[0]
    return zoom(patch, factor, order=1).astype(np.float32)


def preprocess_for_detection(scan_path, use_blob_candidates=True):
    """Full preprocessing pipeline for inference.

    Loads CT scan, finds candidates, extracts patches for DCA-Net.

    Args:
        scan_path: Path to .mhd / .nii.gz file
        use_blob_candidates: Use blob detection for candidates

    Returns:
        candidates: List of dicts with 'nodule_patch', 'context_patch',
                    'location', 'radius'
        ct_normalized: Full normalized CT for visualization
        metadata: dict with origin, spacing, shape
    """
    # Load scan
    ct_raw, origin, spacing, direction = load_ct_scan(scan_path)

    # Normalize for visualization (0-1)
    ct_01 = normalize_hu(ct_raw)

    # Normalize for model input (-1 to 1)
    ct_signed = normalize_hu_signed(ct_raw)

    # Find candidates
    candidates_raw = []
    if use_blob_candidates:
        candidates_raw = find_candidates_blob(ct_01)

    # Extract patches for each candidate
    candidates = []
    for cand in candidates_raw:
        z, y, x = cand['location']

        # Nodule patch: 64³ (for DCA-Net)
        nodule_patch = extract_patch(ct_signed, (z, y, x),
                                     NODULE_PATCH_SIZE, pad_value=-1.0)

        # Context patch: 96³ → downsample to 48³ (for DCA-Net)
        context_patch_96 = extract_patch(ct_signed, (z, y, x),
                                         CONTEXT_PATCH_SIZE, pad_value=-1.0)
        context_patch = downsample_patch(context_patch_96, CONTEXT_TARGET_SIZE)

        candidates.append({
            'nodule_patch': nodule_patch,
            'context_patch': context_patch,
            'location': (z, y, x),
            'radius': cand['radius'],
        })

    metadata = {
        'origin': origin.tolist(),
        'spacing': spacing.tolist(),
        'shape': list(ct_raw.shape),
        'scan_path': str(scan_path),
    }

    return candidates, ct_01, metadata


def extract_classification_patch(ct_normalized, location, size=CLASSIFIER_PATCH_SIZE):
    """Extract a 32³ patch for the malignancy classifier.

    If the detection patch is 64³, this crops the center 32³.

    Args:
        ct_normalized: Normalized CT array or 64³ nodule patch
        location: (z, y, x) center coordinates (ignored if input is a patch)
        size: Output patch size (default 32)

    Returns:
        patch: (size, size, size) numpy array
    """
    if ct_normalized.shape == (NODULE_PATCH_SIZE,) * 3:
        # Crop center of 64³ → 32³
        offset = (NODULE_PATCH_SIZE - size) // 2
        return ct_normalized[offset:offset + size,
                             offset:offset + size,
                             offset:offset + size].copy()

    # Extract from full volume
    return extract_patch(ct_normalized, location, size, pad_value=0)
