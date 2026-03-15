#!/usr/bin/env python3
"""
CT scan preprocessor for the inference pipeline.

This is the earlier working detection flow restored with stricter defaults
to reduce over-detection.
"""

from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom


HU_MIN = -1000
HU_MAX = 400
NODULE_PATCH_SIZE = 64
CONTEXT_PATCH_SIZE = 96
CONTEXT_TARGET_SIZE = 48
CLASSIFIER_PATCH_SIZE = 32


def load_ct_scan(scan_path):
    """Load a CT scan from .mhd, .nii, .nii.gz, .npz, or .npy."""
    scan_path = str(scan_path)
    ext = Path(scan_path).suffix.lower()

    if ext in ['.npz', '.npy']:
        try:
            if ext == '.npz':
                data = np.load(scan_path)
                if 'patch' in data:
                    ct_array = data['patch']
                elif 'image' in data:
                    ct_array = data['image']
                else:
                    ct_array = data[data.files[0]]
            else:
                ct_array = np.load(scan_path)
            return ct_array.astype(np.float32), np.zeros(3), np.ones(3), np.eye(3)
        except Exception as exc:
            raise ValueError(f"Failed to load numpy file {scan_path}: {exc}") from exc

    try:
        image = sitk.ReadImage(scan_path)
    except RuntimeError as exc:
        emsg = str(exc)
        if ext == '.mhd' and 'No such file or directory' in emsg:
            raise ValueError(
                "MHD file error: The associated .raw or .zraw file is missing. "
                "For web uploads, use .nii.gz or .npz if possible."
            ) from exc
        raise ValueError(f"SimpleITK failed to read {scan_path}: {emsg}") from exc

    ct_array = sitk.GetArrayFromImage(image)
    origin = np.array(image.GetOrigin())
    spacing = np.array(image.GetSpacing())
    direction = np.array(image.GetDirection()).reshape(3, 3)
    return ct_array, origin, spacing, direction


def normalize_hu(ct_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip HU values and normalize to [0, 1]."""
    clipped = np.clip(ct_array, hu_min, hu_max)
    normalized = (clipped - hu_min) / (hu_max - hu_min)
    return normalized.astype(np.float32)


def normalize_hu_signed(ct_array, hu_min=HU_MIN, hu_max=HU_MAX):
    """Clip HU values and normalize to [-1, 1]."""
    normalized = normalize_hu(ct_array, hu_min, hu_max)
    return (normalized * 2 - 1).astype(np.float32)


def create_lung_mask(ct_scan, threshold_lung=-320):
    """Create a conservative lung mask for candidate search."""
    from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
    from scipy.ndimage import generate_binary_structure
    from skimage.measure import label, regionprops

    print("Creating lung mask...")

    mask = ct_scan < threshold_lung
    mask = binary_erosion(mask, iterations=1)
    mask = binary_dilation(mask, iterations=1)

    labeled = label(mask)
    regions = regionprops(labeled)
    if not regions:
        print("ERROR: No lung regions found!")
        return np.zeros_like(ct_scan, dtype=bool)

    valid_regions = []
    total_vol = mask.size
    for region in regions:
        if region.area < 1000 or region.area > total_vol * 0.3:
            continue
        cz, _, _ = region.centroid
        if cz < mask.shape[0] * 0.1 or cz > mask.shape[0] * 0.9:
            continue
        valid_regions.append(region)

    valid_regions.sort(key=lambda region: region.area, reverse=True)
    lung_mask = np.zeros_like(mask, dtype=bool)
    for region in valid_regions[:2]:
        lung_mask[labeled == region.label] = True

    lung_mask = binary_fill_holes(lung_mask)
    struct = generate_binary_structure(3, 1)
    lung_mask = binary_dilation(lung_mask, structure=struct, iterations=2)

    print("  Removing mediastinum...")
    center_x = lung_mask.shape[2] // 2
    mediastinum_width = int(lung_mask.shape[2] * 0.25 / 2)
    lung_mask[:, :, center_x - mediastinum_width:center_x + mediastinum_width] = False

    print("  Restricting to lung z-range...")
    lung_slices = np.any(lung_mask, axis=(1, 2))
    lung_z_indices = np.where(lung_slices)[0]

    if len(lung_z_indices) > 0:
        lung_z_min = int(lung_z_indices[0])
        lung_z_max = int(lung_z_indices[-1])
        z_extent = lung_z_max - lung_z_min

        if z_extent > lung_mask.shape[0] * 0.80:
            print(f"    WARNING: Lung mask too large ({z_extent} slices), using defaults")
            lung_z_min = int(lung_mask.shape[0] * 0.15)
            lung_z_max = int(lung_mask.shape[0] * 0.75)
        else:
            margin = int(z_extent * 0.10)
            lung_z_min = max(0, lung_z_min - margin)
            lung_z_max = min(lung_mask.shape[0] - 1, lung_z_max + margin)

        lung_mask[:lung_z_min, :, :] = False
        lung_mask[lung_z_max + 1:, :, :] = False
        print(f"    Lung z-range: [{lung_z_min}:{lung_z_max}]")

    print(f"✓ Final lung mask: {np.sum(lung_mask):,} voxels")
    return lung_mask


def find_candidates_blob(ct_normalized, lung_mask, min_sigma=1.5, max_sigma=7,
                         num_sigma=10, threshold=0.12, max_candidates=50):
    """Find candidate nodules using 3D Difference of Gaussians inside lungs."""
    from skimage.feature import blob_dog

    del num_sigma

    lung_indices = np.where(lung_mask)
    if len(lung_indices[0]) == 0:
        return []

    z_min, z_max = lung_indices[0].min(), lung_indices[0].max()
    y_min, y_max = lung_indices[1].min(), lung_indices[1].max()
    x_min, x_max = lung_indices[2].min(), lung_indices[2].max()

    buffer = 5
    z_min = max(0, z_min - buffer)
    z_max = min(ct_normalized.shape[0], z_max + buffer)
    y_min = max(0, y_min - buffer)
    y_max = min(ct_normalized.shape[1], y_max + buffer)
    x_min = max(0, x_min - buffer)
    x_max = min(ct_normalized.shape[2], x_max + buffer)

    ct_crop = ct_normalized[z_min:z_max, y_min:y_max, x_min:x_max]

    print(f"Detecting blobs in 3D DoG ({ct_crop.shape})...")
    blobs = blob_dog(
        ct_crop,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=threshold,
        overlap=0.5,
    )

    print(f"  Found {len(blobs)} raw candidates")

    candidates = []
    rejected_outside = 0
    rejected_intensity = 0
    rejected_size = 0

    for blob in blobs:
        z_c, y_c, x_c, sigma = blob
        z = int(z_c + z_min)
        y = int(y_c + y_min)
        x = int(x_c + x_min)
        radius = sigma * np.sqrt(3)

        if (
            z >= ct_normalized.shape[0] or
            y >= ct_normalized.shape[1] or
            x >= ct_normalized.shape[2] or
            not lung_mask[z, y, x]
        ):
            rejected_outside += 1
            continue

        patch_size = 3
        z1, z2 = max(0, z - patch_size), min(ct_normalized.shape[0], z + patch_size + 1)
        y1, y2 = max(0, y - patch_size), min(ct_normalized.shape[1], y + patch_size + 1)
        x1, x2 = max(0, x - patch_size), min(ct_normalized.shape[2], x + patch_size + 1)
        patch_intensity = ct_normalized[z1:z2, y1:y2, x1:x2].mean()

        if patch_intensity < 0.18:
            rejected_intensity += 1
            continue

        if radius < 1.0 or radius > 15:
            rejected_size += 1
            continue

        candidates.append({
            'location': (z, y, x),
            'radius': radius,
            'intensity': float(patch_intensity),
        })

    print(f"✓ {len(candidates)} valid candidates")
    print(f"  Rejected: {rejected_outside} outside, {rejected_intensity} low, {rejected_size} size")

    candidates.sort(key=lambda cand: cand['intensity'], reverse=True)
    return candidates[:max_candidates]


def extract_patch(ct_array, center_zyx, patch_size, pad_value=0):
    """Extract a cubic patch centered at the given voxel coordinate."""
    half = patch_size // 2
    cz, cy, cx = center_zyx
    z_dim, y_dim, x_dim = ct_array.shape

    z0, z1 = cz - half, cz + half
    y0, y1 = cy - half, cy + half
    x0, x1 = cx - half, cx + half

    pads = [
        (max(0, -z0), max(0, z1 - z_dim)),
        (max(0, -y0), max(0, y1 - y_dim)),
        (max(0, -x0), max(0, x1 - x_dim)),
    ]
    needs_pad = any(before > 0 or after > 0 for before, after in pads)

    if needs_pad:
        ct_array = np.pad(ct_array, pads, mode='constant', constant_values=pad_value)
        cz += pads[0][0]
        cy += pads[1][0]
        cx += pads[2][0]
        z0, z1 = cz - half, cz + half
        y0, y1 = cy - half, cy + half
        x0, x1 = cx - half, cx + half

    patch = ct_array[z0:z1, y0:y1, x0:x1]
    if patch.shape == (patch_size, patch_size, patch_size):
        return patch

    result = np.full((patch_size, patch_size, patch_size), pad_value, dtype=ct_array.dtype)
    sz = min(patch.shape[0], patch_size)
    sy = min(patch.shape[1], patch_size)
    sx = min(patch.shape[2], patch_size)
    result[:sz, :sy, :sx] = patch[:sz, :sy, :sx]
    return result


def downsample_patch(patch, target_size):
    """Downsample a 3D patch using trilinear interpolation."""
    if patch is None:
        return None
    factor = target_size / patch.shape[0]
    return zoom(patch, factor, order=1).astype(np.float32)


def preprocess_for_detection(scan_path, use_blob_candidates=True):
    """Full preprocessing pipeline for inference."""
    ct_raw, origin, spacing, direction = load_ct_scan(scan_path)

    target_spacing = np.array([1.0, 1.0, 1.0])
    current_spacing = spacing[::-1]
    resize_factor = current_spacing / target_spacing
    if not np.allclose(resize_factor, 1.0, atol=1e-2):
        ct_raw = zoom(ct_raw, resize_factor, order=1).astype(np.float32)
        spacing = target_spacing[::-1]

    ct_01 = normalize_hu(ct_raw)
    ct_signed = normalize_hu_signed(ct_raw)

    lung_mask = create_lung_mask(ct_raw)

    candidates_raw = []
    if use_blob_candidates:
        candidates_raw = find_candidates_blob(ct_01, lung_mask)

    candidates = []
    half_nodule = NODULE_PATCH_SIZE // 2
    for candidate in candidates_raw:
        z, y, x = candidate['location']

        if (
            z < half_nodule or z >= ct_raw.shape[0] - half_nodule or
            y < half_nodule or y >= ct_raw.shape[1] - half_nodule or
            x < half_nodule or x >= ct_raw.shape[2] - half_nodule
        ):
            continue

        nodule_patch = extract_patch(ct_signed, (z, y, x), NODULE_PATCH_SIZE, pad_value=-1.0)
        context_patch_96 = extract_patch(ct_signed, (z, y, x), CONTEXT_PATCH_SIZE, pad_value=-1.0)
        context_patch = downsample_patch(context_patch_96, CONTEXT_TARGET_SIZE)

        candidates.append({
            'nodule_patch': nodule_patch,
            'context_patch': context_patch,
            'location': (z, y, x),
            'radius': candidate['radius'],
            'intensity': candidate['intensity'],
        })

    metadata = {
        'origin': origin.tolist(),
        'spacing': spacing.tolist(),
        'shape': list(ct_raw.shape),
        'scan_path': str(scan_path),
        'direction': direction.tolist(),
    }
    return candidates, ct_01, metadata, lung_mask


def extract_classification_patch(ct_normalized, location, size=CLASSIFIER_PATCH_SIZE):
    """Extract a 32^3 patch for the malignancy classifier."""
    if ct_normalized.shape == (NODULE_PATCH_SIZE,) * 3:
        offset = (NODULE_PATCH_SIZE - size) // 2
        return ct_normalized[
            offset:offset + size,
            offset:offset + size,
            offset:offset + size,
        ].copy()

    return extract_patch(ct_normalized, location, size, pad_value=0)
