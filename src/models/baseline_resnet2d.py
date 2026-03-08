#!/usr/bin/env python3
"""
Baseline 2D ResNet-18 (Slice-Level) — Standalone module.

This module provides the ResNet2D18SliceLevel class as a standalone import
per the ablation study specification. The actual implementation lives in
baselines.py; this file re-exports it for clean per-model imports.

Usage:
    from src.models.baseline_resnet2d import ResNet2D18SliceLevel
    model = ResNet2D18SliceLevel(num_classes=1)

Architecture:
    - Backbone: torchvision resnet18 (2D)
    - Input conv modified: 3-channel → 1-channel (grayscale CT slices)
    - Each 3D volume (B, 1, D, H, W) is processed slice-by-slice:
        1. Reshape → (B*D, 1, H, W)
        2. Forward through ResNet-18 backbone → (B*D, 512)
        3. Reshape back → (B, D, 512)
        4. Global Average Pool over depth → (B, 512)
        5. Classification head → (B, 1)
    - Forward signature: forward(nodule_patch, context_patch=None)
      (context_patch is ignored — 2D baseline uses nodule slices only)
"""

from src.models.baselines import ResNet2D18SliceLevel

__all__ = ['ResNet2D18SliceLevel']
