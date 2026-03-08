#!/usr/bin/env python3
"""
Baseline 3D ResNet-18 — Standalone module.

This module provides the ResNet3D18 class as a standalone import
per the ablation study specification. The actual implementation
lives in baselines.py; this file re-exports it for clean per-model imports.

Usage:
    from src.models.baseline_resnet3d import ResNet3D18
    model = ResNet3D18(num_classes=1)

Architecture:
    - Backbone: torchvision video.r3d_18 (3D ResNet-18)
    - Input conv modified: 3-channel → 1-channel (grayscale CT)
    - Output: binary logit (B, 1)
    - Forward signature: forward(nodule_patch, context_patch=None)
      (context_patch is ignored — baseline uses nodule only)
"""

from src.models.baselines import ResNet3D18

__all__ = ['ResNet3D18']
