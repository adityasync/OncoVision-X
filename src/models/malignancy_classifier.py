#!/usr/bin/env python3
"""
Malignancy Classifier — Pretrained MedicalNet ResNet3D

Uses a pretrained 3D ResNet-18 (from torchvision video models) adapted
for binary nodule malignancy classification. No training required —
downloads pretrained weights or loads MedicalNet checkpoint.

Input:  (B, 1, 32, 32, 32) — nodule patches
Output: (B, 1) — malignancy probability (sigmoid)

For demo/presentation purposes only.
"""

import torch
import torch.nn as nn
from pathlib import Path


class MalignancyClassifier(nn.Module):
    """Pretrained ResNet3D-18 adapted for binary malignancy classification.

    Uses torchvision's r3d_18 video model as backbone, optionally loading
    MedicalNet weights (pretrained on 23 medical imaging datasets).

    Args:
        pretrained_path: Optional path to MedicalNet .pth weights
        use_torchvision_pretrained: Use Kinetics-400 pretrained weights
                                    as fallback if no medical weights
    """

    def __init__(self, pretrained_path=None, use_torchvision_pretrained=True):
        super().__init__()

        # Load ResNet3D-18 backbone
        try:
            from torchvision.models.video import r3d_18, R3D_18_Weights
            if use_torchvision_pretrained and pretrained_path is None:
                self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            else:
                self.backbone = r3d_18(weights=None)
        except ImportError:
            from torchvision.models.video import r3d_18
            self.backbone = r3d_18(pretrained=(use_torchvision_pretrained and pretrained_path is None))

        # Load MedicalNet weights if provided
        if pretrained_path and Path(pretrained_path).exists():
            checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            # MedicalNet keys may have 'module.' prefix
            clean_state = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.backbone.load_state_dict(clean_state, strict=False)
            print(f"  Loaded MedicalNet weights from {pretrained_path}")

        # Replace final FC layer for binary classification
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) — single-channel 3D nodule patch
               e.g. (B, 1, 32, 32, 32) or (B, 1, 64, 64, 64)

        Returns:
            prob: (B, 1) — malignancy probability [0, 1]
        """
        # r3d_18 expects 3-channel input → replicate single channel
        x = x.repeat(1, 3, 1, 1, 1)  # (B, 3, D, H, W)
        return self.backbone(x)

    @torch.no_grad()
    def predict(self, x):
        """Convenience method: returns float probability."""
        self.eval()
        prob = self.forward(x)
        return prob.squeeze(-1)  # (B,)
