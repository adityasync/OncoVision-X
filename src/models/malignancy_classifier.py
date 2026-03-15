#!/usr/bin/env python3
"""Malignancy classifier using a 3D ResNet backbone."""

from pathlib import Path

import torch
import torch.nn as nn


class MalignancyClassifier(nn.Module):
    """3D ResNet-based malignancy classifier."""

    def __init__(self, pretrained_path=None, use_torchvision_pretrained=True):
        super().__init__()

        try:
            from torchvision.models.video import R3D_18_Weights, r3d_18

            if use_torchvision_pretrained and pretrained_path is None:
                self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
            else:
                self.backbone = r3d_18(weights=None)
        except ImportError:
            from torchvision.models.video import r3d_18

            self.backbone = r3d_18(
                pretrained=(use_torchvision_pretrained and pretrained_path is None)
            )

        if pretrained_path and Path(pretrained_path).exists():
            try:
                state = torch.load(pretrained_path, map_location='cpu', weights_only=False)
                if 'state_dict' in state:
                    state = state['state_dict']
                clean_state = {key.replace('module.', ''): value for key, value in state.items()}
                self.backbone.load_state_dict(clean_state, strict=False)
                print(f"  Loaded MedicalNet weights from {pretrained_path}")
            except Exception as exc:
                print(f"  Warning: could not load MedicalNet weights: {exc}")

        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """Forward pass on (B, 1, 32, 32, 32) patches."""
        x = x.repeat(1, 3, 1, 1, 1)
        return self.backbone(x)
