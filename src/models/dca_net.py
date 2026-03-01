#!/usr/bin/env python3
"""
DCA-Net: Dual-Context Attention Network for Lung Nodule Classification

Architecture per roadmap1.md Phase 2:
  - Stream 1 (Nodule): 2.5D CNN with EfficientNet-B0 + cross-slice attention
  - Stream 2 (Context): Lightweight 3D CNN with spatial attention
  - Fusion: Multi-head attention fusion module
  - Prediction head with dropout
  - Uncertainty quantification via MC Dropout
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ─────────────────────────────────────────────────────────────
# Cross-Slice Attention Module
# ─────────────────────────────────────────────────────────────
class CrossSliceAttention(nn.Module):
    """Learn spatial dependencies across adjacent slices (±k neighbors)."""

    def __init__(self, feature_dim, num_neighbors=2):
        super().__init__()
        self.num_neighbors = num_neighbors
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)
        self.scale = feature_dim ** 0.5
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, slice_features):
        """
        Args:
            slice_features: (B, num_slices, D)
        Returns:
            attended: (B, num_slices, D)
        """
        B, S, D = slice_features.shape
        Q = self.query(slice_features)  # (B, S, D)
        K = self.key(slice_features)
        V = self.value(slice_features)

        # Build a local attention mask so each slice only attends to ±k neighbors
        mask = torch.zeros(S, S, device=slice_features.device, dtype=torch.bool)
        for i in range(S):
            lo = max(0, i - self.num_neighbors)
            hi = min(S, i + self.num_neighbors + 1)
            mask[i, lo:hi] = True

        # Scaled dot-product attention
        attn = torch.matmul(Q, K.transpose(-1, -2)) / self.scale  # (B, S, S)
        attn = attn.masked_fill(~mask.unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, V)  # (B, S, D)
        out = self.norm(out + slice_features)  # residual + LayerNorm
        return out, attn


# ─────────────────────────────────────────────────────────────
# Stream 1: Nodule Feature Extractor (2.5D CNN)
# ─────────────────────────────────────────────────────────────
class NoduleStream(nn.Module):
    """
    Process 3D nodule patch as stack of 2D slices through EfficientNet-B0,
    then apply cross-slice attention + temporal 1D convolution.
    Input:  (B, 1, 64, 64, 64)
    Output: (B, 512)
    """

    def __init__(self, backbone_name="efficientnet_b0", feature_dim=512,
                 num_neighbors=2, ablation=None):
        super().__init__()
        self.ablation = ablation
        # 2D backbone (pretrained on ImageNet)
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, in_chans=1, num_classes=0
        )
        backbone_out = self.backbone.num_features  # e.g. 1280 for efficientnet_b0

        # Project backbone features to feature_dim
        self.proj = nn.Linear(backbone_out, feature_dim)

        # Cross-slice attention
        self.cross_attn = CrossSliceAttention(feature_dim, num_neighbors)

        # Temporal 1D convolution across slices
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Adaptive pooling to single vector
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (B, 1, D, H, W) – e.g. (B, 1, 64, 64, 64)
        Returns:
            features: (B, 512)
            attn_weights: (B, num_slices, num_slices)
        """
        B, C, D, H, W = x.shape

        # Reshape: treat depth as batch dim → (B*D, 1, H, W)
        slices = x.squeeze(1)  # (B, D, H, W)
        slices = slices.reshape(B * D, 1, H, W)  # (B*D, 1, H, W)

        # Forward through 2D backbone
        slice_feats = self.backbone(slices)  # (B*D, backbone_out)
        slice_feats = self.proj(slice_feats)  # (B*D, feature_dim)

        # Reshape back: (B, D, feature_dim)
        slice_feats = slice_feats.view(B, D, -1)

        # Cross-slice attention
        if self.ablation == 'no_attention':
            attended = slice_feats
            attn_weights = None
        else:
            attended, attn_weights = self.cross_attn(slice_feats)  # (B, D, feature_dim)

        # Temporal 1D conv: (B, feature_dim, D)
        temporal = attended.permute(0, 2, 1)
        temporal = self.temporal_conv(temporal)

        # Pool across slices → (B, feature_dim)
        features = self.pool(temporal).squeeze(-1)

        return features, attn_weights


# ─────────────────────────────────────────────────────────────
# Stream 2: Anatomical Context Extractor (Lightweight 3D CNN)
# ─────────────────────────────────────────────────────────────
class SpatialAttention3D(nn.Module):
    """Channel-wise spatial attention for 3D features."""

    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attn = self.conv(x)
        return x * attn


class ContextStream(nn.Module):
    """
    Lightweight 3D CNN to capture surrounding anatomy.
    Input:  (B, 1, 48, 48, 48)
    Output: (B, 256)
    """

    def __init__(self, feature_dim=256):
        super().__init__()
        # Scale internal channels based on output dim
        c1, c2, c3 = 64, 128, 256
        if feature_dim > 256:
            c1, c2, c3 = 64, 128, 512

        self.block1 = nn.Sequential(
            nn.Conv3d(1, c1, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c1),
            nn.ReLU(inplace=True),
        )  # 48→24

        self.block2 = nn.Sequential(
            nn.Conv3d(c1, c2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c2),
            nn.ReLU(inplace=True),
        )  # 24→12

        self.block3 = nn.Sequential(
            nn.Conv3d(c2, c3, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(c3),
            nn.ReLU(inplace=True),
        )  # 12→6

        self.spatial_attn = SpatialAttention3D(c3)
        self.gap = nn.AdaptiveAvgPool3d(1)  # → (B, c3, 1, 1, 1)
        self.fc = nn.Linear(c3, feature_dim)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 48, 48, 48)
        Returns:
            features: (B, feature_dim)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.spatial_attn(x)
        x = self.gap(x).flatten(1)  # (B, c3)
        x = self.fc(x)
        return x


# ─────────────────────────────────────────────────────────────
# Fusion Module (Multi-Head Attention)
# ─────────────────────────────────────────────────────────────
class FusionModule(nn.Module):
    """
    Fuse nodule and context features via multi-head attention.
    Input:  [B, 512] + [B, 256] → concatenated [B, 768]
    Output: [B, 256]
    """

    def __init__(self, nodule_dim=512, context_dim=256, fused_dim=256,
                 num_heads=4, dropout=0.5):
        super().__init__()
        total_dim = nodule_dim + context_dim  # 768
        # Project to a dimension divisible by num_heads
        self.proj_in = nn.Linear(total_dim, fused_dim * 2)
        self.attn = nn.MultiheadAttention(
            embed_dim=fused_dim * 2, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, nodule_feats, context_feats):
        """
        Args:
            nodule_feats: (B, 512)
            context_feats: (B, 256)
        Returns:
            fused: (B, 256)
        """
        combined = torch.cat([nodule_feats, context_feats], dim=-1)  # (B, 768)
        proj = self.proj_in(combined)  # (B, fused_dim*2)

        # Self-attention expects (B, seq_len, embed_dim) — treat as seq_len=1
        proj = proj.unsqueeze(1)  # (B, 1, fused_dim*2)
        attn_out, _ = self.attn(proj, proj, proj)  # (B, 1, fused_dim*2)
        attn_out = attn_out.squeeze(1)  # (B, fused_dim*2)

        fused = self.ffn(attn_out)  # (B, fused_dim)
        fused = self.norm(fused)
        return fused


# ─────────────────────────────────────────────────────────────
# Prediction Head
# ─────────────────────────────────────────────────────────────
class PredictionHead(nn.Module):
    """
    Dense 256 → 128 → 1 with BatchNorm, ReLU, Dropout.
    """

    def __init__(self, in_dim=256, hidden_dim=128, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.head(x)


# ─────────────────────────────────────────────────────────────
# Full DCA-Net Model
# ─────────────────────────────────────────────────────────────
class DCANet(nn.Module):
    """
    Dual-Context Attention Network for lung nodule classification.
    
    Inputs:
        nodule_patch:  (B, 1, 64, 64, 64)
        context_patch: (B, 1, 48, 48, 48)
    
    Outputs (training):
        logits: (B, 1)
    
    Outputs (uncertainty mode):
        mean_prob, confidence: (B,), (B,)
    """

    def __init__(self, config=None):
        super().__init__()

        # Parse config or use defaults
        if config is None:
            config = {}
        self.ablation = config.get('ablation', None)
        model_cfg = config.get('model', {})

        backbone = model_cfg.get('backbone', 'efficientnet_b0')
        nodule_dim = model_cfg.get('nodule_feature_dim', 512)
        context_dim = model_cfg.get('context_feature_dim', 256)
        fusion_dim = model_cfg.get('fusion_dim', 256)
        num_heads = model_cfg.get('num_attention_heads', 4)
        dropout = model_cfg.get('dropout', 0.5)
        pred_dropout = model_cfg.get('prediction_dropout', 0.3)
        num_neighbors = model_cfg.get('slice_neighbors', 2)
        self.mc_passes = model_cfg.get('mc_dropout_passes', 5)

        # Streams
        self.nodule_stream = NoduleStream(
            backbone_name=backbone, feature_dim=nodule_dim,
            num_neighbors=num_neighbors, ablation=self.ablation
        )
        self.context_stream = ContextStream(feature_dim=context_dim)

        # Fusion
        self.fusion = FusionModule(
            nodule_dim=nodule_dim, context_dim=context_dim,
            fused_dim=fusion_dim, num_heads=num_heads, dropout=dropout
        )

        # Prediction
        self.prediction_head = PredictionHead(
            in_dim=fusion_dim, hidden_dim=128, dropout=pred_dropout
        )

    def forward(self, nodule_patch, context_patch):
        """Standard forward pass (training mode).
        
        Args:
            nodule_patch:  (B, 1, 64, 64, 64)
            context_patch: (B, 1, 48, 48, 48)
            
        Returns:
            logits: (B, 1)
        """
        nodule_feats, attn_weights = self.nodule_stream(nodule_patch)
        context_feats = self.context_stream(context_patch)
        
        if self.ablation == 'no_context':
            context_feats = torch.zeros_like(context_feats)
            
        fused = self.fusion(nodule_feats, context_feats)
        logits = self.prediction_head(fused)
        return logits

    @torch.no_grad()
    def predict_with_uncertainty(self, nodule_patch, context_patch):
        """Monte Carlo Dropout uncertainty estimation.
        
        Runs multiple forward passes with dropout enabled,
        computes mean prediction and confidence.
        
        Args:
            nodule_patch:  (B, 1, 64, 64, 64)
            context_patch: (B, 1, 48, 48, 48)
            
        Returns:
            mean_prob: (B,) mean probability
            confidence: (B,) confidence score (1 - normalized variance)
        """
        # Enable dropout during inference unless ablation no_uncertainty
        if self.ablation != 'no_uncertainty':
            self.train()
        
        preds = []
        for _ in range(self.mc_passes):
            logits = self.forward(nodule_patch, context_patch)
            prob = torch.sigmoid(logits.squeeze(-1))
            preds.append(prob)
        
        preds = torch.stack(preds, dim=0)  # (mc_passes, B)
        mean_prob = preds.mean(dim=0)       # (B,)
        variance = preds.var(dim=0)         # (B,)
        
        # Confidence: 1 - normalized variance (variance is max 0.25 for Bernoulli)
        confidence = 1.0 - (variance / 0.25).clamp(0, 1)

        self.eval()
        return mean_prob, confidence

    def get_slice_importance(self, nodule_patch, context_patch):
        """Get per-slice importance scores from cross-slice attention.
        
        Returns:
            importance: (B, num_slices) attention-based importance
        """
        self.eval()
        with torch.no_grad():
            _, attn_weights = self.nodule_stream(nodule_patch)
            # Average attention received by each slice
            importance = attn_weights.mean(dim=1)  # (B, num_slices)
        return importance
