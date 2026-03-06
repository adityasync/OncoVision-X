#!/usr/bin/env python3
"""
Loss functions for DCA-Net training.

Combined loss: α·BCE + β·Focal + γ·Uncertainty
  - BCE:         Standard binary cross-entropy
  - Focal:       Focus on hard examples (class imbalance)
  - Uncertainty: Penalize overconfident wrong predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    FL(p_t) = -α_t (1 - p_t)^γ log(p_t)
    """

    def __init__(self, alpha=0.75, gamma=2.0, eps=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw model output
            targets: (B,) binary labels
        """
        # Calculate BCE loss per-element securely
        bce = F.binary_cross_entropy_with_logits(logits.squeeze(-1), targets, reduction='none')
        
        # Calculate probabilities from logits and apply epsilon clamping (CRITICAL for NaN prevention)
        probs = torch.sigmoid(logits.squeeze(-1))
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        
        # Apply focal weight to BCE loss
        loss = focal_weight * bce
        return loss.mean()


class UncertaintyLoss(nn.Module):
    """Penalize overconfident wrong predictions.
    
    When the model is confident but wrong, apply extra penalty.
    When the model is uncertain, reduce penalty (it "knows it doesn't know").
    """

    def __init__(self, eps=1e-7):
        super().__init__()
        self.eps = eps

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw model output
            targets: (B,) binary labels
        """
        probs = torch.sigmoid(logits.squeeze(-1))
        # Clamp probs (CRITICAL)
        probs = torch.clamp(probs, self.eps, 1.0 - self.eps)
        
        # Confidence: how far from 0.5 (max uncertainty)
        confidence = (probs - 0.5).abs() * 2  # [0, 1]
        
        # Correctness: 1 if prediction matches target, 0 otherwise
        predicted = (probs > 0.5).float()
        correct = (predicted == targets).float()
        
        # Penalize: high confidence + wrong prediction
        # Reward: low confidence when wrong (model knows it's unsure)
        penalty = confidence * (1 - correct)
        
        return penalty.mean()


class DCANetLoss(nn.Module):
    """Combined loss for DCA-Net: α·BCE + β·Focal + γ·Uncertainty.
    
    Args:
        bce_weight: Weight for BCE loss (α)
        focal_weight: Weight for Focal loss (β)
        uncertainty_weight: Weight for Uncertainty loss (γ)
        focal_gamma: Focal loss gamma parameter
        focal_alpha: Focal loss alpha parameter
        label_smoothing: Label smoothing factor
    """

    def __init__(self, bce_weight=0.4, focal_weight=0.4, uncertainty_weight=0.2,
                 focal_gamma=2.0, focal_alpha=0.75, label_smoothing=0.1, pos_weight=1.0, eps=1e-7):
        super().__init__()
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.uncertainty_weight = uncertainty_weight
        self.label_smoothing = label_smoothing
        self.eps = eps
        self.register_buffer('pos_weight', torch.tensor([pos_weight]))

        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)
        self.uncertainty_loss = UncertaintyLoss(eps=eps)

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, 1) raw model output
            targets: (B,) binary labels
            
        Returns:
            total_loss: scalar
            loss_dict: dict with individual losses for logging
        """
        # CLAMP logits heavily to prevent any NaNs before BCE
        # A logit of +/- 15 is extremely confident (prob = 0.999999 or 0.000001)
        # Anything beyond that risks exp() overflow/underflow
        logits = torch.clamp(logits, min=-15.0, max=15.0)

        # Apply label smoothing
        smooth_targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # BCE loss with pos_weight
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), smooth_targets,
            pos_weight=self.pos_weight.to(targets.device)
        )

        # Focal loss (uses original targets for p_t computation)
        focal = self.focal_loss(logits, targets)

        # Uncertainty loss
        uncertainty = self.uncertainty_loss(logits, targets)

        # Weighted combination
        total = (self.bce_weight * bce +
                 self.focal_weight * focal +
                 self.uncertainty_weight * uncertainty)

        # NaN safety: if total is NaN, fall back to BCE only
        if torch.isnan(total):
            print("WARNING: NaN detected in loss, using BCE only")
            total = bce
            if torch.isnan(bce):
                 total = torch.tensor(0.0, device=logits.device, requires_grad=True)

        loss_dict = {
            'total': total.item(),
            'bce': bce.item(),
            'focal': focal.item(),
            'uncertainty': uncertainty.item(),
        }

        return total, loss_dict
