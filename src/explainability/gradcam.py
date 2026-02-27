#!/usr/bin/env python3
"""
3D GradCAM for OncoVision-X Lung Nodule Classification.

Generates gradient-weighted class activation maps to visualize
which regions of the input the model focuses on for its predictions.

Usage:
    from src.explainability.gradcam import GradCAM3D

    gradcam = GradCAM3D(model, target_layer='nodule_stream')
    heatmap = gradcam(nodule_patch, context_patch)
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path


class GradCAM3D:
    """3D Gradient-weighted Class Activation Mapping.
    
    Generates heatmaps showing which spatial regions of the input
    volume contribute most to the model's prediction.
    
    Supports two target streams:
      - 'nodule_stream': Visualize nodule patch focus (64³)
      - 'context_stream': Visualize context patch focus (48³)
    """

    def __init__(self, model, target_stream='nodule_stream'):
        """
        Args:
            model: DCANet model (unwrapped from DataParallel)
            target_stream: 'nodule_stream' or 'context_stream'
        """
        self.model = model
        self.model.eval()
        self.target_stream = target_stream

        # Storage for hooks
        self.activations = None
        self.gradients = None

        # Register hooks on the target stream's last conv layer
        if target_stream == 'nodule_stream':
            # Hook into the backbone's final feature extraction
            target_layer = self._get_nodule_target()
        elif target_stream == 'context_stream':
            target_layer = self._get_context_target()
        else:
            raise ValueError(f"Unknown target_stream: {target_stream}")

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _get_nodule_target(self):
        """Get the last convolutional layer of the nodule stream backbone."""
        backbone = self.model.nodule_stream.backbone
        # For EfficientNet, the last conv features are in the final block
        # Use the conv_head or the last conv layer
        if hasattr(backbone, 'conv_head'):
            return backbone.conv_head
        elif hasattr(backbone, 'blocks'):
            # Last block's last conv
            return backbone.blocks[-1]
        else:
            # Fallback: last named module with Conv2d
            last_conv = None
            for module in backbone.modules():
                if isinstance(module, torch.nn.Conv2d):
                    last_conv = module
            if last_conv is None:
                raise RuntimeError("Could not find target layer in nodule backbone")
            return last_conv

    def _get_context_target(self):
        """Get the last conv block of the context stream."""
        return self.model.context_stream.block3

    def _save_activation(self, module, input, output):
        """Forward hook to save activations."""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook to save gradients."""
        self.gradients = grad_output[0].detach()

    def generate(self, nodule_patch, context_patch, device='cpu'):
        """Generate GradCAM heatmap.
        
        Args:
            nodule_patch: numpy array (64, 64, 64) or (1, 1, 64, 64, 64)
            context_patch: numpy array (48, 48, 48) or (1, 1, 48, 48, 48)
            device: torch device
            
        Returns:
            heatmap: numpy array, same spatial dims as target stream input
            probability: model prediction probability
        """
        # Ensure tensor format
        if isinstance(nodule_patch, np.ndarray):
            if nodule_patch.ndim == 3:
                nodule_patch = nodule_patch[np.newaxis, np.newaxis, ...]
            nodule_patch = torch.from_numpy(nodule_patch.astype(np.float32))
        if isinstance(context_patch, np.ndarray):
            if context_patch.ndim == 3:
                context_patch = context_patch[np.newaxis, np.newaxis, ...]
            context_patch = torch.from_numpy(context_patch.astype(np.float32))

        nodule_patch = nodule_patch.to(device).requires_grad_(True)
        context_patch = context_patch.to(device).requires_grad_(True)

        # Forward pass
        self.model.zero_grad()
        logits = self.model(nodule_patch, context_patch)
        prob = torch.sigmoid(logits.squeeze()).item()

        # Backward pass — gradient w.r.t. the class score
        logits.squeeze().backward()

        # Compute GradCAM
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not capture gradients/activations")

        gradients = self.gradients
        activations = self.activations

        # For nodule stream: activations are 2D per-slice (B*D, C, H, W)
        # For context stream: activations are 3D (B, C, D, H, W)
        if self.target_stream == 'nodule_stream':
            # Global average pooling of gradients over spatial dims
            weights = gradients.mean(dim=(-2, -1), keepdim=True)  # (N, C, 1, 1)
            cam = (weights * activations).sum(dim=1)  # (N, H, W)
            cam = F.relu(cam)

            # Reshape to 3D: N slices → (D, H, W)
            D = nodule_patch.shape[2]  # depth
            num_per_slice = cam.shape[0] // nodule_patch.shape[0]
            cam = cam[:num_per_slice]  # Take first batch item

            # Resize each slice to original spatial size
            target_h, target_w = nodule_patch.shape[3], nodule_patch.shape[4]
            cam_resized = F.interpolate(
                cam.unsqueeze(1),  # (D, 1, h, w)
                size=(target_h, target_w),
                mode='bilinear', align_corners=False
            ).squeeze(1)  # (D, H, W)

            heatmap = cam_resized.cpu().numpy()

        else:
            # Context stream: 3D activations
            weights = gradients.mean(dim=(-3, -2, -1), keepdim=True)
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            target_size = context_patch.shape[2:]
            cam = F.interpolate(cam, size=target_size, mode='trilinear', align_corners=False)
            heatmap = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

        return heatmap, prob

    def __call__(self, nodule_patch, context_patch, device='cpu'):
        return self.generate(nodule_patch, context_patch, device)


def plot_gradcam_slices(scan_patch, heatmap, probability, output_path,
                        num_slices=8, title="GradCAM Visualization"):
    """Plot GradCAM overlay on selected slices.
    
    Args:
        scan_patch: 3D numpy array (D, H, W) — original input
        heatmap: 3D numpy array (D, H, W) — GradCAM heatmap
        probability: float — model prediction
        output_path: str — save path
        num_slices: int — number of slices to display
        title: str — plot title
    """
    D = scan_patch.shape[0]
    slice_indices = np.linspace(D // 8, D - D // 8, num_slices, dtype=int)

    fig, axes = plt.subplots(2, num_slices, figsize=(3 * num_slices, 7))

    label = "MALIGNANT" if probability > 0.5 else "BENIGN"
    color = 'red' if probability > 0.5 else 'green'
    fig.suptitle(
        f"{title}\nPrediction: {label} ({probability:.1%})",
        fontsize=14, fontweight='bold', color=color
    )

    for i, idx in enumerate(slice_indices):
        # Original slice
        axes[0, i].imshow(scan_patch[idx], cmap='gray', vmin=-1, vmax=1)
        axes[0, i].set_title(f"Slice {idx}", fontsize=9)
        axes[0, i].axis('off')

        # Overlay
        axes[1, i].imshow(scan_patch[idx], cmap='gray', vmin=-1, vmax=1)
        overlay = axes[1, i].imshow(
            heatmap[idx], cmap='jet', alpha=0.5, vmin=0, vmax=1
        )
        axes[1, i].set_title(f"GradCAM", fontsize=9)
        axes[1, i].axis('off')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.35])
    fig.colorbar(overlay, cax=cbar_ax, label='Attention')

    plt.tight_layout(rect=[0, 0, 0.9, 0.92])
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path


def generate_gradcam_report(model, dataloader, device, output_dir,
                             num_samples=10, stream='nodule_stream'):
    """Generate GradCAM visualizations for multiple samples.
    
    Args:
        model: Trained DCANet model
        dataloader: Test DataLoader
        device: torch device
        output_dir: Directory to save GradCAM plots
        num_samples: Number of samples to visualize
        stream: Which stream to visualize
        
    Returns:
        List of output paths
    """
    import torch.nn as nn
    
    # Unwrap DataParallel if needed
    if isinstance(model, nn.DataParallel):
        raw_model = model.module
    else:
        raw_model = model

    gradcam = GradCAM3D(raw_model, target_stream=stream)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    outputs = []
    sample_count = 0

    for nodule, context, labels in dataloader:
        for i in range(nodule.shape[0]):
            if sample_count >= num_samples:
                return outputs

            nod = nodule[i:i+1]
            ctx = context[i:i+1]
            label = labels[i].item()

            try:
                heatmap, prob = gradcam.generate(nod, ctx, device)
            except Exception as e:
                print(f"  GradCAM failed for sample {sample_count}: {e}")
                continue

            # Get the original scan patch for overlay
            scan_slice = nod.squeeze().numpy()

            # Crop heatmap to match scan if needed
            if heatmap.shape != scan_slice.shape:
                min_d = min(heatmap.shape[0], scan_slice.shape[0])
                heatmap = heatmap[:min_d]
                scan_slice = scan_slice[:min_d]

            gt_str = "pos" if label == 1 else "neg"
            pred_str = "malignant" if prob > 0.5 else "benign"
            correct = (label == 1 and prob > 0.5) or (label == 0 and prob <= 0.5)
            
            out_path = output_dir / f"gradcam_{sample_count:03d}_{gt_str}_pred_{pred_str}.png"
            plot_gradcam_slices(
                scan_slice, heatmap, prob, str(out_path),
                title=f"Sample {sample_count} — GT: {'Cancer' if label==1 else 'Benign'} | "
                      f"{'Correct' if correct else 'WRONG'}"
            )
            outputs.append(str(out_path))
            sample_count += 1

    return outputs
