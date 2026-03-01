import torch
import torch.nn as nn
import torchvision.models as models

class ResNet3D18(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Use torchvision's 3D ResNet-18
        self.model = models.video.r3d_18(weights=None)
        
        # Modify first conv layer to accept 1 channel instead of 3
        old_conv = self.model.stem[0]
        self.model.stem[0] = nn.Conv3d(
            1, old_conv.out_channels, 
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Modify final fully connected layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, nodule_patch, context_patch=None):
        # We only use the nodule patch for the baseline
        # input shape: (B, 1, D, H, W)
        return self.model(nodule_patch)

class ResNet2D18SliceLevel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        # Use torchvision's 2D ResNet-18
        self.backbone = models.resnet18(weights=None)
        
        # Modify first conv to accept 1 channel instead of 3
        old_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        
        # Remove original fc layer, use GAP instead
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        
        # Final classification head after pooling slices
        self.head = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, nodule_patch, context_patch=None):
        # input shape: (B, 1, D, H, W)
        B, C, D, H, W = nodule_patch.shape
        
        # Reshape to treat depth as batch: (B*D, 1, H, W)
        slices = nodule_patch.squeeze(1).view(B*D, 1, H, W)
        
        # Extract features per slice: (B*D, num_ftrs)
        slice_feats = self.backbone(slices)
        
        # Reshape back to (B, D, num_ftrs)
        slice_feats = slice_feats.view(B, D, -1)
        
        # Global Average Pooling over the depth dimension (slices) -> (B, num_ftrs)
        pooled_feats = slice_feats.mean(dim=1)
        
        # Classification
        return self.head(pooled_feats)

def get_baseline_model(model_name):
    if model_name == 'resnet3d18':
        return ResNet3D18(num_classes=1)
    elif model_name == 'resnet2d18':
        return ResNet2D18SliceLevel(num_classes=1)
    else:
        raise ValueError(f"Unknown baseline model: {model_name}")
