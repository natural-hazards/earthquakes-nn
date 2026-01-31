"""
EfficientNet for Seismic Spectrogram Classification
====================================================

Pretrained EfficientNet backbone from torchvision adapted for seismic classification.

Architecture:
    Input Spectrogram: [B, C, H, W]
            |
            v
    EfficientNet Backbone (pretrained on ImageNet)
        - Stem: 3x3 conv, stride 2
        - MBConv blocks with squeeze-excitation
        - Inverted residual connections
        - Compound scaling (width, depth, resolution)
            |
            v
    Global Average Pooling: [B, features]
            |
            v
    Dropout
            |
            v
    Linear Head: [B, num_classes]

References:
    - Tan & Le (2019): "EfficientNet: Rethinking Model Scaling for CNNs"
    - Tan & Le (2021): "EfficientNetV2: Smaller Models and Faster Training"
"""

import torch
import torch.nn as nn
from enum import StrEnum

from torchvision.models import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3, efficientnet_b4,
    efficientnet_v2_s, efficientnet_v2_m,
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights,
    EfficientNet_V2_S_Weights, EfficientNet_V2_M_Weights
)


__all__ = ['EfficientNetBackbone', 'EfficientNet']


class EfficientNetBackbone(StrEnum):
    """Pretrained EfficientNet backbones from torchvision."""
    B0 = 'b0'       # 5.3M params
    B1 = 'b1'       # 7.8M params
    B2 = 'b2'       # 9.2M params
    B3 = 'b3'       # 12M params
    B4 = 'b4'       # 19M params
    V2_S = 'v2_s'   # 21M params
    V2_M = 'v2_m'   # 54M params


_BACKBONE_FACTORY = {
    EfficientNetBackbone.B0: (efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.B1: (efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.B2: (efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.B3: (efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.B4: (efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.V2_S: (efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    EfficientNetBackbone.V2_M: (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
}


class EfficientNet(nn.Module):
    """EfficientNet model for seismic spectrogram classification.

    Uses pretrained EfficientNet backbone from torchvision with adapted input
    channels and classification head.

    Args:
        backbone: EfficientNet variant to use
        in_channels: Input channels (e.g., 3 for Z/N/E seismic components)
        num_classes: Number of output classes
        dropout: Dropout before classification head
        freeze_backbone: If True, freeze backbone weights (feature extraction mode)
    """

    def __init__(
        self,
        backbone: EfficientNetBackbone = EfficientNetBackbone.B0,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Load pretrained EfficientNet
        factory_fn, weights = _BACKBONE_FACTORY[backbone]
        model = factory_fn(weights=weights)

        # Extract features part (everything except classifier)
        self.features = model.features
        self.avgpool = model.avgpool

        # Adapt first conv layer if in_channels != 3
        if in_channels != 3:
            first_conv = self.features[0][0]
            self.features[0][0] = nn.Conv2d(
                in_channels,
                first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
                bias=first_conv.bias is not None
            )

        # Classification head
        num_features = model.classifier[1].in_features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout, inplace=True),
            nn.Linear(num_features, num_classes)
        )

        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Class logits [batch, num_classes]
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
