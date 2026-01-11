"""
Hybrid CNN-ViT Model for Seismic Classification
================================================

This module implements a hybrid architecture combining a CNN stem with a pretrained
Vision Transformer (ViT) backbone. The CNN stem replaces ViT's "patchify" operation
(single 16x16 conv) with multiple smaller convolutions, providing better inductive
bias for small datasets.

Architecture Overview
---------------------

    Input Spectrogram: [B, C, H, W]
            |
            v
    +------------------+
    |    CNN Stem      |   3 conv layers, each with stride=2
    |                  |   Total downsampling: 8x
    +------------------+
            |
            v
    Feature Maps: [B, embed_dim, H/8, W/8]
            |
            v
    Flatten + Transpose: [B, num_patches, embed_dim]
            |
            v
    Prepend CLS Token: [B, num_patches+1, embed_dim]
            |
            v
    Add Positional Embeddings (interpolated from pretrained)
            |
            v
    +------------------+
    | Transformer      |   N blocks of:
    | Encoder          |   - Multi-Head Self-Attention (MHSA)
    | (pretrained)     |   - Feed-Forward Network (MLP)
    +------------------+
            |
            v
    Layer Norm
            |
            v
    Extract CLS Token: [B, embed_dim]
            |
            v
    Linear Head: [B, num_classes]


Key Components
--------------

1. **CNN Stem**: Replaces ViT's single large-kernel conv with 3 smaller convs.
   This provides translation equivariance and local feature extraction that
   helps the model generalize better on small datasets.

2. **CLS Token**: A learnable [1, 1, embed_dim] vector prepended to patch sequence.
   It attends to all patches and aggregates global information for classification.

3. **Positional Embeddings**: Since transformers are permutation-invariant, we add
   learnable position information. Pretrained embeddings are bicubic-interpolated
   to match our patch grid size.

4. **Transformer Blocks**: Each block applies:
   - LayerNorm → Multi-Head Self-Attention → Residual Add
   - LayerNorm → MLP (expand 4x → GELU → contract) → Residual Add

5. **Depth Selection**: We can use only the first N blocks of the pretrained
   transformer. Early layers learn general features that transfer well across
   domains, while later layers are more task-specific.

References
----------
- Xiao et al. (2021): "Early Convolutions Help Transformers See Better"
- Dosovitskiy et al. (2020): "An Image is Worth 16x16 Words"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import StrEnum

import timm
from timm.layers import trunc_normal_


__all__ = [
    'Backbone',
    'HybridViT'
]


class Backbone(StrEnum):
    """Pretrained ViT backbones from timm library."""
    TINY = 'vit_tiny_patch16_224'       # embed_dim=192, depth=12, heads=3
    SMALL = 'vit_small_patch16_224'     # embed_dim=384, depth=12, heads=6
    BASE = 'vit_base_patch16_224'       # embed_dim=768, depth=12, heads=12
    DEIT_TINY = 'deit_tiny_patch16_224' # embed_dim=192, depth=12, heads=3 (distilled)
    DEIT_SMALL = 'deit_small_patch16_224' # embed_dim=384, depth=12, heads=6 (distilled)


def _conv_output_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Calculate spatial dimension after convolution: (size + 2*padding - kernel) // stride + 1"""
    return (size + 2 * padding - kernel) // stride + 1


def _stem_output_size(h: int, w: int) -> tuple[int, int]:
    """Calculate output size after CNN stem.

    Three conv layers with stride=2 each → 8x downsampling total.
    Conv1: kernel=7, padding=3 → preserves size before stride
    Conv2: kernel=3, padding=1 → preserves size before stride
    Conv3: kernel=3, padding=1 → preserves size before stride
    """
    for kernel, padding in [(7, 3), (3, 1), (3, 1)]:
        h = _conv_output_size(h, kernel, 2, padding)
        w = _conv_output_size(w, kernel, 2, padding)
    return h, w


class CNNStem(nn.Module):
    """CNN stem replacing ViT's patchify operation.

    Three convolutional layers progressively extract features:

        Input: [B, in_channels, H, W]
           |
        Conv1: 7x7, stride=2 → [B, 64, H/2, W/2]
        BatchNorm + ReLU
           |
        Conv2: 3x3, stride=2 → [B, 128, H/4, W/4]
        BatchNorm + ReLU
           |
        Conv3: 3x3, stride=2 → [B, embed_dim, H/8, W/8]
        BatchNorm (no ReLU - let transformer handle nonlinearity)

    Why this design:
    - Large 7x7 kernel first captures low-frequency patterns
    - Gradual channel expansion (in→64→128→embed_dim)
    - BatchNorm stabilizes training
    - No bias in conv (BatchNorm handles mean shift)
    """

    def __init__(self, in_channels: int, embed_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class HybridViT(nn.Module):
    """Hybrid CNN-ViT model for seismic classification.

    Combines CNN stem (for local features) with pretrained ViT transformer
    (for global attention). This hybrid approach works better than pure ViT
    on small datasets.

    Args:
        backbone: Pretrained ViT model from timm (determines embed_dim)
        in_channels: Input channels (e.g., 3 for Z/N/E seismic components)
        num_classes: Number of output classes
        img_size: Input (height, width) - used to compute patch grid size
        depth: Number of transformer blocks to use (None = all 12)
        dropout: Dropout after positional embeddings
        freeze_backbone: If True, freeze transformer weights (feature extraction mode)

    Forward pass shapes (example with TINY backbone, img_size=(33, 64), depth=4):
        Input:           [B, 3, 33, 64]
        After stem:      [B, 192, 4, 8]      # 33→4 (8x down), 64→8 (8x down)
        After flatten:   [B, 32, 192]        # 4*8=32 patches
        After CLS:       [B, 33, 192]        # +1 for CLS token
        After pos_embed: [B, 33, 192]        # same shape, values changed
        After blocks:    [B, 33, 192]        # same shape
        After norm:      [B, 33, 192]        # same shape
        CLS extract:     [B, 192]            # take first token
        Output:          [B, num_classes]    # linear projection
    """

    def __init__(
        self,
        backbone: Backbone = Backbone.TINY,
        in_channels: int = 3,
        num_classes: int = 2,
        img_size: tuple[int, int] = (64, 128),
        depth: int | None = None,
        dropout: float = 0.2,
        freeze_backbone: bool = False
    ):
        super().__init__()

        # Calculate patch grid dimensions after CNN stem (8x downsampling)
        self.patch_h, self.patch_w = _stem_output_size(img_size[0], img_size[1])
        self.num_patches = self.patch_h * self.patch_w

        # Load pretrained ViT backbone from timm
        backbone = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.embed_dim = backbone.embed_dim

        # CNN stem: [B, in_channels, H, W] → [B, embed_dim, H/8, W/8]
        self.stem = CNNStem(in_channels, self.embed_dim)

        # CLS token: learnable vector that aggregates global information
        # Shape: [1, 1, embed_dim] → expanded to [B, 1, embed_dim] in forward
        self.cls_token = nn.Parameter(backbone.cls_token.clone())

        # Positional embeddings: interpolated from pretrained (224x224 → our size)
        # Shape: [1, num_patches+1, embed_dim] (+1 for CLS token position)
        self.pos_embed = nn.Parameter(self._interpolate_pos_embed(backbone.pos_embed))
        self.pos_dropout = nn.Dropout(dropout)

        # Transformer blocks: use first `depth` blocks (or all if depth=None)
        # Each block: LayerNorm → MHSA → Add → LayerNorm → MLP → Add
        blocks = list(backbone.blocks.children())
        self.blocks = nn.Sequential(*(blocks[:depth] if depth else blocks))

        # Final layer norm (applied after all transformer blocks)
        self.norm = backbone.norm

        # Classification head: project CLS token to class logits
        self.head = nn.Linear(self.embed_dim, num_classes)

        # Initialize head weights (truncated normal, std=0.02 is ViT convention)
        trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        # Optionally freeze transformer for feature extraction
        if freeze_backbone:
            for param in self.blocks.parameters():
                param.requires_grad = False

    def _interpolate_pos_embed(self, pos_embed: torch.Tensor) -> torch.Tensor:
        """Interpolate pretrained positional embeddings to match our patch grid.

        Pretrained ViT uses 14x14=196 patches (from 224x224 image with 16x16 patches).
        Our patch count depends on img_size and CNN stem downsampling.

        Process:
        1. Separate CLS token position (index 0) from patch positions
        2. Reshape patch positions to 2D grid: [1, 196, D] → [1, 14, 14, D]
        3. Bicubic interpolate to our grid size: [1, 14, 14, D] → [1, H', W', D]
        4. Flatten back: [1, H', W', D] → [1, H'*W', D]
        5. Concatenate CLS position back: [1, 1+H'*W', D]
        """
        cls_pos, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]

        old_num = patch_pos.shape[1]
        if old_num == self.num_patches:
            return pos_embed  # No interpolation needed

        # Reshape to 2D grid for spatial interpolation
        old_size = int(old_num ** 0.5)  # 196 → 14
        patch_pos = patch_pos.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)

        # Bicubic interpolation to new grid size
        patch_pos = F.interpolate(
            patch_pos,
            size=(self.patch_h, self.patch_w),
            mode='bicubic',
            align_corners=False
        )

        # Reshape back to sequence
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, self.num_patches, -1)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor [batch, channels, height, width]

        Returns:
            Class logits [batch, num_classes]
        """
        B = x.shape[0]

        # CNN stem: extract local features, downsample 8x
        # [B, C, H, W] → [B, embed_dim, H/8, W/8]
        x = self.stem(x)

        # Flatten spatial dims to sequence of patches
        # [B, embed_dim, H', W'] → [B, embed_dim, H'*W'] → [B, H'*W', embed_dim]
        x = x.flatten(2).transpose(1, 2)

        # Prepend CLS token to sequence
        # [B, num_patches, embed_dim] → [B, num_patches+1, embed_dim]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)

        # Add positional embeddings (broadcast over batch)
        x = self.pos_dropout(x + self.pos_embed)

        # Transformer blocks: self-attention + MLP
        x = self.blocks(x)

        # Final layer norm
        x = self.norm(x)

        # Classification: extract CLS token (index 0) and project to classes
        return self.head(x[:, 0])
