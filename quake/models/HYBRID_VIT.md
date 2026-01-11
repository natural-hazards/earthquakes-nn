# Hybrid CNN-ViT Model

## Overview

The `HybridViT` model implements a hybrid architecture that combines a Convolutional Neural Network (CNN) stem with a pretrained Vision Transformer (ViT) backbone from `timm`. This approach addresses the optimization instability issues found in the original ViT architecture while maintaining its powerful global attention capabilities.

## Motivation

The original Vision Transformer (ViT) uses a "patchify" stem implemented as a single large-kernel, large-stride convolution (typically 16x16 with stride 16). This design choice deviates from established CNN best practices and leads to:

1. **Optimization instability** - ViT models are highly sensitive to optimizer choice (requiring AdamW over SGD), hyperparameters, and training schedule length
2. **Slower convergence** - Requires longer training schedules to achieve peak performance
3. **Poor performance on small datasets** - Lacks the inductive biases that help CNNs generalize from limited data

## Architecture

### High-Level Overview

```
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
```

### Detailed Shape Transformations

Example with `Backbone.TINY`, `img_size=(33, 64)`, `depth=4`:

| Stage | Operation | Output Shape | Notes |
|-------|-----------|--------------|-------|
| Input | - | [B, 3, 33, 64] | Z, N, E channels |
| Conv1 | 7×7, stride=2, pad=3 | [B, 64, 17, 32] | (33+6-7)/2+1=17 |
| Conv2 | 3×3, stride=2, pad=1 | [B, 128, 9, 16] | (17+2-3)/2+1=9 |
| Conv3 | 3×3, stride=2, pad=1 | [B, 192, 5, 8] | (9+2-3)/2+1=5 |
| Flatten | - | [B, 192, 40] | 5×8=40 patches |
| Transpose | - | [B, 40, 192] | Sequence format |
| + CLS | Concatenate | [B, 41, 192] | +1 for CLS token |
| + Pos | Add | [B, 41, 192] | Same shape |
| Transformer | 4 blocks | [B, 41, 192] | Self-attention |
| LayerNorm | - | [B, 41, 192] | Normalize |
| Extract CLS | Index [:, 0] | [B, 192] | First token |
| Head | Linear | [B, 2] | Class logits |

### CNN Stem

The CNN stem replaces ViT's single 16×16 convolution with three smaller convolutions:

```
Input: [B, in_channels, H, W]
   |
Conv1: 7×7, stride=2, padding=3 → [B, 64, H/2, W/2]
BatchNorm2d(64) + ReLU
   |
Conv2: 3×3, stride=2, padding=1 → [B, 128, H/4, W/4]
BatchNorm2d(128) + ReLU
   |
Conv3: 3×3, stride=2, padding=1 → [B, embed_dim, H/8, W/8]
BatchNorm2d(embed_dim)  # No ReLU - transformer handles nonlinearity
```

**Design choices:**
- **Large 7×7 kernel first**: Captures low-frequency patterns early
- **Gradual channel expansion**: in → 64 → 128 → embed_dim
- **BatchNorm**: Stabilizes training, enables higher learning rates
- **No bias in conv**: BatchNorm handles mean shift
- **No final ReLU**: Let transformer's GELU handle nonlinearity

### CLS Token

The CLS (classification) token is a learnable vector `[1, 1, embed_dim]` that:

1. Gets prepended to the patch sequence before the transformer
2. Has no corresponding input patch - it's purely learned
3. Attends to all patches via self-attention
4. Aggregates global information across the entire input
5. Gets extracted after the transformer for classification

```python
# Before transformer: [B, num_patches, embed_dim]
x = torch.cat([cls_token.expand(B, -1, -1), x], dim=1)
# After: [B, num_patches+1, embed_dim]

# After transformer, extract CLS for classification
output = head(x[:, 0])  # x[:, 0] is the CLS token
```

### Positional Embeddings

Transformers are permutation-invariant - they don't inherently know the order/position of patches. Positional embeddings add this information:

- **Shape**: `[1, num_patches+1, embed_dim]` (one position per patch + CLS)
- **Learnable**: Trained end-to-end with the model
- **Pretrained**: We use embeddings from ImageNet-trained ViT

**Interpolation**: Pretrained ViT uses 14×14=196 patches. Our patch count differs, so we interpolate:

```
1. Separate CLS position from patch positions
2. Reshape patches to 2D: [1, 196, D] → [1, 14, 14, D]
3. Bicubic interpolate: [1, 14, 14, D] → [1, H', W', D]
4. Flatten: [1, H', W', D] → [1, H'×W', D]
5. Concatenate CLS back: [1, 1+H'×W', D]
```

### Transformer Block

Each transformer block (from pretrained ViT) contains:

```
Input: x [B, N, D]
    |
    v
LayerNorm
    |
    v
Multi-Head Self-Attention (MHSA)
    |
    +-----> Residual Add <----- x
                |
                v
            LayerNorm
                |
                v
    MLP: Linear(D→4D) → GELU → Linear(4D→D)
                |
    +-----> Residual Add <----- (from above)
                |
                v
            Output [B, N, D]
```

**Multi-Head Self-Attention**:
- Projects input to Q (query), K (key), V (value)
- Computes attention: `softmax(QK^T / √d) × V`
- Multiple heads learn different attention patterns
- TINY: 3 heads, SMALL: 6 heads, BASE: 12 heads

**MLP**:
- Expands dimension 4× (e.g., 192 → 768)
- GELU activation (smooth ReLU)
- Projects back to embed_dim

### Depth Selection

We can use only the first N transformer blocks:

```python
model = HybridViT(backbone=Backbone.TINY, depth=4)  # Use blocks 0-3
model = HybridViT(backbone=Backbone.TINY, depth=None)  # Use all 12
```

**Why use fewer blocks?**
- Early layers learn general features (edges, textures) - transfer well
- Later layers learn task-specific features (ImageNet objects) - less useful for seismic
- Fewer parameters → less overfitting on small datasets

## Benefits of Convolutional Stem

According to Xiao et al. (2021), replacing the patchify stem with a convolutional stem provides:

1. **Improved optimization stability** - Models train well with both AdamW and SGD
2. **Better peak performance** - +1-2% top-1 accuracy on ImageNet
3. **Faster convergence** - Achieves good results with shorter training schedules
4. **Better generalization on small datasets** - CNN stem provides useful inductive biases

## Usage

```python
from quake.models import Backbone, HybridViT

model = HybridViT(
    backbone=Backbone.TINY,
    depth=4,                 # Use first 4 pretrained blocks (of 12)
    in_channels=3,           # Z, N, E channels
    num_classes=2,
    img_size=(33, 64),       # (freq_bins, time_frames)
    dropout=0.2,
    freeze_backbone=False    # True for feature extraction only
)
```

## Available Backbones

| Backbone | timm Model | Embed Dim | Heads | Params (depth=4) | Params (depth=12) |
|----------|-----------|-----------|-------|------------------|-------------------|
| TINY | vit_tiny_patch16_224 | 192 | 3 | ~1.5M | ~5.7M |
| SMALL | vit_small_patch16_224 | 384 | 6 | ~6M | ~22M |
| BASE | vit_base_patch16_224 | 768 | 12 | ~24M | ~86M |
| DEIT_TINY | deit_tiny_patch16_224 | 192 | 3 | ~1.5M | ~5.7M |
| DEIT_SMALL | deit_small_patch16_224 | 384 | 6 | ~6M | ~22M |

**Tips:**

- Use `depth=4` for small datasets (~10K-50K samples) to reduce overfitting
- Use `freeze_backbone=True` initially, then fine-tune with lower learning rate
- Early transformer layers transfer better across domains (ImageNet → seismic)

## Input Requirements

- **Shape**: `[batch, channels, height, width]`
- **Type**: Float32 tensor
- **Preprocessing**: Recommended to use z-score normalization before STFT

For seismic data, use the `SPECTROGRAM` transform:
```python
from quake.data.adapter import WaveformDataAdapter, TransformOP

adapter = WaveformDataAdapter(
    events=events,
    labels=labels,
    stft_nperseg=64,
    stft_hop=32,
    transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.SPECTROGRAM
)
```

## References

1. **Xiao, T., Singh, M., Mintun, E., Darrell, T., Dollár, P., & Girshick, R.** (2021).
   *Early Convolutions Help Transformers See Better*.
   Advances in Neural Information Processing Systems (NeurIPS), 34.
   - arXiv: [https://arxiv.org/abs/2106.14881](https://arxiv.org/abs/2106.14881)
   - PDF: [NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/file/ff1418e8cc993fe8abcfe3ce2003e5c5-Paper.pdf)

2. **Dosovitskiy, A., Beyer, L., Kolesnikov, A., et al.** (2020).
   *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*.
   International Conference on Learning Representations (ICLR).
   - arXiv: [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

3. **Guo, J., Han, K., Wu, H., et al.** (2022).
   *CMT: Convolutional Neural Networks Meet Vision Transformers*.
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
   - PDF: [CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Guo_CMT_Convolutional_Neural_Networks_Meet_Vision_Transformers_CVPR_2022_paper.pdf)

4. **Touvron, H., Cord, M., Douze, M., et al.** (2021).
   *Training data-efficient image transformers & distillation through attention*.
   International Conference on Machine Learning (ICML).
   - arXiv: [https://arxiv.org/abs/2012.12877](https://arxiv.org/abs/2012.12877)
