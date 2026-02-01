# EfficientNet Model

## Overview

The `EfficientNet` model wraps pretrained EfficientNet backbones from torchvision for seismic spectrogram classification. EfficientNet uses compound scaling to uniformly scale network width, depth, and resolution, achieving better accuracy and efficiency than previous CNN architectures.

## Motivation

EfficientNet addresses the challenge of scaling CNNs efficiently:

1. **Compound Scaling** - Uniformly scales width, depth, and resolution using a compound coefficient
2. **Neural Architecture Search** - Base architecture (B0) discovered via NAS optimization
3. **Efficiency** - Achieves state-of-the-art accuracy with fewer parameters and FLOPs
4. **Transfer Learning** - ImageNet pretrained weights transfer well to other domains

## Architecture

### High-Level Overview

```
Input Spectrogram: [B, C, H, W]
        |
        v
+------------------+
|    Stem Conv     |   3×3, stride=2
+------------------+
        |
        v
+------------------+
|   MBConv Blocks  |   Mobile Inverted Bottleneck blocks
|   (7 stages)     |   with Squeeze-and-Excitation
+------------------+
        |
        v
+------------------+
|    Head Conv     |   1×1 conv to expand channels
+------------------+
        |
        v
Global Average Pooling: [B, features]
        |
        v
Dropout
        |
        v
Linear Head: [B, num_classes]
```

### MBConv Block (Mobile Inverted Bottleneck)

The core building block of EfficientNet:

```
Input: x [B, C_in, H, W]
    |
    v
Expand: 1×1 conv → C_in × expansion_ratio
    |
    v
Depthwise: 3×3 or 5×5 depthwise conv (stride 1 or 2)
    |
    v
Squeeze-and-Excitation (SE)
    |
    v
Project: 1×1 conv → C_out
    |
    +-----> Residual Add (if stride=1 and C_in=C_out) <----- x
                |
                v
            Output [B, C_out, H', W']
```

**Key components:**
- **Inverted Residual**: Expands channels first, then projects back (opposite of ResNet)
- **Depthwise Separable Conv**: Reduces parameters while maintaining receptive field
- **Squeeze-and-Excitation**: Channel attention mechanism for recalibration

### Squeeze-and-Excitation (SE) Block

```
Input: x [B, C, H, W]
    |
    v
Global Average Pool: [B, C, 1, 1]
    |
    v
FC → ReLU → FC → Sigmoid: [B, C, 1, 1]  (squeeze ratio = 0.25)
    |
    v
Scale: x × attention: [B, C, H, W]
```

### EfficientNet-B0 Architecture

| Stage | Operator | Resolution | Channels | Layers |
|-------|----------|------------|----------|--------|
| 1 | Conv 3×3 | H/2 × W/2 | 32 | 1 |
| 2 | MBConv1, k3 | H/2 × W/2 | 16 | 1 |
| 3 | MBConv6, k3 | H/4 × W/4 | 24 | 2 |
| 4 | MBConv6, k5 | H/8 × W/8 | 40 | 2 |
| 5 | MBConv6, k3 | H/16 × W/16 | 80 | 3 |
| 6 | MBConv6, k5 | H/16 × W/16 | 112 | 3 |
| 7 | MBConv6, k5 | H/32 × W/32 | 192 | 4 |
| 8 | MBConv6, k3 | H/32 × W/32 | 320 | 1 |
| 9 | Conv 1×1 + Pool + FC | 1 × 1 | 1280 | 1 |

*MBConv{N} = Mobile Inverted Bottleneck with expansion ratio N, k{N} = kernel size N×N*

### Compound Scaling

EfficientNet scales the base model (B0) using compound coefficient φ:

- **Depth**: d = 1.2^φ (number of layers)
- **Width**: w = 1.1^φ (number of channels)
- **Resolution**: r = 1.15^φ (input image size)

| Model | φ | Depth | Width | Resolution | Parameters |
|-------|---|-------|-------|------------|------------|
| B0 | 0 | 1.0 | 1.0 | 224 | 5.3M |
| B1 | 0.5 | 1.1 | 1.0 | 240 | 7.8M |
| B2 | 1 | 1.2 | 1.1 | 260 | 9.2M |
| B3 | 2 | 1.4 | 1.2 | 300 | 12M |
| B4 | 3 | 1.8 | 1.4 | 380 | 19M |

### EfficientNetV2

EfficientNetV2 improves training speed with:

1. **Fused-MBConv** - Replaces depthwise + 1×1 with regular 3×3 conv in early layers
2. **Progressive Learning** - Gradually increases image size during training
3. **Adaptive Regularization** - Adjusts dropout/augmentation with image size

| Model | Parameters | Top-1 Accuracy |
|-------|------------|----------------|
| V2-S | 21M | 83.9% |
| V2-M | 54M | 85.1% |

## Usage

```python
from quake.models import EfficientNetBackbone, EfficientNet

model = EfficientNet(
    backbone=EfficientNetBackbone.B0,  # Pretrained backbone
    in_channels=3,                      # Z, N, E channels
    num_classes=2,                      # Number of output classes
    dropout=0.2,                        # Dropout before classifier
    freeze_backbone=False               # True for feature extraction
)
```

## Available Backbones

| Backbone | Parameters | ImageNet Top-1 | Notes |
|----------|------------|----------------|-------|
| `B0` | 5.3M | 77.1% | Smallest, fastest |
| `B1` | 7.8M | 79.1% | |
| `B2` | 9.2M | 80.1% | |
| `B3` | 12M | 81.6% | |
| `B4` | 19M | 82.9% | Good accuracy/speed trade-off |
| `V2_S` | 21M | 83.9% | Faster training than B4 |
| `V2_M` | 54M | 85.1% | Best accuracy |

**Tips:**
- Use `B0` or `B1` for small datasets to avoid overfitting
- Use `V2_S` for best training speed with good accuracy
- Use `freeze_backbone=True` for very small datasets, then fine-tune

## Input Requirements

- **Shape**: `[batch, channels, height, width]`
- **Type**: Float32 tensor
- **Preprocessing**: Z-score normalization recommended before STFT

For seismic data:
```python
from quake.data.transforms import STFTTransform, ToTensor

transform = nn.Sequential(
    STFTTransform(nperseg=64, hop=32),
    ToTensor(),
)
```

## Comparison with HybridViT

| Aspect | EfficientNet | HybridViT |
|--------|--------------|-----------|
| Architecture | Pure CNN | CNN stem + Transformer |
| Attention | Local (conv receptive field) + SE | Global (self-attention) |
| Parameters | 5.3M-54M | 1.5M-86M |
| Inductive bias | Strong (locality, translation equivariance) | Weak (learns from data) |
| Small datasets | Good with pretrained weights | Better with depth limiting |
| Training speed | Faster | Slower (attention is O(n²)) |

## References

1. **Tan, M., & Le, Q.** (2019).
   *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*.
   International Conference on Machine Learning (ICML).
   - arXiv: [https://arxiv.org/abs/1905.11946](https://arxiv.org/abs/1905.11946)

2. **Tan, M., & Le, Q.** (2021).
   *EfficientNetV2: Smaller Models and Faster Training*.
   International Conference on Machine Learning (ICML).
   - arXiv: [https://arxiv.org/abs/2104.00298](https://arxiv.org/abs/2104.00298)

3. **Sandler, M., Howard, A., Zhu, M., et al.** (2018).
   *MobileNetV2: Inverted Residuals and Linear Bottlenecks*.
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
   - arXiv: [https://arxiv.org/abs/1801.04381](https://arxiv.org/abs/1801.04381)

4. **Hu, J., Shen, L., & Sun, G.** (2018).
   *Squeeze-and-Excitation Networks*.
   IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).
   - arXiv: [https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)
