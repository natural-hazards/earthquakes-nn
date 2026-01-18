import numpy as np
import torch.nn as nn


__all__ = [
    'TimeShift',
    'AddNoise',
    'AmplitudeScale',
    'TimeStretch',
    'ChannelDropout',
    'Compose',
    'RandomApply'
]


class TimeShift(nn.Module):
    """Randomly shift signal in time (circular shift)."""

    def __init__(self, max_shift: float = 0.2):
        """
        Args:
            max_shift: Maximum shift as fraction of sequence length (0.0-1.0)
        """
        super().__init__()
        self.max_shift = max_shift

    def forward(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        shift = np.random.randint(-int(seq_len * self.max_shift), int(seq_len * self.max_shift) + 1)
        return np.roll(x, shift, axis=0)


class AddNoise(nn.Module):
    """Add Gaussian noise with random SNR."""

    def __init__(self, snr_min: float = 10.0, snr_max: float = 40.0):
        """
        Args:
            snr_min: Minimum signal-to-noise ratio in dB
            snr_max: Maximum signal-to-noise ratio in dB
        """
        super().__init__()
        self.snr_min = snr_min
        self.snr_max = snr_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        snr_db = np.random.uniform(self.snr_min, self.snr_max)
        signal_power = np.mean(x ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), x.shape)
        return (x + noise).astype(x.dtype)


class AmplitudeScale(nn.Module):
    """Random amplitude scaling."""

    def __init__(self, scale_min: float = 0.8, scale_max: float = 1.2):
        """
        Args:
            scale_min: Minimum scale factor
            scale_max: Maximum scale factor
        """
        super().__init__()
        self.scale_min = scale_min
        self.scale_max = scale_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        scale = np.random.uniform(self.scale_min, self.scale_max)
        return (x * scale).astype(x.dtype)


class TimeStretch(nn.Module):
    """Slight time stretching/compression using linear interpolation."""

    def __init__(self, rate_min: float = 0.9, rate_max: float = 1.1):
        """
        Args:
            rate_min: Minimum stretch rate (< 1.0 compresses, > 1.0 stretches)
            rate_max: Maximum stretch rate
        """
        super().__init__()
        self.rate_min = rate_min
        self.rate_max = rate_max

    def forward(self, x: np.ndarray) -> np.ndarray:
        rate = np.random.uniform(self.rate_min, self.rate_max)
        seq_len, n_channels = x.shape
        new_len = int(seq_len * rate)

        # Interpolate each channel
        old_indices = np.arange(seq_len)
        new_indices = np.linspace(0, seq_len - 1, new_len)

        stretched = np.zeros((new_len, n_channels), dtype=x.dtype)
        for ch in range(n_channels):
            stretched[:, ch] = np.interp(new_indices, old_indices, x[:, ch])

        # Resize back to original length
        if new_len != seq_len:
            final_indices = np.linspace(0, new_len - 1, seq_len)
            result = np.zeros_like(x)
            for ch in range(n_channels):
                result[:, ch] = np.interp(final_indices, np.arange(new_len), stretched[:, ch])
            return result

        return stretched


class ChannelDropout(nn.Module):
    """Randomly zero one or more channels."""

    def __init__(self, p_drop: float = 0.5, max_channels: int = 1):
        """
        Args:
            p_drop: Probability of dropping each selected channel
            max_channels: Maximum number of channels to drop
        """
        super().__init__()
        self.p_drop = p_drop
        self.max_channels = max_channels

    def forward(self, x: np.ndarray) -> np.ndarray:
        n_channels = x.shape[1]
        result = x.copy()

        n_drop = np.random.randint(1, min(self.max_channels, n_channels) + 1)
        channels_to_drop = np.random.choice(n_channels, n_drop, replace=False)

        for ch in channels_to_drop:
            if np.random.random() < self.p_drop:
                result[:, ch] = 0

        return result


class Compose(nn.Module):
    """Compose multiple augmentations."""

    def __init__(self, augmentations: list[nn.Module]):
        """
        Args:
            augmentations: List of augmentations to apply in order
        """
        super().__init__()
        self.augmentations = nn.ModuleList(augmentations)

    def forward(self, x: np.ndarray) -> np.ndarray:
        for aug in self.augmentations:
            x = aug(x)
        return x


class RandomApply(nn.Module):
    """Apply augmentation with given probability."""

    def __init__(self, augmentation: nn.Module, p: float = 0.5):
        """
        Args:
            augmentation: Augmentation to apply
            p: Probability of applying the augmentation
        """
        super().__init__()
        self.augmentation = augmentation
        self.p = p

    def forward(self, x: np.ndarray) -> np.ndarray:
        if np.random.random() < self.p:
            return self.augmentation(x)
        return x
