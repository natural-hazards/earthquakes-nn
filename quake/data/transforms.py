"""
Dynamic transforms for on-the-fly FFT/STFT computation.
Inherits from torch.nn.Module for PyTorch compatibility.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import stft


__all__ = [
    'ToTensor',
    'FFTTransform',
    'STFTTransform'
]


class ToTensor(nn.Module):
    """Convert numpy array to torch tensor."""

    def forward(self, x: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(x.copy())


class FFTTransform(nn.Module):
    """Compute FFT magnitude spectrum.

    Input: [seq_len, channels]
    Output: [fft_output_size, channels]
    """

    def __init__(self, fft_size: int = 512):
        super().__init__()
        self.fft_size = fft_size
        self.output_size = fft_size // 2 + 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [seq_len, channels]
        start_idx = self.fft_size // 2 - 1
        fft_full = np.abs(np.fft.fft(x, n=self.fft_size, axis=0))
        return fft_full[start_idx:start_idx + self.output_size, :].astype(np.float32)


class STFTTransform(nn.Module):
    """Compute STFT spectrogram.

    Input: [seq_len, channels]
    Output: [channels, freq_bins, time_frames]
    """

    def __init__(
        self,
        nperseg: int = 64,
        hop: int = 32,
        window: str = 'hann'
    ):
        super().__init__()
        self.nperseg = nperseg
        self.hop = hop
        self.noverlap = nperseg - hop
        self.window = window
        self.freq_bins = nperseg // 2 + 1

    def forward(self, x: np.ndarray) -> np.ndarray:
        # x: [seq_len, channels]
        seq_len, n_channels = x.shape

        # Compute for first channel to get dimensions
        _, _, Zxx = stft(
            x[:, 0],
            nperseg=self.nperseg,
            noverlap=self.noverlap,
            window=self.window
        )
        freq_bins, time_frames = Zxx.shape

        # Allocate output: [channels, freq_bins, time_frames]
        result = np.empty((n_channels, freq_bins, time_frames), dtype=np.float32)
        result[0] = np.abs(Zxx)

        for ch in range(1, n_channels):
            _, _, Zxx = stft(
                x[:, ch],
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                window=self.window
            )
            result[ch] = np.abs(Zxx)

        return result
