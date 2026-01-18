import torch as tch
import numpy as np
from torch import Tensor
from typing import Callable


__all__ = [
    'WaveformDataset'
]


class WaveformDataset:
    """Dataset with optional on-the-fly transforms.

    Pipeline (when transform is provided):
        1. Get time series sample
        2. Apply transform (augmentation → FFT/STFT → spec augmentation)

    Args:
        events: Array of shape [N, seq_len, channels]
        labels: Array of shape [N]
        transform: Optional transform pipeline (e.g., nn.Sequential or Compose)
    """

    def __init__(
        self,
        events: np.ndarray | list[np.ndarray],
        labels: np.ndarray,
        transform: Callable[[np.ndarray], Tensor] | None = None
    ) -> None:
        self.__events = events
        self.__labels = labels
        self.__transform = transform

    def __getitem__(self, idx) -> tuple[Tensor, int]:
        event = self.__events[idx]
        label = self.__labels[idx]

        if self.__transform is not None:
            event = self.__transform(event)
        else:
            event = tch.from_numpy(event)

        return event, label

    def __len__(self) -> int:
        return len(self.__events)
