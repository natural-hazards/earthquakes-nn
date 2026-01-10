import torch as tch
import numpy as np


__all__ = [
    'WaveformDataset'
]


class WaveformDataset:

    def __init__(
        self,
        events: np.ndarray | list[np.ndarray],
        labels: np.ndarray,
    ) -> None:
        self.__events = events
        self.__labels = labels

    def __getitem__(self, idx):
        event = self.__events[idx]
        label = self.__labels[idx]

        return event, label

    def __len__(self) -> int:
        return len(self.__events)
