import numpy as np
import pandas as pd

from enum import IntFlag
from scipy import stats
from sklearn import model_selection
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from quake.data.dataset import WaveformDataset


__all__ = [
    'TransformOP',
    'WaveformDataAdapter'
]


class TransformOP(IntFlag):
    NONE = 1 << 0
    DROP_NAN = 1 << 1
    TRIMMING = 1 << 3
    ZSCORE = 1 << 4
    PCA = 1 << 5


class WaveformDataAdapter(object):

    def __init__(
        self,
        events: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
        labels: np.ndarray,
        channels: list[str] | tuple[str, ...] | None = ('Z', 'N', 'E'),
        transforms: TransformOP = TransformOP.NONE,
        test_ratio: float = 0.3,
    ) -> None:
        self.__channels: list[str] | tuple[str, ...] | None = None
        self.channels = channels

        self.__transforms = transforms
        self.transforms = transforms

        self.__test_ratio = test_ratio
        self.test_ratio = test_ratio

        self.__events: pd.DataFrame | None = None
        self.events = events

        self.__labels: np.ndarray | None = None
        self.labels = labels

        self.__ds_train: WaveformDataset | None = None
        self.__ds_test: WaveformDataset | None = None

    def __del__(self) -> None:
        self.__reset()

    def __reset(self) -> None:
        if hasattr(self, '_WaveformDataAdapter__ds_train'):
            del self.__ds_train
            self.__ds_train = None

        if hasattr(self, '_WaveformDataAdapter__ds_test'):
            del self.__ds_test
            self.__ds_test = None

    @property
    def events(
        self
    ) -> pd.DataFrame | None:
        return self.__events

    @events.setter
    def events(
        self,
        lst_events: list[pd.DataFrame] | tuple[pd.DataFrame] | None
    ) -> None:
        if lst_events is None:
            del self.events
            return

        del self.events
        self.__events = lst_events

    @events.deleter
    def events(
        self
    ) -> None:
        if hasattr(self, '_WaveformDataAdapter__events'):
            del self.__events
            self.__events = None

    @property
    def labels(
        self
    ) -> np.ndarray | None:
        return self.__labels

    @labels.setter
    def labels(
        self,
        np_labels: np.ndarray
    ) -> None:
        if np.array_equal(self.__labels, np_labels):
            return

        del self.labels
        self.__labels = np_labels

    @labels.deleter
    def labels(
        self
    ) -> None:
        if hasattr(self, '_WaveformDataAdapter__labels'):
            del self.__labels
            self.__labels = None

    @property
    def transforms(
        self
    ) -> TransformOP:
        return self.__transforms

    @transforms.setter
    def transforms(
        self,
        op: TransformOP
    ) -> None:
        if op == self.__transforms:
            return

        self.__reset()
        self.__transforms = op

    @property
    def channels(
        self
    ) -> list[str] | tuple[str, ...] | None:
        return self.__channels

    @channels.setter
    def channels(
        self,
        channels: list[str] | tuple[str, ...] | None
    ) -> None:
        if channels is None or channels == self.__channels:
            return

        self.__reset()
        self.__channels = list(channels)

    @property
    def test_ratio(
        self
    ) -> float:
        return self.__test_ratio

    @test_ratio.setter
    def test_ratio(
        self,
        ratio: float
    ) -> None:
        if ratio == self.__test_ratio:
            return

        self.__reset()
        self.__test_ratio = ratio

    def __process_events(
        self,
        events: list[pd.DataFrame]
    ) -> np.ndarray:
        channels = self.channels[0] if len(self.channels) == 1 else self.channels

        # First pass: DROP_NAN and convert to numpy
        arrays = []
        for event in tqdm(events, desc="Converting events"):
            if self.__transforms & TransformOP.DROP_NAN:
                event = event.dropna()
            arrays.append(event[channels].to_numpy(dtype=np.float32))

        # Trim to min length and stack
        if self.__transforms & TransformOP.TRIMMING:
            min_len = min(len(arr) for arr in arrays)
            arrays = [arr[:min_len] for arr in arrays]

        stacked = np.stack(arrays)  # [n_samples, seq_len, n_channels]

        # Batch ZSCORE
        if self.__transforms & TransformOP.ZSCORE:
            for ch in range(stacked.shape[2]):
                stacked[:, :, ch] = stats.zscore(stacked[:, :, ch], axis=1)

        return stacked

    def __create_datasets(
        self,
        transform_train=None,
        transform_test=None
    ) -> None:
        encoder_label = LabelEncoder()
        labels = encoder_label.fit_transform(self.labels)

        print(f"Processing {len(self.events)} events...")
        events = self.__process_events(list(self.events))

        events_train, events_test, labels_train, labels_test = model_selection.train_test_split(
            events,
            labels,
            test_size=self.test_ratio
        )

        self.__ds_train = WaveformDataset(
            events=events_train,
            labels=labels_train,
            transform=transform_train
        )
        self.__ds_test = WaveformDataset(
            events=events_test,
            labels=labels_test,
            transform=transform_test
        )

    def get_datasets(
        self,
        transform_train=None,
        transform_test=None
    ) -> tuple[WaveformDataset, WaveformDataset]:
        """Get train/test datasets with optional transforms.

        Args:
            transform_train: Transform pipeline for training data (with augmentation)
            transform_test: Transform pipeline for test data (without augmentation)

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        if self.__ds_train is not None and self.__ds_test is not None:
            return self.__ds_train, self.__ds_test

        self.__create_datasets(transform_train, transform_test)
        return self.__ds_train, self.__ds_test
