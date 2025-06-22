import numpy as np
import pandas as pd

from enum import IntFlag
from scipy import stats
from sklearn import model_selection

from sklearn.preprocessing import LabelEncoder, StandardScaler
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
    FFT = 1 << 6
    SPECTROGRAM = 1 << 7


class WaveformDataAdapter(object):

    def __init__(
        self,
        events: list[pd.DataFrame] | tuple[pd.DataFrame, ...],
        labels: np.ndarray,
        channels: list[str] | tuple[str, ...] | None = ('Z', 'N', 'E'),
        fft_size: int = 32,
        transforms: TransformOP = TransformOP.NONE,
        test_ratio: float = 0.3,
    ) -> None:
        self.__channels: list[str] | tuple[str, ...] | None = None
        self.channels = channels

        self.__fft_size = fft_size
        self.fft_size = fft_size

        self.__transforms = transforms
        self.transforms = transforms

        self.__test_ratio = test_ratio
        self.test_ratio = test_ratio

        self.__events: pd.DataFrame | None = None
        self.events = events

        self.__labels: np.ndarray | None = None
        self.labels = labels

        self.__ds_train: tuple | None = None
        self.__ds_test: tuple | None = None

    def __del__(self) -> None:
        self.__reset()

    def __reset(self) -> None:
        if hasattr(self, '__ds_train'):
            del self.__ds_train
            self.__ds_train = None

        if hasattr(self, '__ds_test'):
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
        if hasattr(self, '__events'):
            del self.events
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
        if hasattr(self, '__labels'):
            del self.labels
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
    def fft_size(
        self
    ) -> int:
        return self.__fft_size

    @fft_size.setter
    def fft_size(
        self,
        size: int
    ) -> None:
        if size == self.__fft_size:
            return

        self.__reset()
        self.__fft_size = size

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

    def __process_event(
        self,
        event: pd.DataFrame,
        min_length: int = 0,
    ) -> np.ndarray:
        if self.__transforms & TransformOP.DROP_NAN == TransformOP.DROP_NAN:
            event = event.dropna()
        if self.__transforms & TransformOP.TRIMMING == TransformOP.TRIMMING:
            event = event[self.channels[0] if len(self.channels) == 1 else self.channels][:min_length]

        if isinstance(event, pd.DataFrame):
            event = event.to_numpy(dtype=np.float32)

        if self.__transforms & TransformOP.ZSCORE == TransformOP.ZSCORE:
            for ch in range(event.shape[1]):
                event[:, ch] = stats.zscore(event[:, ch])
        if self.__transforms & TransformOP.FFT == TransformOP.FFT:
            fft_result = np.empty((self.fft_size, event.shape[1]), dtype=np.float64)
            for ch in range(event.shape[1]):
                fft_result[:, ch] = np.abs(np.fft.fft(event[:, ch], n=self.fft_size))

        return event

    def __create_datasets(
        self
    ) -> None:
        len_events: list = [len(event) for event in self.events]
        min_length = min(len_events)

        encoder_label = LabelEncoder()
        labels = encoder_label.fit_transform(self.labels)

        events = list()
        for event in self.events:
            event = self.__process_event(
                event=event,
                min_length=min_length
            )
            events.append(event)

        events_train, events_test, labels_train, labels_test = model_selection.train_test_split(
            events,
            labels,
            test_size=self.test_ratio
        )

        self.__ds_train = (events_train, labels_train)
        self.__ds_test = (events_test, labels_test)

    def get_datasets(
        self
    ) -> tuple[tuple, tuple]:
        if self.__ds_train is not None and self.__ds_test is not None:
            return self.__ds_train, self.__ds_test

        self.__create_datasets()
        return self.__ds_train, self.__ds_test
