import pandas as pd

from enum import Enum

from quake.data.dataset import WaveformDataset


__all__ = [
    'TransformOP',
    'WaveformDataAdapter'
]


class TransformOP(Enum):
    NONE = 1 << 0
    DROP_NAN = 1 << 1
    TRIMMING = 1 << 3
    PADDING = 1 << 4
    ZSCORE = 1 << 5
    PCA = 1 << 6
    FFT = 1 << 7

    def __and__(self, other):
        if isinstance(other, TransformOP):
            return TransformOP(self.value & other.value)
        return NotImplemented


class WaveformDataAdapter(object):

    def __init__(
        self,
        waveform_info: pd.DataFrame,
        transforms: TransformOP,
        pca_components: int,
        fft_size: int,
        test_ratio: float = 0.3,
    ) -> None:
        self.__transforms = transforms
        self.transforms = transforms

        self.__waveform_info: pd.DataFrame | None = None
        self.waveform_info = waveform_info

        self.__pca_components = pca_components
        self.pca_components = pca_components

        self.__fft_size = fft_size
        self.fft_size = fft_size

        self.__test_ratio = test_ratio
        self.test_ratio = test_ratio

    def __del__(self) -> None:
        self.__reset()

    def __reset(self) -> None:
        if hasattr(self, 'waveform_info'):
            del self.waveform_info
            self.waveform_info = None

    @property
    def waveform_info(
        self
    ) -> pd.DataFrame | None:
        return self.__waveform_info

    @waveform_info.setter
    def waveform_info(
        self,
        waveform_info: pd.DataFrame | None
    ) -> None:
        if self.waveform_info is None:
            self.__reset()
            self.__waveform_info = None
            return

        if waveform_info.equals(self.waveform_info):
            return

        self.__reset()
        self.__waveform_info = waveform_info

    @waveform_info.deleter
    def waveform_info(
        self
    ) -> None:
        if hasattr(self, '__waveform_info'):
            del self.waveform_info
            self.waveform_info = None

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
    def pca_components(
        self
    ) -> int:
        return self.__pca_components

    @pca_components.setter
    def pca_components(
        self,
        components: int
    ) -> None:
        if components == self.__pca_components:
            return

        self.__reset()
        self.__pca_components = components

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

    def getDatasets(
        self
    ) -> tuple[WaveformDataset, WaveformDataset]:
        pass
