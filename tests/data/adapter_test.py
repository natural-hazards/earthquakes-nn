import numpy as np
import pandas as pd
import pytest

from quake.data.adapter import WaveformDataAdapter, TransformOP


def _events(lengths: list[int]) -> list[pd.DataFrame]:
    rng = np.random.default_rng(0)
    return [pd.DataFrame(rng.normal(size=(n, 3)), columns=['Z', 'N', 'E']) for n in lengths]


def test_padding() -> None:
    lengths = [50, 120, 80, 200] * 5
    labels = np.array(['E', 'M', 'T', 'M'] * 5)
    adapter = WaveformDataAdapter(
        events=_events(lengths),
        labels=labels,
        transforms=TransformOP.PADDING | TransformOP.ZSCORE,
        test_ratio=0.2,
        seq_len=100,
        random_state=0
    )
    ds_train, ds_test = adapter.get_datasets()

    for ds in (ds_train, ds_test):
        for i in range(len(ds)):
            x, _ = ds[i]
            assert x.shape == (100, 3)

            n = int((x.abs().sum(dim=1) > 0).sum())
            assert n in (50, 80, 100)
            # z-score computed on the real signal only, padding stays zero
            assert np.allclose(x[:n].mean(dim=0), 0, atol=1e-5)
            assert np.allclose(x[:n].std(dim=0, unbiased=False), 1, atol=1e-4)
            assert (x[n:] == 0).all()


def test_padding_requires_seq_len() -> None:
    with pytest.raises(ValueError):
        WaveformDataAdapter(_events([10]), np.array(['M']), transforms=TransformOP.PADDING)

    with pytest.raises(ValueError):
        WaveformDataAdapter(
            _events([10]), np.array(['M']),
            transforms=TransformOP.PADDING | TransformOP.TRIMMING, seq_len=10
        )
