import numpy as np
import pandas as pd

from quake.data.resample import resample_events


def test_resample_length_and_frequency() -> None:
    fs_in, fs_out, f0 = 125.0, 100.0, 5.0
    t = np.arange(1000) / fs_in
    event = pd.DataFrame({c: np.sin(2 * np.pi * f0 * t) for c in ('Z', 'N', 'E')})

    out, = resample_events([event], fs_in, fs_out)

    assert out.shape == (800, 3)
    assert list(out.columns) == ['Z', 'N', 'E']
    assert out.dtypes.iloc[0] == np.float32

    # dominant frequency is preserved at the new rate
    spec = np.abs(np.fft.rfft(out['Z'].to_numpy()))
    freqs = np.fft.rfftfreq(len(out), d=1 / fs_out)
    assert abs(freqs[spec.argmax()] - f0) < 0.2


def test_resample_identity() -> None:
    event = pd.DataFrame(np.zeros((10, 3)), columns=['Z', 'N', 'E'])
    assert resample_events([event], 100, 100)[0] is event
