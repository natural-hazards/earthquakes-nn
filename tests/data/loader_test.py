import numpy as np
import pandas as pd
import pytest

from quake.data.loader import read_pickle


@pytest.mark.parametrize('pickle_path', ('./resources/hh_selected.pkl',))
def test_loader(
    pickle_path
) -> None:
    events, labels = read_pickle(pickle_path)

    assert isinstance(events, tuple)
    assert all(isinstance(event, pd.DataFrame) for event in events)
    assert isinstance(labels, np.ndarray)
    assert len(events) == 4433
    assert labels.shape == (4433,)
