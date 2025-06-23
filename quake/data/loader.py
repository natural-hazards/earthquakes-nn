import os
import pickle as pkl

import pandas as pd
import numpy as np

from pathlib import Path

def read_pickle(
    fn: str | Path
) -> (tuple[pd.DataFrame, ...], np.ndarray):
    if not isinstance(fn, str | Path):
        msg: str = f'Not supported argument type #1 ({type(fn)})! It must be string or Path!'
        raise TypeError(msg)

    if not os.path.exists(fn):
        msg: str = f'File {fn} not exist!'
        raise IOError(msg)

    with open(fn, 'rb') as file:
        events, labels = pkl.load(file)

    return tuple(events), labels
