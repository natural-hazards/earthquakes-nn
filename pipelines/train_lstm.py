import numpy as np
import matplotlib.pyplot as plt

from quake.data.loader import read_pickle
from quake.data.adapter import WaveformDataAdapter, TransformOP

def main() -> None:
    path_resource: str = './resources/hh_selected.pkl'
    events, labels = read_pickle(path_resource)

    assert len(events) == len(labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(f'Number of events: {len(events)}')
    print(f'Type of events: {np.unique(labels)}')

    plt.bar(unique, counts)
    plt.show()

    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=('Z', 'N', 'E'),
        fft_size=256,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
        test_ratio=0.3
    )
    events_train, events_test = adapter.get_datasets()


if __name__ == '__main__':
    main()