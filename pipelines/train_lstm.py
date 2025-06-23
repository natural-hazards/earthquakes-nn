import numpy as np
import matplotlib.pyplot as plt

from quake.data.loader import read_pickle
from quake.data.adapter import WaveformDataAdapter, TransformOP

from quake.models.lstm import LSTMModel
from quake.procs.train import train_model

def main() -> None:
    channels: tuple = ('Z', 'N', 'E')

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
        channels=channels,
        fft_size=1500,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
        test_ratio=0.3
    )
    events_train, events_test = adapter.get_datasets()

    lstm_model = LSTMModel(
        channels=len(channels),
        classes=2,
        hidden=32,
        layers=3,
        dropout=0.7
    )
    lstm_model.to('cuda')

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=lstm_model
    )


if __name__ == '__main__':
    main()