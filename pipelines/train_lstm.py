import numpy as np
import matplotlib.pyplot as plt
import torch

from quake.data.loader import read_pickle
from quake.data.adapter import WaveformDataAdapter, TransformOP
from quake.models.lstm import LSTMModel
from quake.models.lstm_mhsa import LSTMAttentionModel
from quake.procs.train import train_model


def train_lstm(
    events_train,
    events_test,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    model = LSTMModel(
        channels=channels,
        classes=2,
        hidden=64,
        layers=2,
        dropout=0.3
    ).to(device)

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model
    )

def train_lstm_attention(
    events_train,
    events_test,
    channels: int = 3,
    device: str = 'cuda'
) -> None:
    model = LSTMAttentionModel(
        channels=channels,
        classes=2,
        hidden=64,
        layers=2,
        heads=2,
        dropout=0.3
    ).to(device)

    train_model(
        events_train=events_train,
        events_test=events_test,
        model=model
    )

def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    channels = ('Z', 'N', 'E')
    # path_resource = './resources/hh_selected.pkl'
    path_resource = './resources/VRAC.pkl'

    events, labels = read_pickle(path_resource)
    assert len(events) == len(labels), 'Events and labels must have the same length.'

    unique, counts = np.unique(labels, return_counts=True)
    print(f'Number of events: {len(events)}')
    print(f'Event types: {unique}')

    plt.bar(unique, counts)
    plt.title('Event Distribution')
    plt.xlabel('Event Type')
    plt.ylabel('Count')
    plt.show()

    adapter = WaveformDataAdapter(
        events=events,
        labels=labels,
        channels=channels,
        fft_size=512,
        transforms=TransformOP.DROP_NAN | TransformOP.TRIMMING | TransformOP.ZSCORE | TransformOP.FFT,
        test_ratio=0.3
    )
    events_train, events_test = adapter.get_datasets()

    train_lstm(events_train, events_test, channels=len(channels), device=device)
    # train_lstm_attention(events_train, events_test, channels=len(channels), device=device)

if __name__ == "__main__":
    main()