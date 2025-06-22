import numpy as np
import matplotlib.pyplot as plt

from quake.data.loader import read_pickle

def main() -> None:
    path_resource: str = './resources/hh_selected.pkl'
    events, labels = read_pickle(path_resource)

    assert len(events) == len(labels)

    unique, counts = np.unique(labels, return_counts=True)
    print(f'Number of events: {len(events)}')
    print(f'Type of events: {np.unique(labels)}')

    plt.bar(unique, counts)
    plt.show()

if __name__ == '__main__':
    main()