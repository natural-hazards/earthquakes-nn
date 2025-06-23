import torch as tch
import torch.nn as nn

from torch.utils.data import DataLoader
from quake.data.dataset import WaveformDataset

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ProgressBar
from ignite.metrics import Fbeta


__all__ = [
    'train_model'
]


def _register_handlers(
    trainer,
    evaluator,
    loader_test
) -> None:
    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate_test(engine):
        evaluator.run(loader_test)
        test_f1 = evaluator.state.metrics['f1']
        print(f'Epoch {engine.state.epoch} - Test F1: {test_f1:.4f}')

    pbar = ProgressBar()
    pbar.attach(trainer, output_transform=lambda output: {"loss": output})


def train_lstm(
    events_train: WaveformDataset,
    events_test: WaveformDataset,
    model: nn.Module,
    batch_size: int = 32,
    batch_size_test: int = 32
) -> None:
    trainer = create_supervised_trainer(
        model=model,
        optimizer=tch.optim.AdamW(model.parameters(), lr=0.001),
        # todo adaptive learning rate
        loss_fn=nn.CrossEntropyLoss(),
        device='cuda' if tch.cuda.is_available() else 'cpu'
    )

    loader_train = DataLoader(
        dataset=events_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'f1': Fbeta(1.0, average=True),
        },
        device='cuda' if tch.cuda.is_available() else 'cpu'
    )

    loader_test = DataLoader(
       dataset=events_test,
       batch_size=batch_size_test,
       shuffle=False,
       num_workers=4
    )

    _register_handlers(trainer, evaluator, loader_test)
    trainer.run(loader_train, max_epochs=10)
