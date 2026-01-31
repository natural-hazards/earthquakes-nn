import torch as tch
import torch.nn as nn

from torch.utils.data import DataLoader
from quake.data.dataset import WaveformDataset

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Events
from ignite.handlers import ProgressBar, LRScheduler
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


def train_model(
    events_train: WaveformDataset,
    events_test: WaveformDataset,
    model: nn.Module,
    batch_size: int = 32,
    batch_size_test: int = 32,
    epochs: int = 20,
    lr: float = 0.001,
    device: str = 'cuda',
    use_amp: bool = True,
    compile_model: bool = True
) -> None:
    loader_train = DataLoader(
        dataset=events_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4
    )

    loader_test = DataLoader(
        dataset=events_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=4
    )

    if compile_model:
        model = tch.compile(model)

    optimizer = tch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = tch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=.85)
    lr_handler = LRScheduler(lr_scheduler)

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(),
        device=device,
        amp_mode="amp" if use_amp and device == 'cuda' else None
    )

    evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'f1': Fbeta(1.0, average=True),
        },
        device=device
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_handler)
    _register_handlers(trainer, evaluator, loader_test)

    trainer.run(loader_train, max_epochs=epochs)
