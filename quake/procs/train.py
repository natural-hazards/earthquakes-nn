import numpy as np
import torch as tch
import torch.nn as nn

from torch.utils.data import DataLoader
from quake.data.dataset import WaveformDataset

from ignite.engine import create_supervised_trainer, create_supervised_evaluator, Engine, Events
from ignite.handlers import ProgressBar, LRScheduler
from ignite.metrics import Fbeta


def _compute_class_weights(
    dataset: WaveformDataset,
    device: str,
    weighting: str | None = 'inverse'
) -> tch.Tensor:
    """Compute class weights for imbalanced datasets.

    Args:
        dataset: Training dataset
        device: Target device
        weighting: 'inverse' (N / (K * n_k)), 'sqrt' (square root of the
            inverse weights) or None (all ones)
    """
    labels = dataset.labels
    classes, counts = np.unique(labels, return_counts=True)
    weights = len(labels) / (len(classes) * counts)
    if weighting is None:
        weights = np.ones_like(weights)
    elif weighting == 'sqrt':
        weights = np.sqrt(weights)
    elif weighting != 'inverse':
        raise ValueError(f'Unknown weighting {weighting!r}!')
    return tch.tensor(weights, dtype=tch.float32).to(device)



def _clip_before_step(optimizer: tch.optim.Optimizer, max_norm: float) -> None:
    """Clip the global gradient norm right before each optimizer step."""
    params = [p for group in optimizer.param_groups for p in group['params']]

    def clip(optimizer, args, kwargs) -> None:
        tch.nn.utils.clip_grad_norm_(params, max_norm)

    # A step pre-hook (unlike replacing optimizer.step) is compatible with LR schedulers
    # and runs after GradScaler has unscaled the gradients.
    optimizer.register_step_pre_hook(clip)


__all__ = [
    'train_model',
    'train_svm_model'
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
    compile_model: bool = True,
    weighting: str | None = 'inverse',
    lr_gamma: float = .85,
    grad_clip: float | None = None
) -> None:
    loader_train = DataLoader(
        dataset=events_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True  # a trailing batch of size 1 breaks BatchNorm in training mode
    )

    loader_test = DataLoader(
        dataset=events_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=4
    )

    if compile_model:
        model = tch.compile(model)

    class_weights = _compute_class_weights(events_train, device, weighting)

    optimizer = tch.optim.AdamW(model.parameters(), lr=lr)
    if grad_clip is not None:
        _clip_before_step(optimizer, grad_clip)
    lr_scheduler = tch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    lr_handler = LRScheduler(lr_scheduler)

    trainer = create_supervised_trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=nn.CrossEntropyLoss(weight=class_weights),
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


def _svm_train_step(model, optimizer, device, use_amp, class_weights):
    scaler = tch.amp.GradScaler('cuda', enabled=use_amp and device == 'cuda')

    def step(engine, batch):
        model.train()
        x, y = batch[0].to(device), batch[1].to(device)

        with tch.amp.autocast('cuda', enabled=use_amp and device == 'cuda'):
            features = model.extract_features(x)
            output = model.classifier(features)
            sample_weights = class_weights[y]
            if output.dim() == 1:
                # Binary hinge loss on a single decision value
                y_svm = 2 * y.float() - 1
                loss = (sample_weights * tch.clamp(1 - y_svm * output, min=0)).mean()
            else:
                # One-vs-rest hinge loss: +1 for the true class, -1 elsewhere
                y_ovr = tch.full_like(output, -1.0)
                y_ovr[tch.arange(len(y), device=y.device), y] = 1.0
                loss = (sample_weights.unsqueeze(1) * tch.clamp(1 - y_ovr * output, min=0)).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        model.classifier.fit_step(features, y, sample_weight=sample_weights)

        return loss.item()

    return Engine(step)


def _svm_eval_output_transform(output):
    y_pred, y = output
    if y_pred.dim() == 1:
        return (y_pred > 0).long(), y
    # Multiclass decision values [B, classes]: ignite takes argmax itself
    return y_pred, y


def train_svm_model(
    events_train: WaveformDataset,
    events_test: WaveformDataset,
    model: nn.Module,
    batch_size: int = 32,
    batch_size_test: int = 32,
    epochs: int = 20,
    lr: float = 0.001,
    device: str = 'cuda',
    use_amp: bool = True,
    compile_model: bool = False,
    weighting: str | None = 'inverse',
    lr_gamma: float = .85,
    grad_clip: float | None = None
) -> None:
    loader_train = DataLoader(
        dataset=events_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True  # a trailing batch of size 1 breaks BatchNorm in training mode
    )

    loader_test = DataLoader(
        dataset=events_test,
        batch_size=batch_size_test,
        shuffle=False,
        num_workers=4
    )

    if compile_model:
        model = tch.compile(model)

    class_weights = _compute_class_weights(events_train, device, weighting)

    optimizer = tch.optim.AdamW(model.parameters(), lr=lr)
    if grad_clip is not None:
        _clip_before_step(optimizer, grad_clip)
    lr_scheduler = tch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=lr_gamma)
    lr_handler = LRScheduler(lr_scheduler)

    trainer = _svm_train_step(model, optimizer, device, use_amp, class_weights)

    evaluator = create_supervised_evaluator(
        model=model,
        metrics={
            'f1': Fbeta(1.0, average=True, output_transform=_svm_eval_output_transform),
        },
        device=device
    )

    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_handler)
    _register_handlers(trainer, evaluator, loader_test)

    trainer.run(loader_train, max_epochs=epochs)
