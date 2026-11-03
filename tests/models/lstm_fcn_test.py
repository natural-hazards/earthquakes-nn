import torch

from quake.models.lstm_fcn import LSTMFCNModel


def test_lstm_fcn_shapes() -> None:
    model = LSTMFCNModel(channels=3, classes=3, hidden=16, conv_channels=(16, 16, 16))
    model.eval()
    out = model(torch.randn(4, 513, 3))
    assert out.shape == (4, 3)


def test_lstm_fcn_default_is_karim() -> None:
    model = LSTMFCNModel(channels=3, classes=2)
    assert model.classifier.in_features == 128 + 128
    assert model(torch.randn(2, 100, 3)).shape == (2, 2)
