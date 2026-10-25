import torch
import numpy as np

from torch import nn
from sklearn.linear_model import SGDClassifier


__all__ = [
    'SVMClassifier'
]


class SVMFunction(torch.autograd.Function):
    """Custom autograd that passes gradients through to upstream layers.

    SVM weights (w, b) are updated by sklearn's SGDClassifier, not by
    PyTorch's optimizer. Only the gradient w.r.t. input x is computed
    so the LSTM receives gradients via backpropagation.
    """

    @staticmethod
    def forward(ctx, x, w, b):
        ctx.save_for_backward(w,)
        return x @ w.T + b

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        grad_x = grad_output @ w
        return grad_x, None, None


class SVMClassifier(nn.Module):
    """Linear SVM classifier trained via sklearn's SGDClassifier.

    Weights are updated by sklearn's partial_fit() and synced to PyTorch
    buffers. A custom autograd function passes gradients through to
    upstream layers (LSTM) while keeping SVM weights outside PyTorch's
    optimizer.

    For binary classification (classes=2) the output is a single decision
    value per sample [B]. For multiclass (classes>2) a one-vs-rest scheme
    is used, matching sklearn, and the output is [B, classes].

    Args:
        in_features: Dimension of input feature vector
        classes: Number of output classes
        alpha: L2 regularization strength (equivalent to 1/C)
        eta0: Learning rate for SGDClassifier
    """

    def __init__(
        self,
        in_features: int,
        classes: int = 2,
        alpha: float = 1e-4,
        eta0: float = 0.01
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.classes = classes

        # sklearn stores one weight vector for binary, one per class for OvR
        n_out = 1 if classes == 2 else classes
        self.register_buffer('w', torch.zeros(n_out, in_features))
        self.register_buffer('b', torch.zeros(n_out))

        self.sgd_clf = SGDClassifier(
            loss='hinge',
            alpha=alpha,
            learning_rate='constant',
            eta0=eta0,
            fit_intercept=True,
        )
        self._fitted = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = SVMFunction.apply(x, self.w, self.b)
        return out.squeeze(-1) if self.classes == 2 else out

    def fit_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sample_weight: torch.Tensor | None = None
    ) -> None:
        """Update SVM weights via sklearn and sync to PyTorch buffers.

        Call this after backward() each training step. `sample_weight`
        should match the weights used in the feature extractor's loss,
        otherwise the SVM and the extractor optimize different objectives.
        """
        x_np = x.detach().float().cpu().numpy()
        y_np = y.detach().cpu().numpy()
        sw_np = None if sample_weight is None else sample_weight.detach().float().cpu().numpy()

        self.sgd_clf.partial_fit(x_np, y_np, classes=np.arange(self.classes), sample_weight=sw_np)
        self._fitted = True

        self.w.copy_(torch.from_numpy(self.sgd_clf.coef_).to(self.w.device))
        self.b.copy_(torch.from_numpy(self.sgd_clf.intercept_).to(self.b.device))
