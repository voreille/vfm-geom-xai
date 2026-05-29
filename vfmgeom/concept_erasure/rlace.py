from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Callable

import numpy as np
import torch
from torch import Tensor
from torch.optim import Optimizer
from sklearn.linear_model import SGDClassifier


@dataclass(frozen=True)
class RLaceEraser:
    """
    R-LACE eraser.

    Stores the rank-r subspace to erase as an orthonormal matrix `directions`
    of shape [d, r]. The full projection is:

        P = I - directions @ directions.T

    and the affine erasure is:

        x' = x - (x - bias) @ directions @ directions.T
    """

    directions: Tensor
    bias: Tensor | None = None
    metadata: dict | None = None

    @classmethod
    def fit(
        cls,
        x: Tensor,
        z: Tensor,
        *,
        x_dev: Tensor | None = None,
        z_dev: Tensor | None = None,
        rank: int = 1,
        affine: bool = True,
        device: str | torch.device | None = None,
        out_iters: int = 50_000,
        in_iters_adv: int = 1,
        in_iters_clf: int = 1,
        batch_size: int = 256,
        evaluate_every: int = 1_000,
        epsilon: float = 1e-3,
        optimizer_class: type[Optimizer] = torch.optim.SGD,
        optimizer_params_P: dict | None = None,
        optimizer_params_predictor: dict | None = None,
        seed: int | None = None,
    ) -> "RLaceEraser":
        return fit_rlace(
            x=x,
            z=z,
            x_dev=x_dev,
            z_dev=z_dev,
            rank=rank,
            affine=affine,
            device=device,
            out_iters=out_iters,
            in_iters_adv=in_iters_adv,
            in_iters_clf=in_iters_clf,
            batch_size=batch_size,
            evaluate_every=evaluate_every,
            epsilon=epsilon,
            optimizer_class=optimizer_class,
            optimizer_params_P=optimizer_params_P,
            optimizer_params_predictor=optimizer_params_predictor,
            seed=seed,
        )

    @property
    def P(self) -> Tensor:
        """Full projection matrix."""
        d = self.directions.shape[0]
        eye = torch.eye(d, device=self.directions.device, dtype=self.directions.dtype)
        return eye - self.directions @ self.directions.mH

    def __call__(self, x: Tensor) -> Tensor:
        """Apply R-LACE projection."""
        input_device = x.device
        input_dtype = x.dtype

        directions = self.directions.to(device=x.device, dtype=x.dtype)
        bias = None if self.bias is None else self.bias.to(device=x.device, dtype=x.dtype)

        delta = x - bias if bias is not None else x
        x_erased = x - (delta @ directions) @ directions.mH

        return x_erased.to(device=input_device, dtype=input_dtype)

    def to(self, device: torch.device | str) -> "RLaceEraser":
        return RLaceEraser(
            directions=self.directions.to(device),
            bias=self.bias.to(device) if self.bias is not None else None,
            metadata=self.metadata,
        )

    def state_dict(self) -> dict:
        return {
            "directions": self.directions,
            "bias": self.bias,
            "metadata": self.metadata,
        }

    @classmethod
    def from_state_dict(cls, state: dict) -> "RLaceEraser":
        return cls(
            directions=state["directions"],
            bias=state.get("bias", None),
            metadata=state.get("metadata", None),
        )

    def save(self, path: str | PathLike) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | PathLike,
        map_location: torch.device | str | None = None,
    ) -> "RLaceEraser":
        state = torch.load(path, map_location=map_location)
        return cls.from_state_dict(state)

def fit_rlace(
    x: Tensor,
    z: Tensor,
    *,
    x_dev: Tensor | None = None,
    z_dev: Tensor | None = None,
    rank: int = 1,
    affine: bool = True,
    device: str | torch.device | None = None,
    out_iters: int = 50_000,
    in_iters_adv: int = 1,
    in_iters_clf: int = 1,
    batch_size: int = 256,
    evaluate_every: int = 1_000,
    epsilon: float = 1e-3,
    optimizer_class: type[Optimizer] = torch.optim.SGD,
    optimizer_params_P: dict | None = None,
    optimizer_params_predictor: dict | None = None,
    seed: int | None = None,
) -> RLaceEraser:
    """
    Fit an R-LACE eraser.

    Parameters
    ----------
    x:
        Representation matrix of shape [n, d].
    z:
        Concept labels. Can be integer labels [n] or one-hot labels [n, k].
    x_dev, z_dev:
        Optional dev set used for early stopping / model selection.
        This should not be your final test set.
    rank:
        Number of dimensions to erase.
    """

    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    if optimizer_params_P is None:
        optimizer_params_P = {"lr": 0.003, "weight_decay": 1e-4}

    if optimizer_params_predictor is None:
        optimizer_params_predictor = {"lr": 0.003, "weight_decay": 1e-4}

    if device is None:
        device = x.device

    x = x.detach().to(device=device, dtype=torch.float32)
    y = _labels_from_z(z).detach().to(device=device)

    if x_dev is None:
        x_dev = x

    if z_dev is None:
        z_dev = y
    else:
        z_dev = _labels_from_z(z_dev)

    x_dev = x_dev.detach().to(device=device, dtype=torch.float32)
    y_dev = z_dev.detach().to(device=device)

    n, d = x.shape

    if rank <= 0:
        raise ValueError("rank must be positive.")

    if rank >= d:
        raise ValueError(f"rank must be < x_dim, got rank={rank}, x_dim={d}.")

    bias = x.mean(dim=0) if affine else None

    num_classes = int(y.max().item()) + 1

    if num_classes == 2:
        predictor = torch.nn.Linear(d, 1).to(device)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        y_for_loss = y.float()
    else:
        predictor = torch.nn.Linear(d, num_classes).to(device)
        loss_fn = torch.nn.CrossEntropyLoss()
        y_for_loss = y.long()

    # R is the soft rank-r concept-removal matrix.
    # The actual representation given to the adversary is:
    #
    #     x_erased = x - (x - bias) @ R
    #
    # During optimization, R is constrained to the Fantope:
    # symmetric, eigenvalues in [0, 1], trace = rank.
    R = 1e-1 * torch.randn(d, d, device=device)
    R = _project_to_fantope(R, rank)
    R.requires_grad_(True)

    optimizer_predictor = optimizer_class(
        predictor.parameters(),
        **optimizer_params_predictor,
    )
    optimizer_R = optimizer_class(
        [R],
        **optimizer_params_P,
    )

    majority_acc = _majority_accuracy(y_dev.detach().cpu().numpy())

    best_R: Tensor | None = None
    best_gap = float("inf")
    best_score = 1.0

    for step in range(out_iters):
        for _ in range(in_iters_adv):
            optimizer_R.zero_grad()

            idx = torch.randperm(n, device=device)[:batch_size]
            xb = x[idx]
            yb = y_for_loss[idx]

            xb_erased = _apply_soft_erasure(xb, R, bias)
            logits = predictor(xb_erased)

            loss = _classification_loss(logits, yb, loss_fn, num_classes)

            # The projection tries to make the concept classifier worse.
            loss_R = -loss
            loss_R.backward()
            optimizer_R.step()

            with torch.no_grad():
                R.copy_(_project_to_fantope(R, rank))

        for _ in range(in_iters_clf):
            optimizer_predictor.zero_grad()

            idx = torch.randperm(n, device=device)[:batch_size]
            xb = x[idx]
            yb = y_for_loss[idx]

            xb_erased = _apply_soft_erasure(xb, R.detach(), bias)
            logits = predictor(xb_erased)

            loss = _classification_loss(logits, yb, loss_fn, num_classes)
            loss.backward()
            optimizer_predictor.step()

        if step % evaluate_every == 0:
            with torch.no_grad():
                directions = _directions_from_soft_projection(R, rank)

            score = _linear_probe_score(
                x_train=x,
                y_train=y,
                x_dev=x_dev,
                y_dev=y_dev,
                directions=directions,
                bias=bias,
            )

            gap = abs(score - majority_acc)

            if gap < best_gap:
                best_gap = gap
                best_score = score
                best_R = R.detach().clone()

            if best_gap < epsilon:
                break

    if best_R is None:
        best_R = R.detach().clone()

    directions = _directions_from_soft_projection(best_R, rank)

    metadata = {
        "rank": rank,
        "majority_acc": majority_acc,
        "best_probe_acc": best_score,
        "best_gap": best_gap,
        "out_iters": out_iters,
    }

    return RLaceEraser(
        directions=directions.detach().cpu(),
        bias=bias.detach().cpu() if bias is not None else None,
        metadata=metadata,
    )

def _labels_from_z(z: Tensor) -> Tensor:
    """
    Accept either integer labels [n] or one-hot / soft labels [n, k].
    """
    if z.ndim == 1:
        return z.long()

    if z.ndim == 2:
        return z.argmax(dim=1).long()

    raise ValueError(f"Expected z with shape [n] or [n, k], got {tuple(z.shape)}.")


def _classification_loss(
    logits: Tensor,
    y: Tensor,
    loss_fn: Callable,
    num_classes: int,
) -> Tensor:
    if num_classes == 2:
        return loss_fn(logits.squeeze(-1), y.float())

    return loss_fn(logits, y.long())


def _apply_soft_erasure(
    x: Tensor,
    R: Tensor,
    bias: Tensor | None,
) -> Tensor:
    """
    Apply the current soft R-LACE projection.

    R is not necessarily idempotent during optimization.
    """
    delta = x - bias if bias is not None else x
    return x - delta @ R


def _symmetrize(A: Tensor) -> Tensor:
    return 0.5 * (A + A.mH)


@torch.no_grad()
def _project_to_fantope(A: Tensor, rank: int) -> Tensor:
    """
    Project a symmetric matrix onto the Fantope:

        {R : 0 <= R <= I, trace(R) = rank}

    This is the relaxed rank-r projection constraint used by R-LACE.
    """
    A = _symmetrize(A)

    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals_projected = _project_eigenvalues_to_capped_simplex(eigvals, rank)

    return eigvecs @ torch.diag(eigvals_projected) @ eigvecs.mH


def _project_eigenvalues_to_capped_simplex(
    eigvals: Tensor,
    rank: int,
    max_iter: int = 50,
) -> Tensor:
    """
    Solve:

        min ||lambda - eigvals||^2
        s.t. 0 <= lambda_i <= 1
             sum_i lambda_i = rank

    by bisection.
    """
    if rank < 0 or rank > eigvals.numel():
        raise ValueError(f"Invalid rank={rank} for {eigvals.numel()} eigenvalues.")

    lower = eigvals.min() - 1.0
    upper = eigvals.max()

    for _ in range(max_iter):
        theta = 0.5 * (lower + upper)
        projected = torch.clamp(eigvals - theta, min=0.0, max=1.0)

        if projected.sum() > rank:
            lower = theta
        else:
            upper = theta

    theta = 0.5 * (lower + upper)
    return torch.clamp(eigvals - theta, min=0.0, max=1.0)


@torch.no_grad()
def _directions_from_soft_projection(R: Tensor, rank: int) -> Tensor:
    """
    Convert the soft R-LACE matrix into a hard rank-r erased subspace.

    Returns directions of shape [d, rank].
    """
    R = _symmetrize(R)
    eigvals, eigvecs = torch.linalg.eigh(R)

    # Largest eigenvalues correspond to the erased directions.
    idx = torch.argsort(eigvals, descending=True)[:rank]
    directions = eigvecs[:, idx]

    # Numerical safety.
    directions, _ = torch.linalg.qr(directions)

    return directions[:, :rank]


def _majority_accuracy(y: np.ndarray) -> float:
    values, counts = np.unique(y, return_counts=True)
    return float(counts.max() / counts.sum())


def _linear_probe_score(
    *,
    x_train: Tensor,
    y_train: Tensor,
    x_dev: Tensor,
    y_dev: Tensor,
    directions: Tensor,
    bias: Tensor | None,
) -> float:
    """
    Evaluate how much concept information remains after erasure.

    This uses a fresh sklearn linear classifier, similarly to the original R-LACE repo.
    """
    x_train_erased = _apply_hard_erasure_numpy(x_train, directions, bias)
    x_dev_erased = _apply_hard_erasure_numpy(x_dev, directions, bias)

    y_train_np = y_train.detach().cpu().numpy()
    y_dev_np = y_dev.detach().cpu().numpy()

    clf = SGDClassifier(
        loss="log_loss",
        fit_intercept=True,
        max_iter=25_000,
        tol=1e-4,
        n_iter_no_change=15,
        alpha=1e-4,
        n_jobs=32,
    )

    clf.fit(x_train_erased, y_train_np)
    return float(clf.score(x_dev_erased, y_dev_np))


def _apply_hard_erasure_numpy(
    x: Tensor,
    directions: Tensor,
    bias: Tensor | None,
) -> np.ndarray:
    x_np = x.detach().cpu().float().numpy()
    directions_np = directions.detach().cpu().float().numpy()

    if bias is not None:
        bias_np = bias.detach().cpu().float().numpy()
        delta = x_np - bias_np
    else:
        delta = x_np

    x_erased = x_np - (delta @ directions_np) @ directions_np.T
    return x_erased