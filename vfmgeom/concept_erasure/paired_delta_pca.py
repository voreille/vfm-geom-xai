from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any

import torch
from torch import Tensor

from .shrinkage import optimal_linear_shrinkage


@dataclass(frozen=True)
class PairedDeltaPcaEraser:
    """Erase a nuisance subspace estimated by PCA on paired deltas.

    The affine transformation is

        x' = bias + P (x - bias),
        P  = I - proj_left @ proj_right.

    With ``whitening=False``, ``P`` is the orthogonal projection onto the
    complement of the leading eigenspace of ``Cov(delta)``.

    With ``whitening=True``, PCA is performed on

        Cov(X)^(-1/2) Cov(delta) Cov(X)^(-1/2),

    and the resulting projection is mapped back to the original feature space.
    The mapped projection is generally oblique rather than orthogonal.
    """

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None
    eigenvalues: Tensor
    requested_rank: int
    whitening: bool = False

    @property
    def rank(self) -> int:
        """Effective rank after numerical truncation."""
        return self.proj_left.shape[1]

    @property
    def P(self) -> Tensor:
        """Materialize the full ``d x d`` projection matrix."""
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the eraser while preserving the input device and dtype."""
        input_device = x.device
        input_dtype = x.dtype

        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        centered = work - self.bias if self.bias is not None else work

        # Avoid materializing the full d x d matrix.
        correction = (centered @ self.proj_right.mH) @ self.proj_left.mH
        output = centered - correction

        if self.bias is not None:
            output = output + self.bias

        return output.to(device=input_device, dtype=input_dtype)

    def transform(self, x: Tensor) -> Tensor:
        """Alias for ``__call__``."""
        return self(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "proj_left": self.proj_left,
            "proj_right": self.proj_right,
            "bias": self.bias,
            "eigenvalues": self.eigenvalues,
            "requested_rank": self.requested_rank,
            "whitening": self.whitening,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PairedDeltaPcaEraser":
        return cls(
            proj_left=state["proj_left"],
            proj_right=state["proj_right"],
            bias=state["bias"],
            eigenvalues=state.get(
                "eigenvalues",
                torch.empty(
                    state["proj_left"].shape[1],
                    device=state["proj_left"].device,
                    dtype=state["proj_left"].dtype,
                ),
            ),
            requested_rank=state.get("requested_rank", state["proj_left"].shape[1]),
            whitening=state.get("whitening", False),
        )

    def save(self, path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        map_location: torch.device | str | None = None,
    ) -> "PairedDeltaPcaEraser":
        state = torch.load(path, map_location=map_location, weights_only=True)
        return cls.from_state_dict(state)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "PairedDeltaPcaEraser":
        return PairedDeltaPcaEraser(
            proj_left=self.proj_left.to(device=device, dtype=dtype),
            proj_right=self.proj_right.to(device=device, dtype=dtype),
            bias=(
                self.bias.to(device=device, dtype=dtype)
                if self.bias is not None
                else None
            ),
            eigenvalues=self.eigenvalues.to(device=device, dtype=dtype),
            requested_rank=self.requested_rank,
            whitening=self.whitening,
        )


class PairedDeltaPcaFitter:
    """Streaming fitter for PCA erasure from paired nuisance deltas.

    Examples
    --------
    Incremental fitting::

        fitter = PairedDeltaPcaFitter(x_dim=d, rank=16)
        fitter.update(x=batch_x, delta=batch_delta)
        eraser = fitter.eraser

    Fitting directly from paired views::

        fitter.update_from_pairs(x=batch_x, x1=batch_x1, x2=batch_x2)

    ``x`` is used to estimate the affine bias and, when requested, the feature
    covariance used for whitening. ``delta`` may contain a different number of
    samples from ``x`` because their streaming statistics are independent.
    """

    @classmethod
    def fit(
        cls,
        x: Tensor,
        delta: Tensor,
        **kwargs: Any,
    ) -> "PairedDeltaPcaFitter":
        x_flat = x.reshape(-1, x.shape[-1])
        fitter = cls(
            x_dim=x_flat.shape[-1],
            device=x.device,
            dtype=x.dtype,
            **kwargs,
        )
        return fitter.update(x=x, delta=delta)

    @classmethod
    def fit_from_pairs(
        cls,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
        **kwargs: Any,
    ) -> "PairedDeltaPcaFitter":
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )
        return cls.fit(x=x, delta=x1 - x2, **kwargs)

    def __init__(
        self,
        x_dim: int,
        rank: int,
        *,
        whitening: bool = False,
        affine: bool = True,
        shrinkage: bool = True,
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        if not 1 <= rank <= x_dim:
            raise ValueError(
                f"rank must satisfy 1 <= rank <= x_dim; got rank={rank}, x_dim={x_dim}."
            )
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if svd_tol < 0:
            raise ValueError("svd_tol must be non-negative.")
        if shrinkage and whitening and optimal_linear_shrinkage is None:
            raise ImportError(
                "shrinkage=True requires optimal_linear_shrinkage. "
                "Install/import the shrinkage module or set shrinkage=False."
            )

        self.x_dim = x_dim
        self.rank = rank
        self.whitening = whitening
        self.affine = affine
        self.shrinkage = shrinkage
        self.ridge = ridge
        self.svd_tol = svd_tol

        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_delta = torch.zeros(x_dim, device=device, dtype=dtype)

        # X covariance is only needed for whitening. Keeping it conditional saves
        # d^2 storage in the default raw-PCA case.
        self.sigma_xx_: Tensor | None = (
            torch.zeros(x_dim, x_dim, device=device, dtype=dtype) if whitening else None
        )
        self.sigma_dd_ = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)

        self.n_x = torch.tensor(0, device=device, dtype=torch.long)
        self.n_delta = torch.tensor(0, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, x: Tensor, delta: Tensor) -> "PairedDeltaPcaFitter":
        """Update feature and paired-delta statistics from one batch."""
        x = self._prepare(x, name="x", reference=self.mean_x)
        delta = self._prepare(delta, name="delta", reference=self.mean_delta)

        self._update_x(x)
        self._update_delta(delta)
        return self

    @torch.no_grad()
    def update_from_pairs(
        self,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
    ) -> "PairedDeltaPcaFitter":
        """Update using paired views, with ``delta = x1 - x2``."""
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )

        x_prepared = self._prepare(x, name="x", reference=self.mean_x)
        delta_prepared = self._prepare(
            x1 - x2,
            name="x1 - x2",
            reference=self.mean_delta,
        )
        self._update_x(x_prepared)
        self._update_delta(delta_prepared)
        return self

    def _prepare(self, value: Tensor, *, name: str, reference: Tensor) -> Tensor:
        if value.shape[-1] != self.x_dim:
            raise ValueError(
                f"{name} must have last dimension {self.x_dim}, "
                f"got shape {tuple(value.shape)}."
            )
        return value.reshape(-1, self.x_dim).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    @torch.no_grad()
    def _update_mean_and_m2(
        batch: Tensor,
        mean: Tensor,
        m2: Tensor | None,
        count: Tensor,
    ) -> None:
        """Merge a batch into running mean/M2 statistics using batch Welford."""
        batch_n = batch.shape[0]
        if batch_n == 0:
            return

        old_n = count.clone()
        new_n = old_n + batch_n
        batch_mean = batch.mean(dim=0)
        mean_shift = batch_mean - mean

        if m2 is not None:
            centered = batch - batch_mean
            batch_m2 = centered.mH @ centered
            cross = torch.outer(mean_shift, mean_shift) * (
                old_n.to(batch.dtype) * batch_n / new_n.to(batch.dtype)
            )
            m2.add_(batch_m2 + cross)

        mean.add_(mean_shift * (batch_n / new_n.to(batch.dtype)))
        count.copy_(new_n)

    def _update_x(self, x: Tensor) -> None:
        self._update_mean_and_m2(
            batch=x,
            mean=self.mean_x,
            m2=self.sigma_xx_,
            count=self.n_x,
        )

    def _update_delta(self, delta: Tensor) -> None:
        self._update_mean_and_m2(
            batch=delta,
            mean=self.mean_delta,
            m2=self.sigma_dd_,
            count=self.n_delta,
        )

    @property
    def sigma_xx(self) -> Tensor:
        """Estimated covariance of X, optionally with linear shrinkage."""
        if self.sigma_xx_ is None:
            raise RuntimeError("Cov(X) is not tracked when whitening=False.")
        if self.n_x.item() <= 1:
            raise RuntimeError("At least two X samples are required.")

        m2 = self._symmetrize(self.sigma_xx_)
        if self.shrinkage:
            assert optimal_linear_shrinkage is not None
            covariance_mle = m2 / self.n_x.to(m2.dtype)
            covariance = optimal_linear_shrinkage(
                covariance_mle,
                self.n_x,
                inplace=False,
            )
        else:
            covariance = m2 / (self.n_x - 1).to(m2.dtype)

        return self._symmetrize(covariance)

    @property
    def sigma_dd(self) -> Tensor:
        """Unbiased covariance estimate of paired deltas."""
        if self.n_delta.item() <= 1:
            raise RuntimeError("At least two delta samples are required.")
        covariance = self.sigma_dd_ / (self.n_delta - 1).to(self.sigma_dd_.dtype)
        return self._symmetrize(covariance)

    @staticmethod
    def _symmetrize(matrix: Tensor) -> Tensor:
        return (matrix + matrix.mH) / 2

    @classmethod
    def _eigh_psd(
        cls,
        matrix: Tensor,
        *,
        ridge: float = 0.0,
    ) -> tuple[Tensor, Tensor]:
        matrix = cls._symmetrize(matrix)
        if ridge:
            matrix = matrix + ridge * torch.eye(
                matrix.shape[0],
                device=matrix.device,
                dtype=matrix.dtype,
            )
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return eigenvalues.clamp_min(0), eigenvectors

    @staticmethod
    def _top_components(
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        *,
        rank: int,
        tol: float,
    ) -> tuple[Tensor, Tensor]:
        order = torch.argsort(eigenvalues, descending=True)[:rank]
        selected_values = eigenvalues[order]
        selected_vectors = eigenvectors[:, order]

        # Relative threshold when a non-zero spectrum exists; otherwise absolute.
        if selected_values.numel() == 0:
            return selected_values, selected_vectors
        scale = eigenvalues.max().clamp_min(1.0)
        keep = selected_values > tol * scale
        return selected_values[keep], selected_vectors[:, keep]

    def make_eraser(self) -> PairedDeltaPcaEraser:
        """Build a fresh eraser from the currently accumulated statistics."""
        delta_covariance = self.sigma_dd

        if not self.whitening:
            eigenvalues, eigenvectors = self._eigh_psd(delta_covariance)
            selected_values, components = self._top_components(
                eigenvalues,
                eigenvectors,
                rank=self.rank,
                tol=self.svd_tol,
            )
            proj_left = components
            proj_right = components.mH
        else:
            feature_covariance = self.sigma_xx
            values_x, vectors_x = self._eigh_psd(
                feature_covariance,
                ridge=self.ridge,
            )

            threshold_x = self.svd_tol * values_x.max().clamp_min(1.0)
            valid = values_x > threshold_x
            inv_sqrt_values = torch.where(valid, values_x.rsqrt(), 0.0)
            sqrt_values = torch.where(valid, values_x.sqrt(), 0.0)

            x_inv_sqrt = (vectors_x * inv_sqrt_values) @ vectors_x.mH
            x_sqrt = (vectors_x * sqrt_values) @ vectors_x.mH

            whitened_delta_covariance = self._symmetrize(
                x_inv_sqrt @ delta_covariance @ x_inv_sqrt.mH
            )
            eigenvalues, eigenvectors = self._eigh_psd(whitened_delta_covariance)
            selected_values, components = self._top_components(
                eigenvalues,
                eigenvectors,
                rank=self.rank,
                tol=self.svd_tol,
            )

            # Column-vector correction:
            #   A^(1/2) U U^H A^(-1/2) x
            # The eraser operates on row vectors, hence the factor order used by
            # proj_left/proj_right below.
            proj_left = x_sqrt @ components
            proj_right = components.mH @ x_inv_sqrt

        return PairedDeltaPcaEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            bias=self.mean_x.clone() if self.affine else None,
            eigenvalues=selected_values,
            requested_rank=self.rank,
            whitening=self.whitening,
        )
