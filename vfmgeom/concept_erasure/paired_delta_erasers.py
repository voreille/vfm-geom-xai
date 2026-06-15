from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal

import torch
from torch import Tensor

try:
    from .shrinkage import optimal_linear_shrinkage
except ImportError:
    optimal_linear_shrinkage = None


DeltaMoment = Literal["covariance", "second_moment"]


@dataclass(frozen=True)
class PairedDeltaPcaEraser:
    """Erase a nuisance subspace estimated from paired deltas."""

    proj_left: Tensor
    proj_right: Tensor
    bias: Tensor | None
    eigenvalues: Tensor
    requested_rank: int
    whitening: bool
    delta_moment: DeltaMoment

    @property
    def rank(self) -> int:
        return int(self.proj_left.shape[1])

    @property
    def P(self) -> Tensor:
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        centered = work - self.bias if self.bias is not None else work
        correction = (centered @ self.proj_right.mH) @ self.proj_left.mH
        output = centered - correction
        if self.bias is not None:
            output = output + self.bias
        return output.to(device=input_device, dtype=input_dtype)

    def transform(self, x: Tensor) -> Tensor:
        return self(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "proj_left": self.proj_left,
            "proj_right": self.proj_right,
            "bias": self.bias,
            "eigenvalues": self.eigenvalues,
            "requested_rank": self.requested_rank,
            "whitening": self.whitening,
            "delta_moment": self.delta_moment,
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
            delta_moment=state.get("delta_moment", "covariance"),
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
            delta_moment=self.delta_moment,
        )

    def transform_delta(self, delta: Tensor) -> Tensor:
        input_device = delta.device
        input_dtype = delta.dtype

        work = delta.to(
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )

        correction = (work @ self.proj_right.mH) @ self.proj_left.mH

        output = work - correction

        return output.to(
            device=input_device,
            dtype=input_dtype,
        )


@dataclass(frozen=True)
class SoftDeltaProjectionEraser:
    """Immutable soft paired-delta eraser."""

    P: Tensor | None
    proj_left: Tensor | None
    proj_right: Tensor | None
    bias: Tensor | None
    lam: float
    rank: int | None
    delta_moment: DeltaMoment

    @property
    def is_low_rank(self) -> bool:
        return self.P is None

    @property
    def full_P(self) -> Tensor:
        if self.P is not None:
            return self.P
        if self.proj_left is None or self.proj_right is None:
            raise RuntimeError("Missing low-rank factors.")
        eye = torch.eye(
            self.proj_left.shape[0],
            device=self.proj_left.device,
            dtype=self.proj_left.dtype,
        )
        return eye - self.proj_left @ self.proj_right

    def __call__(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        reference = self.P if self.P is not None else self.proj_left
        if reference is None:
            raise RuntimeError("Either P or low-rank factors must be provided.")
        work = x.to(device=reference.device, dtype=reference.dtype)
        centered = work - self.bias if self.bias is not None else work
        if self.P is not None:
            output = centered @ self.P.mH
        else:
            if self.proj_left is None or self.proj_right is None:
                raise RuntimeError("Either P or low-rank factors must be provided.")
            correction = (centered @ self.proj_right.mH) @ self.proj_left.mH
            output = centered - correction
        if self.bias is not None:
            output = output + self.bias
        return output.to(device=input_device, dtype=input_dtype)

    def transform(self, x: Tensor) -> Tensor:
        return self(x)

    def state_dict(self) -> dict[str, Any]:
        return {
            "P": self.P,
            "proj_left": self.proj_left,
            "proj_right": self.proj_right,
            "bias": self.bias,
            "lam": self.lam,
            "rank": self.rank,
            "delta_moment": self.delta_moment,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "SoftDeltaProjectionEraser":
        return cls(
            P=state["P"],
            proj_left=state["proj_left"],
            proj_right=state["proj_right"],
            bias=state["bias"],
            lam=float(state["lam"]),
            rank=state["rank"],
            delta_moment=state.get("delta_moment", "second_moment"),
        )

    def save(self, path: str | PathLike[str]) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(
        cls,
        path: str | PathLike[str],
        map_location: torch.device | str | None = None,
    ) -> "SoftDeltaProjectionEraser":
        state = torch.load(path, map_location=map_location, weights_only=True)
        return cls.from_state_dict(state)

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SoftDeltaProjectionEraser":
        return SoftDeltaProjectionEraser(
            P=self.P.to(device=device, dtype=dtype) if self.P is not None else None,
            proj_left=(
                self.proj_left.to(device=device, dtype=dtype)
                if self.proj_left is not None
                else None
            ),
            proj_right=(
                self.proj_right.to(device=device, dtype=dtype)
                if self.proj_right is not None
                else None
            ),
            bias=(
                self.bias.to(device=device, dtype=dtype)
                if self.bias is not None
                else None
            ),
            lam=self.lam,
            rank=self.rank,
            delta_moment=self.delta_moment,
        )

    def transform_delta(self, delta: Tensor) -> Tensor:
        input_device = delta.device
        input_dtype = delta.dtype

        reference = self.P if self.P is not None else self.proj_left
        if reference is None:
            raise RuntimeError("Missing projection parameters.")

        work = delta.to(
            device=reference.device,
            dtype=reference.dtype,
        )

        if self.P is not None:
            output = work @ self.P.mH
        else:
            if self.proj_left is None or self.proj_right is None:
                raise RuntimeError("Missing low-rank factors.")

            correction = (work @ self.proj_right.mH) @ self.proj_left.mH
            output = work - correction

        return output.to(
            device=input_device,
            dtype=input_dtype,
        )


class PairedDeltaFitter:
    """Accumulate statistics shared by PCA and soft paired-delta erasure."""

    @classmethod
    def fit(cls, x: Tensor, delta: Tensor, **kwargs: Any) -> "PairedDeltaFitter":
        fitter = cls(x_dim=x.shape[-1], device=x.device, dtype=x.dtype, **kwargs)
        return fitter.update(x=x, delta=delta)

    @classmethod
    def fit_from_pairs(
        cls,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
        **kwargs: Any,
    ) -> "PairedDeltaFitter":
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )
        return cls.fit(x=x, delta=x1 - x2, **kwargs)

    def __init__(
        self,
        x_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        self.x_dim = x_dim
        self.mean_x = torch.zeros(x_dim, device=device, dtype=dtype)
        self.mean_delta = torch.zeros(x_dim, device=device, dtype=dtype)
        self.m2_x = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)
        self.m2_delta = torch.zeros(x_dim, x_dim, device=device, dtype=dtype)
        self.n_x = torch.tensor(0, device=device, dtype=torch.long)
        self.n_delta = torch.tensor(0, device=device, dtype=torch.long)

    @torch.no_grad()
    def update(self, x: Tensor, delta: Tensor) -> "PairedDeltaFitter":
        x = self._prepare(x, name="x", reference=self.mean_x)
        delta = self._prepare(delta, name="delta", reference=self.mean_delta)
        self._update_mean_and_m2(x, self.mean_x, self.m2_x, self.n_x)
        self._update_mean_and_m2(delta, self.mean_delta, self.m2_delta, self.n_delta)
        return self

    @torch.no_grad()
    def update_from_pairs(
        self,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
    ) -> "PairedDeltaFitter":
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )
        return self.update(x=x, delta=x1 - x2)

    def _prepare(self, value: Tensor, *, name: str, reference: Tensor) -> Tensor:
        if value.shape[-1] != self.x_dim:
            raise ValueError(
                f"{name} must have last dimension {self.x_dim}, got {tuple(value.shape)}."
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
        m2: Tensor,
        count: Tensor,
    ) -> None:
        batch_n = batch.shape[0]
        if batch_n == 0:
            return
        old_n = count.clone()
        new_n = old_n + batch_n
        batch_mean = batch.mean(dim=0)
        mean_shift = batch_mean - mean
        centered = batch - batch_mean
        batch_m2 = centered.mH @ centered
        cross_weight = old_n.to(batch.dtype) * batch_n / new_n.to(batch.dtype)
        cross = torch.outer(mean_shift, mean_shift.conj()) * cross_weight
        m2.add_(batch_m2 + cross)
        mean.add_(mean_shift * (batch_n / new_n.to(batch.dtype)))
        count.copy_(new_n)

    @staticmethod
    def _symmetrize(matrix: Tensor) -> Tensor:
        return (matrix + matrix.mH) / 2

    @classmethod
    def _eigh_psd(cls, matrix: Tensor) -> tuple[Tensor, Tensor]:
        eigenvalues, eigenvectors = torch.linalg.eigh(cls._symmetrize(matrix))
        return eigenvalues.clamp_min(0), eigenvectors

    @staticmethod
    def _validate_numerics(*, ridge: float, svd_tol: float) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if svd_tol < 0:
            raise ValueError("svd_tol must be non-negative.")

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
        if selected_values.numel() == 0:
            return selected_values, selected_vectors
        scale = eigenvalues.max().clamp_min(1.0)
        keep = selected_values > tol * scale
        return selected_values[keep], selected_vectors[:, keep]

    def covariance_x(self, *, shrinkage: bool) -> Tensor:
        if self.n_x.item() <= 1:
            raise RuntimeError("At least two X samples are required.")
        m2 = self._symmetrize(self.m2_x)
        if shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = m2 / self.n_x.to(m2.dtype)
            covariance = optimal_linear_shrinkage(
                covariance_mle,
                self.n_x,
                inplace=False,
            )
        else:
            covariance = m2 / (self.n_x - 1).to(m2.dtype)
        return self._symmetrize(covariance)

    def covariance_delta(self, *, shrinkage: bool = False) -> Tensor:
        if self.n_delta.item() <= 1:
            raise RuntimeError("At least two delta samples are required.")
        m2 = self._symmetrize(self.m2_delta)
        if shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = m2 / self.n_delta.to(m2.dtype)
            covariance = optimal_linear_shrinkage(
                covariance_mle,
                self.n_delta,
                inplace=False,
            )
        else:
            covariance = m2 / (self.n_delta - 1).to(m2.dtype)
        return self._symmetrize(covariance)

    def second_moment_delta(self, *, shrinkage: bool = False) -> Tensor:
        if self.n_delta.item() <= 0:
            raise RuntimeError("At least one delta sample is required.")
        if self.n_delta.item() == 1:
            covariance_population = torch.zeros_like(self.m2_delta)
        elif shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = self._symmetrize(self.m2_delta) / self.n_delta.to(
                self.m2_delta.dtype
            )
            covariance_population = optimal_linear_shrinkage(
                covariance_mle,
                self.n_delta,
                inplace=False,
            )
        else:
            covariance_population = self._symmetrize(self.m2_delta) / self.n_delta.to(
                self.m2_delta.dtype
            )
        mean_outer = torch.outer(self.mean_delta, self.mean_delta.conj())
        return self._symmetrize(covariance_population + mean_outer)

    def delta_matrix(
        self,
        *,
        moment: DeltaMoment,
        shrinkage: bool = False,
    ) -> Tensor:
        if moment == "covariance":
            return self.covariance_delta(shrinkage=shrinkage)
        if moment == "second_moment":
            return self.second_moment_delta(shrinkage=shrinkage)
        raise ValueError(f"Unknown delta moment {moment!r}.")

    def whitening_matrices(
        self,
        *,
        shrinkage: bool,
        ridge: float,
        svd_tol: float,
    ) -> tuple[Tensor, Tensor]:
        self._validate_numerics(ridge=ridge, svd_tol=svd_tol)
        covariance = self.covariance_x(shrinkage=shrinkage)
        eye = torch.eye(
            self.x_dim,
            device=covariance.device,
            dtype=covariance.dtype,
        )
        covariance_reg = self._symmetrize(covariance + ridge * eye)
        eigenvalues, eigenvectors = self._eigh_psd(covariance_reg)
        threshold = svd_tol * eigenvalues.max().clamp_min(1.0)
        valid = eigenvalues > threshold
        inv_sqrt_values = torch.where(valid, eigenvalues.rsqrt(), 0.0)
        sqrt_values = torch.where(valid, eigenvalues.sqrt(), 0.0)
        inv_sqrt = (eigenvectors * inv_sqrt_values) @ eigenvectors.mH
        sqrt = (eigenvectors * sqrt_values) @ eigenvectors.mH
        return self._symmetrize(inv_sqrt), self._symmetrize(sqrt)

    @torch.no_grad()
    def make_pca_eraser(
        self,
        *,
        rank: int,
        whitening: bool = False,
        affine: bool = True,
        delta_moment: DeltaMoment = "covariance",
        shrink_A: bool = True,
        shrink_B: bool = False,
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> PairedDeltaPcaEraser:
        if not 1 <= rank <= self.x_dim:
            raise ValueError(f"rank must be between 1 and {self.x_dim}.")
        self._validate_numerics(ridge=ridge, svd_tol=svd_tol)
        delta_matrix = self.delta_matrix(
            moment=delta_moment,
            shrinkage=shrink_B,
        )
        if not whitening:
            eigenvalues, eigenvectors = self._eigh_psd(delta_matrix)
            selected_values, components = self._top_components(
                eigenvalues,
                eigenvectors,
                rank=rank,
                tol=svd_tol,
            )
            proj_left = components
            proj_right = components.mH
        else:
            x_inv_sqrt, x_sqrt = self.whitening_matrices(
                shrinkage=shrink_A,
                ridge=ridge,
                svd_tol=svd_tol,
            )
            whitened_delta_matrix = self._symmetrize(
                x_inv_sqrt @ delta_matrix @ x_inv_sqrt.mH
            )
            eigenvalues, eigenvectors = self._eigh_psd(whitened_delta_matrix)
            selected_values, components = self._top_components(
                eigenvalues,
                eigenvectors,
                rank=rank,
                tol=svd_tol,
            )
            proj_left = x_sqrt @ components
            proj_right = components.mH @ x_inv_sqrt
        return PairedDeltaPcaEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            bias=self.mean_x.clone() if affine else None,
            eigenvalues=selected_values,
            requested_rank=rank,
            whitening=whitening,
            delta_moment=delta_moment,
        )

    @torch.no_grad()
    def make_soft_eraser(
        self,
        *,
        lam: float,
        rank: int | None = None,
        affine: bool = True,
        delta_moment: DeltaMoment = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> SoftDeltaProjectionEraser:
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        if rank is not None and not 1 <= rank <= self.x_dim:
            raise ValueError(f"rank must be between 1 and {self.x_dim}, or None.")
        self._validate_numerics(ridge=ridge, svd_tol=svd_tol)
        A = self.covariance_x(shrinkage=shrink_A)
        B = self.delta_matrix(moment=delta_moment, shrinkage=shrink_B)
        eye = torch.eye(self.x_dim, device=A.device, dtype=A.dtype)
        A_reg = A + ridge * eye
        C = self._symmetrize(A_reg + float(lam) * B)
        P_full = torch.linalg.solve(C.mH, A_reg.mH).mH
        P_full = P_full.to(dtype=A.dtype)
        bias = self.mean_x.clone() if affine else None
        if rank is None:
            return SoftDeltaProjectionEraser(
                P=P_full,
                proj_left=None,
                proj_right=None,
                bias=bias,
                lam=float(lam),
                rank=None,
                delta_moment=delta_moment,
            )
        residual = eye - P_full
        U, singular_values, Vh = torch.linalg.svd(residual, full_matrices=False)
        r = min(rank, singular_values.numel())
        selected_values = singular_values[:r]
        keep = selected_values > svd_tol
        U_r = U[:, :r][:, keep]
        S_r = selected_values[keep]
        Vh_r = Vh[:r, :][keep]
        proj_left = U_r * S_r.sqrt()
        proj_right = Vh_r * S_r.sqrt().unsqueeze(1)
        return SoftDeltaProjectionEraser(
            P=None,
            proj_left=proj_left,
            proj_right=proj_right,
            bias=bias,
            lam=float(lam),
            rank=int(proj_left.shape[1]),
            delta_moment=delta_moment,
        )
