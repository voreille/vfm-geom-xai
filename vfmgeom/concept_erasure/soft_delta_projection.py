from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


try:
    from .shrinkage import optimal_linear_shrinkage
except ImportError:
    optimal_linear_shrinkage = None


@dataclass(frozen=True)
class SoftDeltaProjectionEraser:
    """Immutable soft paired-delta eraser.

    Applies

        x' = bias + P (x - bias)

    where the full-rank optimum is

        P = A (A + lambda B)^-1,

    with

        A = Cov(X)
        B = E[Delta Delta^T].

    In low-rank mode, the residual Q = I - P is stored implicitly as

        Q_r = proj_left @ proj_right.
    """

    P: Tensor | None
    proj_left: Tensor | None
    proj_right: Tensor | None
    bias: Tensor | None
    lam: float
    rank: int | None

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
        centered = x - self.bias if self.bias is not None else x

        if self.P is not None:
            out = centered @ self.P.mH
        else:
            if self.proj_left is None or self.proj_right is None:
                raise RuntimeError("Either P or low-rank factors must be provided.")

            correction = (centered @ self.proj_right.mH) @ self.proj_left.mH
            out = centered - correction

        if self.bias is not None:
            out = out + self.bias

        return out.type_as(x)

    def to(
        self,
        device: torch.device | str,
    ) -> "SoftDeltaProjectionEraser":
        return SoftDeltaProjectionEraser(
            P=self.P.to(device) if self.P is not None else None,
            proj_left=(
                self.proj_left.to(device) if self.proj_left is not None else None
            ),
            proj_right=(
                self.proj_right.to(device) if self.proj_right is not None else None
            ),
            bias=self.bias.to(device) if self.bias is not None else None,
            lam=self.lam,
            rank=self.rank,
        )


class SoftDeltaProjectionFitter:
    """Streaming fitter for soft paired-delta projection.

    Objective:

        min_P E ||P X - X||^2 + lambda E ||P Delta||^2

    with Delta = X1 - X2.

    Full-rank closed form:

        P = A (A + lambda B)^-1

    where

        A = Cov(X)
        B = E[Delta Delta^T].

    The fitter accumulates data-dependent statistics with update(). Immutable
    erasers with different lambda and rank values are then derived using
    make_eraser().
    """

    @classmethod
    def fit(
        cls,
        x: Tensor,
        delta: Tensor,
        **kwargs,
    ) -> "SoftDeltaProjectionFitter":
        d = x.shape[-1]
        fitter = cls(
            x_dim=d,
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
        **kwargs,
    ) -> "SoftDeltaProjectionFitter":
        return cls.fit(
            x=x,
            delta=x1 - x2,
            **kwargs,
        )

    def __init__(
        self,
        x_dim: int,
        *,
        affine: bool = True,
        shrink_A: bool = True,
        shrink_B: bool = False,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")

        self.x_dim = x_dim
        self.affine = affine
        self.shrink_A = shrink_A
        self.shrink_B = shrink_B

        self.mean_x = torch.zeros(
            x_dim,
            device=device,
            dtype=dtype,
        )
        self.mean_delta = torch.zeros(
            x_dim,
            device=device,
            dtype=dtype,
        )

        self.sigma_xx_ = torch.zeros(
            x_dim,
            x_dim,
            device=device,
            dtype=dtype,
        )
        self.sigma_dd_ = torch.zeros(
            x_dim,
            x_dim,
            device=device,
            dtype=dtype,
        )

        self.n_x = torch.tensor(
            0,
            device=device,
            dtype=torch.long,
        )
        self.n_delta = torch.tensor(
            0,
            device=device,
            dtype=torch.long,
        )

    @torch.no_grad()
    def update(
        self,
        x: Tensor,
        delta: Tensor,
    ) -> "SoftDeltaProjectionFitter":
        x = x.reshape(-1, self.x_dim).to(
            device=self.mean_x.device,
            dtype=self.mean_x.dtype,
        )
        delta = delta.reshape(-1, self.x_dim).to(
            device=self.mean_delta.device,
            dtype=self.mean_delta.dtype,
        )

        self._update_x(x)
        self._update_delta(delta)
        return self

    @torch.no_grad()
    def update_from_pairs(
        self,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
    ) -> "SoftDeltaProjectionFitter":
        return self.update(
            x=x,
            delta=x1 - x2,
        )

    @torch.no_grad()
    def _update_x(self, x: Tensor) -> None:
        n = x.shape[0]
        if n == 0:
            return

        self.n_x += n

        delta = x - self.mean_x
        self.mean_x += delta.sum(dim=0) / self.n_x
        delta2 = x - self.mean_x

        self.sigma_xx_.addmm_(
            delta.mH,
            delta2,
        )

    @torch.no_grad()
    def _update_delta(self, delta_batch: Tensor) -> None:
        n = delta_batch.shape[0]
        if n == 0:
            return

        self.n_delta += n

        delta = delta_batch - self.mean_delta
        self.mean_delta += delta.sum(dim=0) / self.n_delta
        delta2 = delta_batch - self.mean_delta

        self.sigma_dd_.addmm_(
            delta.mH,
            delta2,
        )

    @property
    def sigma_xx(self) -> Tensor:
        """Sample covariance of X."""
        if self.n_x <= 1:
            raise RuntimeError("Call update() with at least two X samples first.")

        S = (self.sigma_xx_ + self.sigma_xx_.mH) / 2
        S = S / (self.n_x - 1)

        if self.shrink_A:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrink_A=True, but optimal_linear_shrinkage could not be imported."
                )

            S = optimal_linear_shrinkage(
                S,
                self.n_x,
                inplace=False,
            )

        return (S + S.mH) / 2

    @property
    def second_moment_dd(self) -> Tensor:
        """Second moment E[Delta Delta^T].

        The objective penalizes E||P Delta||^2, so the required matrix is the
        uncentered second moment, not merely Cov(Delta).

        Using the running centered statistics:

            E[Delta Delta^T]
            = Cov_population(Delta) + mean_delta mean_delta^T.
        """
        if self.n_delta <= 0:
            raise RuntimeError("Call update() with at least one delta sample first.")

        covariance = (self.sigma_dd_ + self.sigma_dd_.mH) / 2
        covariance = covariance / self.n_delta

        mean_outer = torch.outer(
            self.mean_delta,
            self.mean_delta.conj(),
        )
        second_moment = covariance + mean_outer

        if self.shrink_B:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrink_B=True, but optimal_linear_shrinkage could not be imported."
                )

            second_moment = optimal_linear_shrinkage(
                second_moment,
                self.n_delta,
                inplace=False,
            )

        return (second_moment + second_moment.mH) / 2

    @torch.no_grad()
    def make_eraser(
        self,
        lam: float,
        rank: int | None = None,
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> SoftDeltaProjectionEraser:
        """Create one immutable eraser from the fitted statistics."""
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        if rank is not None and not 1 <= rank <= self.x_dim:
            raise ValueError(f"rank must be between 1 and {self.x_dim}, or None.")
        if ridge < 0:
            raise ValueError("ridge must be non-negative.")
        if svd_tol < 0:
            raise ValueError("svd_tol must be non-negative.")

        d = self.x_dim
        device = self.mean_x.device
        dtype = self.mean_x.dtype

        eye = torch.eye(
            d,
            device=device,
            dtype=dtype,
        )

        A = self.sigma_xx
        B = self.second_moment_dd

        # Stabilize the preservation geometry and the solve.
        A_reg = A + ridge * eye
        C = A_reg + lam * B
        C = (C + C.mH) / 2

        # P = A_reg @ inv(C), evaluated without an explicit inverse.
        P_full = torch.linalg.solve(
            C.mH,
            A_reg.mH,
        ).mH
        P_full = P_full.to(dtype=dtype)

        bias = self.mean_x.clone() if self.affine else None

        if rank is None:
            return SoftDeltaProjectionEraser(
                P=P_full,
                proj_left=None,
                proj_right=None,
                bias=bias,
                lam=float(lam),
                rank=None,
            )

        # Low-rank residual approximation:
        # P ~= I - Q_r, where Q = I - P_full.
        Q = eye - P_full
        U, S, Vh = torch.linalg.svd(
            Q,
            full_matrices=False,
        )

        r = min(rank, S.numel())
        keep = S[:r] > svd_tol

        U_r = U[:, :r][:, keep]
        S_r = S[:r][keep]
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
        )
