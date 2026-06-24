from __future__ import annotations

from dataclasses import dataclass
from os import PathLike
from typing import Any, Literal, Mapping, Sequence

import torch
from torch import Tensor

try:
    from .shrinkage import optimal_linear_shrinkage
except ImportError:
    optimal_linear_shrinkage = None


DeltaMoment = Literal["covariance", "second_moment"]
MomentNormalization = Literal["none", "trace", "frobenius"]
JointNormalization = Literal["none", "match_x_trace"]


@dataclass(frozen=True)
class DeltaSourceSpec:
    """Describe how one nuisance-delta source contributes to the joint moment.

    Parameters
    ----------
    name:
        Name used when the delta source was added to :class:`PairedDeltaFitter`.
    weight:
        Relative source weight. When ``normalize_source_weights=True`` in an
        eraser builder, positive weights are normalized to sum to one.
    moment:
        Use either the centered covariance or the uncentered second moment.
    shrinkage:
        Apply linear shrinkage to the covariance part of this source.
    normalization:
        Optional scale normalization before combining source matrices.
        ``trace`` is useful when sources have very different absolute energy.
    """

    name: str
    weight: float = 1.0
    moment: DeltaMoment = "second_moment"
    shrinkage: bool = False
    normalization: MomentNormalization = "none"

    @classmethod
    def from_value(
        cls, value: "DeltaSourceSpec | Mapping[str, Any]"
    ) -> "DeltaSourceSpec":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise TypeError(
                "Delta source specifications must be DeltaSourceSpec objects "
                "or mappings."
            )
        return cls(
            name=str(value["name"]),
            weight=float(value.get("weight", 1.0)),
            moment=value.get("moment", "second_moment"),
            shrinkage=bool(value.get("shrinkage", False)),
            normalization=value.get("normalization", "none"),
        )


@dataclass
class _RunningMoments:
    mean: Tensor
    m2: Tensor
    count: Tensor

    @classmethod
    def create(
        cls,
        dim: int,
        *,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> "_RunningMoments":
        return cls(
            mean=torch.zeros(dim, device=device, dtype=dtype),
            m2=torch.zeros(dim, dim, device=device, dtype=dtype),
            count=torch.tensor(0, device=device, dtype=torch.long),
        )


@dataclass(frozen=True)
class PairedDeltaPcaEraser:
    """Erase top directions of a combined nuisance-delta moment matrix."""

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

    def apply_linear(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        correction = (work @ self.proj_right.mH) @ self.proj_left.mH
        return (work - correction).to(device=input_device, dtype=input_dtype)

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        work = x.to(device=self.proj_left.device, dtype=self.proj_left.dtype)
        centered = work - self.bias if self.bias is not None else work
        output = self.apply_linear(centered)
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
    def from_state_dict(cls, state: Mapping[str, Any]) -> "PairedDeltaPcaEraser":
        return cls(
            proj_left=state["proj_left"],
            proj_right=state["proj_right"],
            bias=state.get("bias"),
            eigenvalues=state.get(
                "eigenvalues",
                torch.empty(
                    state["proj_left"].shape[1],
                    device=state["proj_left"].device,
                    dtype=state["proj_left"].dtype,
                ),
            ),
            requested_rank=int(
                state.get("requested_rank", state["proj_left"].shape[1])
            ),
            whitening=bool(state.get("whitening", False)),
            delta_moment=state.get("delta_moment", "second_moment"),
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
    joint_normalization: JointNormalization = "none"

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

    def apply_linear(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        reference = self.P if self.P is not None else self.proj_left
        if reference is None:
            raise RuntimeError("Missing projection parameters.")
        work = x.to(device=reference.device, dtype=reference.dtype)

        if self.P is not None:
            output = work @ self.P.mH
        else:
            if self.proj_left is None or self.proj_right is None:
                raise RuntimeError("Missing low-rank factors.")
            correction = (work @ self.proj_right.mH) @ self.proj_left.mH
            output = work - correction

        return output.to(device=input_device, dtype=input_dtype)

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        input_device = x.device
        input_dtype = x.dtype
        reference = self.P if self.P is not None else self.proj_left
        if reference is None:
            raise RuntimeError("Missing projection parameters.")
        work = x.to(device=reference.device, dtype=reference.dtype)
        centered = work - self.bias if self.bias is not None else work
        output = self.apply_linear(centered)
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
            "joint_normalization": self.joint_normalization,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SoftDeltaProjectionEraser":
        return cls(
            P=state.get("P"),
            proj_left=state.get("proj_left"),
            proj_right=state.get("proj_right"),
            bias=state.get("bias"),
            lam=float(state["lam"]),
            rank=state.get("rank"),
            delta_moment=state.get("delta_moment", "second_moment"),
            joint_normalization=state.get("joint_normalization", "none"),
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
        def move(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.to(device=device, dtype=dtype)

        return SoftDeltaProjectionEraser(
            P=move(self.P),
            proj_left=move(self.proj_left),
            proj_right=move(self.proj_right),
            bias=move(self.bias),
            lam=self.lam,
            rank=self.rank,
            delta_moment=self.delta_moment,
            joint_normalization=self.joint_normalization,
        )


class PairedDeltaFitter:
    """Accumulate X statistics and any number of named delta sources.

    PCA and soft erasure both operate on a joint nuisance matrix

        B_joint = sum_k w_k * normalize(B_k),

    where one ``B_k`` is estimated independently for every source. For PCA,
    the top eigendirections of ``B_joint`` (or its X-whitened form) are removed.
    For the soft method, ``B_joint`` enters

        P = A (A + lambda B_joint)^-1.

    Keeping source moments separate avoids accidental weighting by the number of
    generated deltas and allows scanner and stain sources to be balanced
    explicitly.
    """

    DEFAULT_SOURCE_NAME = "default"

    @classmethod
    def fit(
        cls,
        x: Tensor,
        delta: Tensor | None = None,
        *,
        source_name: str = DEFAULT_SOURCE_NAME,
        **kwargs: Any,
    ) -> "PairedDeltaFitter":
        fitter = cls(x_dim=x.shape[-1], device=x.device, dtype=x.dtype, **kwargs)
        fitter.update_x(x)
        if delta is not None:
            fitter.update_delta_source(source_name, delta)
        return fitter

    @classmethod
    def fit_sources(
        cls,
        x: Tensor,
        delta_sources: Mapping[str, Tensor],
        **kwargs: Any,
    ) -> "PairedDeltaFitter":
        fitter = cls.fit(x=x, **kwargs)
        for name, delta in delta_sources.items():
            fitter.update_delta_source(name, delta)
        return fitter

    @classmethod
    def fit_from_pairs(
        cls,
        x: Tensor,
        x1: Tensor,
        x2: Tensor,
        *,
        source_name: str = DEFAULT_SOURCE_NAME,
        **kwargs: Any,
    ) -> "PairedDeltaFitter":
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )
        return cls.fit(
            x=x,
            delta=x1 - x2,
            source_name=source_name,
            **kwargs,
        )

    def __init__(
        self,
        x_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        if x_dim <= 0:
            raise ValueError("x_dim must be positive.")
        self.x_dim = int(x_dim)
        self._x_stats = _RunningMoments.create(
            self.x_dim,
            device=device,
            dtype=dtype,
        )
        self._delta_stats: dict[str, _RunningMoments] = {}

    @property
    def mean_x(self) -> Tensor:
        return self._x_stats.mean

    @property
    def n_x(self) -> Tensor:
        return self._x_stats.count

    @property
    def delta_source_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._delta_stats))

    def delta_count(self, name: str) -> int:
        return int(self._get_delta_stats(name).count.item())

    @torch.no_grad()
    def update_x(self, x: Tensor) -> "PairedDeltaFitter":
        x = self._prepare(x, name="x", reference=self._x_stats.mean)
        self._update_running_moments(x, self._x_stats)
        return self

    @torch.no_grad()
    def update_delta_source(
        self,
        name: str,
        delta: Tensor,
    ) -> "PairedDeltaFitter":
        name = str(name)
        if not name:
            raise ValueError("Delta source name must be non-empty.")

        stats = self._delta_stats.get(name)
        if stats is None:
            stats = _RunningMoments.create(
                self.x_dim,
                device=self._x_stats.mean.device,
                dtype=self._x_stats.mean.dtype,
            )
            self._delta_stats[name] = stats

        delta = self._prepare(delta, name=f"delta[{name}]", reference=stats.mean)
        self._update_running_moments(delta, stats)
        return self

    @torch.no_grad()
    def update(
        self,
        x: Tensor | None = None,
        delta: Tensor | None = None,
        *,
        source_name: str = DEFAULT_SOURCE_NAME,
    ) -> "PairedDeltaFitter":
        """Backward-compatible update method."""
        if x is None and delta is None:
            raise ValueError("At least one of x or delta must be provided.")
        if x is not None:
            self.update_x(x)
        if delta is not None:
            self.update_delta_source(source_name, delta)
        return self

    @torch.no_grad()
    def update_from_pairs(
        self,
        x: Tensor | None,
        x1: Tensor,
        x2: Tensor,
        *,
        source_name: str = DEFAULT_SOURCE_NAME,
    ) -> "PairedDeltaFitter":
        if x1.shape != x2.shape:
            raise ValueError(
                f"x1 and x2 must have identical shapes, got {x1.shape} and {x2.shape}."
            )
        return self.update(x=x, delta=x1 - x2, source_name=source_name)

    def _prepare(self, value: Tensor, *, name: str, reference: Tensor) -> Tensor:
        if value.shape[-1] != self.x_dim:
            raise ValueError(
                f"{name} must have last dimension {self.x_dim}, "
                f"got {tuple(value.shape)}."
            )
        return value.reshape(-1, self.x_dim).to(
            device=reference.device,
            dtype=reference.dtype,
        )

    @staticmethod
    @torch.no_grad()
    def _update_running_moments(batch: Tensor, stats: _RunningMoments) -> None:
        batch_n = batch.shape[0]
        if batch_n == 0:
            return

        old_n = stats.count.clone()
        new_n = old_n + batch_n
        batch_mean = batch.mean(dim=0)
        mean_shift = batch_mean - stats.mean
        centered = batch - batch_mean
        batch_m2 = centered.mH @ centered

        cross_weight = old_n.to(batch.dtype) * batch_n / new_n.to(batch.dtype)
        cross = torch.outer(mean_shift, mean_shift.conj()) * cross_weight

        stats.m2.add_(batch_m2 + cross)
        stats.mean.add_(mean_shift * (batch_n / new_n.to(batch.dtype)))
        stats.count.copy_(new_n)

    def _get_delta_stats(self, name: str) -> _RunningMoments:
        try:
            return self._delta_stats[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown delta source {name!r}. Available: {self.delta_source_names}"
            ) from exc

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

    @staticmethod
    def _validate_source_specs(
        specs: Sequence[DeltaSourceSpec],
    ) -> None:
        if not specs:
            raise ValueError("At least one delta source specification is required.")
        names = [spec.name for spec in specs]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate delta source names in recipe: {names}")
        if any(spec.weight < 0 for spec in specs):
            raise ValueError("Delta source weights must be non-negative.")
        if not any(spec.weight > 0 for spec in specs):
            raise ValueError("At least one delta source weight must be positive.")
        allowed_moments = {"covariance", "second_moment"}
        allowed_normalizations = {"none", "trace", "frobenius"}
        for spec in specs:
            if spec.moment not in allowed_moments:
                raise ValueError(f"Unknown delta moment {spec.moment!r}.")
            if spec.normalization not in allowed_normalizations:
                raise ValueError(
                    f"Unknown moment normalization {spec.normalization!r}."
                )

    def covariance_x(self, *, shrinkage: bool) -> Tensor:
        stats = self._x_stats
        if stats.count.item() <= 1:
            raise RuntimeError("At least two X samples are required.")
        m2 = self._symmetrize(stats.m2)
        if shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = m2 / stats.count.to(m2.dtype)
            covariance = optimal_linear_shrinkage(
                covariance_mle,
                stats.count,
                inplace=False,
            )
        else:
            covariance = m2 / (stats.count - 1).to(m2.dtype)
        return self._symmetrize(covariance)

    def covariance_delta_source(
        self,
        name: str,
        *,
        shrinkage: bool = False,
    ) -> Tensor:
        stats = self._get_delta_stats(name)
        if stats.count.item() <= 1:
            raise RuntimeError(f"At least two deltas are required for source {name!r}.")
        m2 = self._symmetrize(stats.m2)
        if shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = m2 / stats.count.to(m2.dtype)
            covariance = optimal_linear_shrinkage(
                covariance_mle,
                stats.count,
                inplace=False,
            )
        else:
            covariance = m2 / (stats.count - 1).to(m2.dtype)
        return self._symmetrize(covariance)

    def second_moment_delta_source(
        self,
        name: str,
        *,
        shrinkage: bool = False,
    ) -> Tensor:
        stats = self._get_delta_stats(name)
        if stats.count.item() <= 0:
            raise RuntimeError(f"At least one delta is required for source {name!r}.")

        if stats.count.item() == 1:
            covariance_population = torch.zeros_like(stats.m2)
        elif shrinkage:
            if optimal_linear_shrinkage is None:
                raise ImportError(
                    "shrinkage=True, but optimal_linear_shrinkage could not be imported."
                )
            covariance_mle = self._symmetrize(stats.m2) / stats.count.to(stats.m2.dtype)
            covariance_population = optimal_linear_shrinkage(
                covariance_mle,
                stats.count,
                inplace=False,
            )
        else:
            covariance_population = self._symmetrize(stats.m2) / stats.count.to(
                stats.m2.dtype
            )

        mean_outer = torch.outer(stats.mean, stats.mean.conj())
        return self._symmetrize(covariance_population + mean_outer)

    def delta_matrix_for_source(
        self,
        name: str,
        *,
        moment: DeltaMoment,
        shrinkage: bool = False,
    ) -> Tensor:
        if moment == "covariance":
            return self.covariance_delta_source(name, shrinkage=shrinkage)
        if moment == "second_moment":
            return self.second_moment_delta_source(name, shrinkage=shrinkage)
        raise ValueError(f"Unknown delta moment {moment!r}.")

    @staticmethod
    def _normalize_moment_matrix(
        matrix: Tensor,
        normalization: MomentNormalization,
        *,
        eps: float,
    ) -> Tensor:
        if normalization == "none":
            return matrix
        if normalization == "trace":
            scale = torch.real(torch.trace(matrix)).abs()
        elif normalization == "frobenius":
            scale = torch.linalg.matrix_norm(matrix, ord="fro")
        else:
            raise ValueError(f"Unknown normalization {normalization!r}.")

        if scale <= eps:
            raise RuntimeError(
                f"Cannot apply {normalization!r} normalization to a near-zero moment."
            )
        return matrix / scale

    @staticmethod
    def _match_trace(
        *,
        matrix: Tensor,
        reference: Tensor,
        eps: float,
    ) -> Tensor:
        """Rescale ``matrix`` so its trace matches ``reference``.

        This is useful after per-source nuisance normalization. It makes the
        soft-erasure lambda approximately dimensionless: ``lam=1`` means that
        the joint nuisance penalty has the same total trace as the feature
        covariance used for preservation.
        """
        matrix_trace = torch.real(torch.trace(matrix)).abs()
        reference_trace = torch.real(torch.trace(reference)).abs()

        if matrix_trace <= eps:
            raise RuntimeError("Cannot trace-match a near-zero nuisance matrix.")
        if reference_trace <= eps:
            raise RuntimeError("Cannot trace-match to a near-zero reference matrix.")

        return matrix * (reference_trace / matrix_trace)

    @classmethod
    def _normalize_joint_delta_matrix(
        cls,
        *,
        matrix: Tensor,
        reference: Tensor,
        normalization: JointNormalization,
        eps: float,
    ) -> Tensor:
        if normalization == "none":
            return matrix
        if normalization == "match_x_trace":
            return cls._match_trace(matrix=matrix, reference=reference, eps=eps)
        raise ValueError(f"Unknown joint normalization {normalization!r}.")

    def combined_delta_matrix(
        self,
        source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]],
        *,
        normalize_source_weights: bool = True,
        normalization_eps: float = 1e-12,
    ) -> Tensor:
        specs = [DeltaSourceSpec.from_value(value) for value in source_specs]
        self._validate_source_specs(specs)

        positive_weight_sum = sum(spec.weight for spec in specs if spec.weight > 0)
        combined = torch.zeros(
            self.x_dim,
            self.x_dim,
            device=self.mean_x.device,
            dtype=self.mean_x.dtype,
        )

        for spec in specs:
            if spec.weight == 0:
                continue
            matrix = self.delta_matrix_for_source(
                spec.name,
                moment=spec.moment,
                shrinkage=spec.shrinkage,
            )
            matrix = self._normalize_moment_matrix(
                matrix,
                spec.normalization,
                eps=normalization_eps,
            )
            weight = spec.weight
            if normalize_source_weights:
                weight /= positive_weight_sum
            combined.add_(matrix, alpha=float(weight))

        return self._symmetrize(combined)

    def source_diagnostics(
        self,
        source_specs: Sequence[DeltaSourceSpec | Mapping[str, Any]],
    ) -> dict[str, dict[str, float | int | str | bool]]:
        diagnostics: dict[str, dict[str, float | int | str | bool]] = {}
        for value in source_specs:
            spec = DeltaSourceSpec.from_value(value)
            raw = self.delta_matrix_for_source(
                spec.name,
                moment=spec.moment,
                shrinkage=spec.shrinkage,
            )
            normalized = self._normalize_moment_matrix(
                raw,
                spec.normalization,
                eps=1e-12,
            )
            diagnostics[spec.name] = {
                "n_delta": self.delta_count(spec.name),
                "weight": spec.weight,
                "moment": spec.moment,
                "shrinkage": spec.shrinkage,
                "normalization": spec.normalization,
                "raw_trace": float(torch.real(torch.trace(raw)).item()),
                "raw_frobenius": float(torch.linalg.matrix_norm(raw, ord="fro").item()),
                "normalized_trace": float(torch.real(torch.trace(normalized)).item()),
                "normalized_frobenius": float(
                    torch.linalg.matrix_norm(normalized, ord="fro").item()
                ),
            }
        return diagnostics

    # Backward-compatible aliases for a single unnamed source.
    def covariance_delta(self, *, shrinkage: bool = False) -> Tensor:
        return self.covariance_delta_source(
            self.DEFAULT_SOURCE_NAME,
            shrinkage=shrinkage,
        )

    def second_moment_delta(self, *, shrinkage: bool = False) -> Tensor:
        return self.second_moment_delta_source(
            self.DEFAULT_SOURCE_NAME,
            shrinkage=shrinkage,
        )

    def delta_matrix(
        self,
        *,
        moment: DeltaMoment,
        shrinkage: bool = False,
    ) -> Tensor:
        return self.delta_matrix_for_source(
            self.DEFAULT_SOURCE_NAME,
            moment=moment,
            shrinkage=shrinkage,
        )

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

    def _resolve_joint_delta_matrix(
        self,
        *,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None,
        delta_moment: DeltaMoment,
        shrink_B: bool,
        source_normalization: MomentNormalization,
        normalize_source_weights: bool,
    ) -> Tensor:
        if delta_sources is None:
            delta_sources = [
                DeltaSourceSpec(
                    name=self.DEFAULT_SOURCE_NAME,
                    weight=1.0,
                    moment=delta_moment,
                    shrinkage=shrink_B,
                    normalization=source_normalization,
                )
            ]
        return self.combined_delta_matrix(
            delta_sources,
            normalize_source_weights=normalize_source_weights,
        )

    @torch.no_grad()
    def make_pca_eraser(
        self,
        *,
        rank: int,
        whitening: bool = False,
        affine: bool = True,
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: DeltaMoment = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: MomentNormalization = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> PairedDeltaPcaEraser:
        if not 1 <= rank <= self.x_dim:
            raise ValueError(f"rank must be between 1 and {self.x_dim}.")
        self._validate_numerics(ridge=ridge, svd_tol=svd_tol)

        joint_delta_matrix = self._resolve_joint_delta_matrix(
            delta_sources=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
        )

        if not whitening:
            eigenvalues, eigenvectors = self._eigh_psd(joint_delta_matrix)
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
                x_inv_sqrt @ joint_delta_matrix @ x_inv_sqrt.mH
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
        delta_sources: Sequence[DeltaSourceSpec | Mapping[str, Any]] | None = None,
        normalize_source_weights: bool = True,
        delta_moment: DeltaMoment = "second_moment",
        shrink_A: bool = True,
        shrink_B: bool = False,
        source_normalization: MomentNormalization = "none",
        joint_normalization: JointNormalization = "none",
        ridge: float = 1e-4,
        svd_tol: float = 1e-7,
    ) -> SoftDeltaProjectionEraser:
        if lam < 0:
            raise ValueError("lam must be non-negative.")
        if rank is not None and not 1 <= rank <= self.x_dim:
            raise ValueError(f"rank must be between 1 and {self.x_dim}, or None.")
        self._validate_numerics(ridge=ridge, svd_tol=svd_tol)

        A = self.covariance_x(shrinkage=shrink_A)
        B = self._resolve_joint_delta_matrix(
            delta_sources=delta_sources,
            delta_moment=delta_moment,
            shrink_B=shrink_B,
            source_normalization=source_normalization,
            normalize_source_weights=normalize_source_weights,
        )
        B = self._normalize_joint_delta_matrix(
            matrix=B,
            reference=A,
            normalization=joint_normalization,
            eps=1e-12,
        )

        eye = torch.eye(self.x_dim, device=A.device, dtype=A.dtype)
        A_reg = A + ridge * eye
        C = self._symmetrize(A_reg + float(lam) * B)
        P_full = torch.linalg.solve(C.mH, A_reg.mH).mH.to(dtype=A.dtype)
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
                joint_normalization=joint_normalization,
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
            joint_normalization=joint_normalization,
        )
