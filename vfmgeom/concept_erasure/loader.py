"""Runtime loader for saved linear eraser artifacts.

This module is intentionally small and benchmark-friendly. It does not import
the fitting code, SCORPION utilities, stain-cache code, or experiment runners.

It supports two artifact formats:

1. Single eraser NPZ
   - explicit map: ``P``
   - or low-rank residual map: ``proj_left`` and ``proj_right``
   - optional affine center: ``bias``

2. Chained eraser NPZ
   - ``metadata_json`` with ``{"type": "chained_linear_eraser", "n_components": ...}``
   - component arrays such as ``component_0_P``, ``component_1_proj_left``, ...
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor


def _json_from_npz_key(data: np.lib.npyio.NpzFile, key: str) -> dict[str, Any]:
    if key not in data.files:
        return {}

    raw: Any = data[key]
    if isinstance(raw, np.ndarray):
        raw = raw.item()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")

    text = str(raw)
    if not text:
        return {}

    return json.loads(text)


def _npz_has_chained_keys(data: np.lib.npyio.NpzFile) -> bool:
    return any(key.startswith("component_0_") for key in data.files)


def _infer_n_components(data: np.lib.npyio.NpzFile) -> int:
    indices: set[int] = set()

    for key in data.files:
        if not key.startswith("component_"):
            continue
        parts = key.split("_", 2)
        if len(parts) < 3:
            continue
        try:
            indices.add(int(parts[1]))
        except ValueError:
            continue

    if not indices:
        return 0

    expected = set(range(max(indices) + 1))
    if indices != expected:
        raise ValueError(
            "Chained eraser component indices must be contiguous from 0. "
            f"Found {sorted(indices)}."
        )

    return max(indices) + 1


@dataclass(frozen=True)
class SavedLinearEraser:
    """Affine linear eraser reconstructed from a saved NPZ artifact.

    The map is either stored explicitly as ``P`` or implicitly as a low-rank
    residual projection:

        P = I - proj_left @ proj_right

    For feature vectors, the optional bias is used affinely:

        x' = bias + P @ (x - bias)

    For deltas, the bias is ignored:

        delta' = P @ delta
    """

    P: Tensor | None
    proj_left: Tensor | None
    proj_right: Tensor | None
    bias: Tensor | None
    metadata: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SavedLinearEraser":
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Saved eraser not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            eraser = cls.from_npz(
                data,
                prefix="",
                map_location=map_location,
                dtype=dtype,
                metadata_key="metadata_json",
            )
            available = sorted(data.files)

        eraser.validate(available_keys=available)
        return eraser

    @classmethod
    def from_npz(
        cls,
        data: np.lib.npyio.NpzFile,
        *,
        prefix: str = "",
        map_location: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        metadata_key: str | None = None,
    ) -> "SavedLinearEraser":
        metadata = _json_from_npz_key(data, metadata_key) if metadata_key else {}

        return cls(
            P=cls._load_tensor(data, f"{prefix}P", map_location, dtype),
            proj_left=cls._load_tensor(
                data,
                f"{prefix}proj_left",
                map_location,
                dtype,
            ),
            proj_right=cls._load_tensor(
                data,
                f"{prefix}proj_right",
                map_location,
                dtype,
            ),
            bias=cls._load_tensor(data, f"{prefix}bias", map_location, dtype),
            metadata=metadata,
        )

    @staticmethod
    def _load_tensor(
        data: np.lib.npyio.NpzFile,
        name: str,
        map_location: torch.device | str | None,
        dtype: torch.dtype,
    ) -> Tensor | None:
        if name not in data.files:
            return None

        array = np.asarray(data[name])
        tensor = torch.from_numpy(array).to(dtype=dtype)

        if map_location is not None:
            tensor = tensor.to(map_location)

        return tensor

    @staticmethod
    def _load_metadata(data: np.lib.npyio.NpzFile) -> dict[str, Any]:
        return _json_from_npz_key(data, "metadata_json")

    def validate(self, *, available_keys: list[str] | None = None) -> None:
        if self.P is None and (self.proj_left is None or self.proj_right is None):
            detail = "" if available_keys is None else f" Available keys: {available_keys}"
            raise KeyError(
                "Invalid eraser artifact. Expected either 'P', or both "
                f"'proj_left' and 'proj_right'.{detail}"
            )

        if self.P is not None:
            if self.P.ndim != 2 or self.P.shape[0] != self.P.shape[1]:
                raise ValueError(f"Expected square P, got shape {tuple(self.P.shape)}.")

        if self.proj_left is not None or self.proj_right is not None:
            if self.proj_left is None or self.proj_right is None:
                raise ValueError(
                    "Low-rank eraser artifacts must contain both proj_left and "
                    "proj_right."
                )

            if self.proj_left.ndim != 2 or self.proj_right.ndim != 2:
                raise ValueError("proj_left and proj_right must be matrices.")

            if self.proj_left.shape[1] != self.proj_right.shape[0]:
                raise ValueError(
                    "Low-rank factor mismatch: "
                    f"{tuple(self.proj_left.shape)} vs "
                    f"{tuple(self.proj_right.shape)}."
                )

            if self.proj_left.shape[0] != self.proj_right.shape[1]:
                raise ValueError("Low-rank factors do not define a square map.")

        if self.P is not None and self.proj_left is not None:
            if self.P.shape[0] != self.proj_left.shape[0]:
                raise ValueError(
                    "Explicit and low-rank maps disagree on input dimension: "
                    f"{tuple(self.P.shape)} vs {tuple(self.proj_left.shape)}."
                )

        if self.bias is not None:
            if self.bias.ndim != 1:
                raise ValueError(f"Expected one-dimensional bias, got {self.bias.shape}.")
            if self.bias.shape[0] != self.input_dim:
                raise ValueError(
                    f"Bias dimension {self.bias.shape[0]} does not match "
                    f"input dimension {self.input_dim}."
                )

    @property
    def input_dim(self) -> int:
        if self.P is not None:
            return int(self.P.shape[0])

        assert self.proj_left is not None
        return int(self.proj_left.shape[0])

    @property
    def reference_tensor(self) -> Tensor:
        if self.P is not None:
            return self.P

        assert self.proj_left is not None
        return self.proj_left

    def apply_linear(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected last dimension {self.input_dim}, got {x.shape[-1]}."
            )

        if self.P is not None:
            return x @ self.P.T

        if self.proj_left is None or self.proj_right is None:
            raise RuntimeError("Missing low-rank eraser factors.")

        correction = (x @ self.proj_right.T) @ self.proj_left.T
        return x - correction

    def transform_delta(self, delta: Tensor) -> Tensor:
        return self.apply_linear(delta)

    def __call__(self, x: Tensor) -> Tensor:
        centered = x - self.bias if self.bias is not None else x
        output = self.apply_linear(centered)
        return output + self.bias if self.bias is not None else output

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SavedLinearEraser":
        def move(value: Tensor | None) -> Tensor | None:
            if value is None:
                return None
            return value.to(device=device, dtype=dtype)

        return SavedLinearEraser(
            P=move(self.P),
            proj_left=move(self.proj_left),
            proj_right=move(self.proj_right),
            bias=move(self.bias),
            metadata=self.metadata,
        )


@dataclass(frozen=True)
class SavedChainedEraser:
    """Sequential composition of saved linear erasers."""

    components: tuple[SavedLinearEraser, ...]
    metadata: dict[str, Any]

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "SavedChainedEraser":
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Chained eraser not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            metadata = _json_from_npz_key(data, "metadata_json")

            if "n_components" in metadata:
                n_components = int(metadata["n_components"])
            else:
                n_components = _infer_n_components(data)

            if n_components <= 0:
                raise ValueError(
                    "Chained eraser contains no components. Expected "
                    "metadata_json['n_components'] or component_0_* arrays."
                )

            components: list[SavedLinearEraser] = []
            available = sorted(data.files)

            for i in range(n_components):
                component = SavedLinearEraser.from_npz(
                    data,
                    prefix=f"component_{i}_",
                    map_location=map_location,
                    dtype=dtype,
                    metadata_key=f"component_{i}_metadata_json",
                )

                try:
                    component.validate(available_keys=available)
                except Exception as error:
                    raise type(error)(
                        f"Invalid chained eraser component {i}: {error}"
                    ) from error

                components.append(component)

        input_dim = components[0].input_dim
        for i, component in enumerate(components[1:], start=1):
            if component.input_dim != input_dim:
                raise ValueError(
                    "All chained eraser components must have the same input "
                    f"dimension. Component 0 has {input_dim}; component {i} has "
                    f"{component.input_dim}."
                )

        return cls(
            components=tuple(components),
            metadata=metadata,
        )

    @property
    def input_dim(self) -> int:
        return self.components[0].input_dim

    @property
    def reference_tensor(self) -> Tensor:
        return self.components[0].reference_tensor

    def __call__(self, x: Tensor) -> Tensor:
        for component in self.components:
            x = component(x)
        return x

    def transform_delta(self, delta: Tensor) -> Tensor:
        for component in self.components:
            delta = component.transform_delta(delta)
        return delta

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "SavedChainedEraser":
        return SavedChainedEraser(
            components=tuple(
                component.to(device=device, dtype=dtype)
                for component in self.components
            ),
            metadata=self.metadata,
        )


SavedEraserArtifact = SavedLinearEraser | SavedChainedEraser


class SavedEraser:
    """Factory for loading single or chained saved eraser artifacts."""

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        map_location: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SavedEraserArtifact:
        path = Path(path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Saved eraser not found: {path}")

        with np.load(path, allow_pickle=False) as data:
            metadata = _json_from_npz_key(data, "metadata_json")
            is_chain = (
                metadata.get("type") == "chained_linear_eraser"
                or _npz_has_chained_keys(data)
            )

        if is_chain:
            return SavedChainedEraser.load(
                path,
                map_location=map_location,
                dtype=dtype,
            )

        return SavedLinearEraser.load(
            path,
            map_location=map_location,
            dtype=dtype,
        )


__all__ = [
    "SavedChainedEraser",
    "SavedEraser",
    "SavedEraserArtifact",
    "SavedLinearEraser",
]