from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import timm
import torch
from torch import nn

# adapt this import to your actual package
from vfmgeom.concept_erasure.leace import LeaceEraser


class H0MiniLeaceCLS(nn.Module):
    """
    H0-mini + LEACE projection wrapper for EVA offline classification.

    EVA expects:
        forward(x) -> Tensor[B, D]

    This wrapper returns CLS embeddings after LEACE.
    """

    def __init__(
        self,
        leace_path: str | Path,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.output_dim = 768

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=pretrained,
            **timm_kwargs,
        )

        # Load on CPU. EVA / Lightning will move the full module to the right device.
        self.leace_eraser = LeaceEraser.load(
            leace_path,
            map_location="cpu",
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(images)

        # H0-mini through timm appears to return token embeddings [B, N, D]
        cls_features = tokens[:, 0]

        cls_features = self.leace_eraser(cls_features)

        return cls_features


class H0MiniCLS(nn.Module):
    """
    H0-mini

    EVA expects:
        forward(x) -> Tensor[B, D]

    This wrapper returns CLS embeddings after LEACE.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.output_dim = 768

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=pretrained,
            **timm_kwargs,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(images)

        # H0-mini through timm appears to return token embeddings [B, N, D]
        cls_features = tokens[:, 0]

        return cls_features


class H0MiniGAP(nn.Module):
    """
    H0-mini

    EVA expects:
        forward(x) -> Tensor[B, D]

    This wrapper returns CLS embeddings after LEACE.
    """

    def __init__(
        self,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.output_dim = 768

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=pretrained,
            **timm_kwargs,
        )
        self.num_prefix_tokens = self.backbone.num_prefix_tokens

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(images)

        # H0-mini through timm appears to return token embeddings [B, N, D]
        patch_features = tokens[:, self.num_prefix_tokens :]
        features = patch_features.mean(dim=1)

        return features



class H0MiniPCAProjCLS(nn.Module):
    """
    H0-mini + PCA scanner-delta projection wrapper for EVA offline classification.

    EVA expects:
        forward(x) -> Tensor[B, D_out]

    Two modes are supported:

    1. same_dim=True:
        Project away the first `rank` PCA components and keep the original
        embedding dimension.

    2. same_dim=False:
        Rotate into the PCA basis and keep only the remainder coordinates,
        reducing dimensionality from D to D - rank.
    """

    DEFAULT_PCA_PATH = (
        "/home/valentin/workspaces/vfm-geom-xai/"
        "output/scorpion_analysis/paired_scanner_delta_pca_group_mean/"
        "h0-mini_scorpion_224px_0p5mpp_cls/"
        "fold_projectors/paired_scanner_delta_pca_fold0.npz"
    )

    def __init__(
        self,
        pca_path: str | Path | None = None,
        rank: int = 16,
        same_dim: bool = True,
        normalize_output: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        self.embedding_dim = 768
        self.rank = rank
        self.same_dim = same_dim
        self.normalize_output = normalize_output

        self.output_dim = self.embedding_dim if same_dim else self.embedding_dim - rank

        if self.output_dim <= 0:
            raise ValueError(
                f"Invalid output_dim={self.output_dim}. "
                f"rank={rank} must be smaller than embedding_dim={self.embedding_dim}."
            )

        timm_kwargs: dict[str, Any] = {
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
        }

        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=pretrained,
            **timm_kwargs,
        )

        # Load on CPU. EVA / Lightning will move the full module to the right device.
        components = self._load_pca_components(pca_path)

        # Register as buffer so it follows .to(device), .cuda(), checkpointing, etc.
        self.register_buffer(
            "pca_components",
            torch.from_numpy(components),
            persistent=True,
        )

    def _resolve_pca_path(self, pca_path: str | Path | None) -> Path:
        if pca_path is not None:
            return Path(pca_path)

        return Path(self.DEFAULT_PCA_PATH)

    def _load_pca_components(self, pca_path: str | Path | None) -> np.ndarray:
        pca_path = self._resolve_pca_path(pca_path)

        if not pca_path.exists():
            raise FileNotFoundError(f"PCA projector not found: {pca_path}")

        data = np.load(pca_path, allow_pickle=True)

        if "components" not in data:
            raise KeyError(f"Missing 'components' in PCA projector: {pca_path}")

        components = data["components"].astype(np.float32)

        if components.ndim != 2:
            raise ValueError(
                "Expected PCA components with shape "
                f"(n_components, dim), got {components.shape}."
            )

        if components.shape[1] != self.embedding_dim:
            raise ValueError(
                "PCA component dimension mismatch. "
                f"Expected {self.embedding_dim}, got {components.shape[1]}."
            )

        if self.rank > components.shape[0]:
            raise ValueError(
                f"Requested rank={self.rank}, but projector only contains "
                f"{components.shape[0]} components."
            )

        if not self.same_dim and components.shape[0] < self.embedding_dim:
            raise ValueError(
                "same_dim=False requires a full PCA basis with D components. "
                f"Got only {components.shape[0]} components for D={self.embedding_dim}."
            )

        return components

    def _apply_pca_projection(self, features: torch.Tensor) -> torch.Tensor:
        components = self.pca_components.to(
            device=features.device,
            dtype=features.dtype,
        )

        if self.same_dim:
            # Remove first `rank` scanner-delta directions.
            C = components[: self.rank]  # [rank, D]
            return features - (features @ C.T) @ C

        # Rotate into PCA basis and keep the orthogonal complement coordinates.
        C_rem = components[self.rank :]  # [D - rank, D]
        return features @ C_rem.T

    def _normalize(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return features / (features.norm(dim=1, keepdim=True) + eps)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(images)

        # H0-mini through timm appears to return token embeddings [B, N, D].
        cls_features = tokens[:, 0]

        cls_features = self._apply_pca_projection(cls_features)

        if self.normalize_output:
            cls_features = self._normalize(cls_features)

        return cls_features