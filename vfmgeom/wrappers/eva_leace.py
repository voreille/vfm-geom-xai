# vfmgeom/eva_wrappers/h0mini_leace.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import timm

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


class H0MiniWithPCAProj(nn.Module):
    """H0-mini with a PCA scanner-delta projection applied to CLS-token features.

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
        normalize_output: bool = True,
    ):
        super().__init__()

        self.embedding_dim = 768
        self.rank = rank
        self.same_dim = same_dim
        self.normalize_output = normalize_output
        self.mixed_precision = mixed_precision

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

        feature_extractor = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            **timm_kwargs,
        )

        self.feature_extractor, self.device = prepare_module(
            feature_extractor,
            device,
            self.mixed_precision,
        )

        if self.device is None:
            self.feature_extractor = self.feature_extractor.module

        self._init_pca_projection(pca_path)

    def _resolve_pca_path(self, pca_path: str | Path | None) -> Path:
        if pca_path is not None:
            return Path(pca_path)

        env_path = os.environ.get("H0MINI_PCA_PROJECTOR_PATH")
        if env_path is not None:
            return Path(env_path)

        return Path(self.DEFAULT_PCA_PATH)

    def _init_pca_projection(self, pca_path: str | Path | None) -> None:
        pca_path = self._resolve_pca_path(pca_path)

        if not pca_path.exists():
            raise FileNotFoundError(f"PCA projector not found: {pca_path}")

        data = np.load(pca_path, allow_pickle=True)

        if "components" not in data:
            raise KeyError(f"Missing 'components' in PCA projector: {pca_path}")

        components = data["components"].astype(np.float32)

        if components.ndim != 2:
            raise ValueError(
                f"Expected PCA components with shape (n_components, dim), "
                f"got {components.shape}."
            )

        if components.shape[1] != self.embedding_dim:
            raise ValueError(
                f"PCA component dimension mismatch. "
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

        self.pca_path = pca_path

        self.pca_components = torch.from_numpy(components)

    @property  # type: ignore
    def transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )

    def _apply_pca_projection(self, features: torch.Tensor) -> torch.Tensor:
        components = self.pca_components.to(
            device=features.device,
            dtype=features.dtype,
        )

        if self.same_dim:
            # Remove first `rank` scanner-delta directions.
            C = components[: self.rank]  # (rank, dim)
            return features - (features @ C.T) @ C

        # Rotate into PCA basis and keep the orthogonal complement coordinates.
        C_rem = components[self.rank :]  # (dim-rank, dim)
        return features @ C_rem.T

    def _normalize(self, features: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return features / (features.norm(dim=1, keepdim=True) + eps)

    def _to_numpy(self, features: torch.Tensor) -> np.ndarray:
        return features.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> np.ndarray:
        images = images.to(self.device)

        last_hidden_state = self.feature_extractor(images)
        features = last_hidden_state[:, 0]  # CLS token

        features = self._apply_pca_projection(features)

        if self.normalize_output:
            features = self._normalize(features)

        return self._to_numpy(features)
