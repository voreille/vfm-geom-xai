from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

from vfmgeom.data.utils import resolve_tile_path


class TileDataset(Dataset):
    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        transform: Optional[T.Compose] = None,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image = Image.open(resolve_tile_path(self.tile_dir, row)).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, idx


class RandomAugmentationDeltaDataset(Dataset):
    """Dataset yielding randomly augmented views used to estimate delta directions.

    For ``original_to_augmented`` it returns one augmented image and the row index.
    For ``augmented_to_augmented`` it returns two independently augmented images and
    the row index.
    """

    def __init__(
        self,
        tile_dir: Path,
        metadata: pd.DataFrame,
        n_augmentations_per_image: int,
        augmentation,
        tensor_transform: T.Compose,
        delta_mode: str,
    ) -> None:
        self.tile_dir = tile_dir
        self.metadata = metadata.reset_index(drop=True)
        self.n_augmentations_per_image = n_augmentations_per_image
        self.augmentation = augmentation
        self.tensor_transform = tensor_transform
        self.delta_mode = delta_mode

        if delta_mode not in {"original_to_augmented", "augmented_to_augmented"}:
            raise ValueError(f"Unknown delta_mode: {delta_mode}")

    def __len__(self) -> int:
        return len(self.metadata) * self.n_augmentations_per_image

    def _load_image_array(self, row_idx: int) -> np.ndarray:
        row = self.metadata.iloc[row_idx]
        image = Image.open(resolve_tile_path(self.tile_dir, row)).convert("RGB")
        return np.asarray(image)

    def _augment_to_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        augmented = self.augmentation(image=image_np)["image"]
        image = Image.fromarray(augmented).convert("RGB")
        return self.tensor_transform(image)

    def __getitem__(self, idx: int):
        row_idx = idx // self.n_augmentations_per_image
        image_np = self._load_image_array(row_idx)

        if self.delta_mode == "original_to_augmented":
            x_aug = self._augment_to_tensor(image_np)
            return x_aug, row_idx

        x_aug_a = self._augment_to_tensor(image_np)
        x_aug_b = self._augment_to_tensor(image_np)
        return x_aug_a, x_aug_b, row_idx
