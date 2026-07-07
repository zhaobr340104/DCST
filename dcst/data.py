"""Indian Pines data preparation for the DCST release."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
import torch
from torch.utils.data import DataLoader, Dataset


class DualPathPatchDataset(Dataset):
    def __init__(
        self,
        pca_image: np.ndarray,
        raw_image: np.ndarray,
        labels: np.ndarray,
        positions: np.ndarray,
        patch_size: int,
    ) -> None:
        self.pca_image = pca_image
        self.raw_image = raw_image
        self.labels = labels
        self.positions = positions.astype(np.int64, copy=False)
        self.patch_size = int(patch_size)

    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def __getitem__(
        self,
        index: int,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        row, column = self.positions[index]
        size = self.patch_size
        pca_patch = self.pca_image[
            row : row + size,
            column : column + size,
        ].transpose(2, 0, 1)
        raw_patch = self.raw_image[
            row : row + size,
            column : column + size,
        ].transpose(2, 0, 1)
        label = int(self.labels[row, column]) - 1
        return (
            torch.from_numpy(pca_patch.copy()).float(),
            torch.from_numpy(raw_patch.copy()).float(),
        ), torch.tensor(label, dtype=torch.long)


@dataclass
class PreparedData:
    pca_image: np.ndarray
    raw_image: np.ndarray
    labels: np.ndarray
    train_positions: np.ndarray
    test_positions: np.ndarray
    patch_size: int
    audit: dict


def _minmax_normalize(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=np.float64)
    minimum = image.min(axis=(0, 1), keepdims=True)
    maximum = image.max(axis=(0, 1), keepdims=True)
    denominator = maximum - minimum
    denominator[denominator == 0] = 1.0
    return (image - minimum) / denominator


def _pad(image: np.ndarray, margin: int) -> np.ndarray:
    return np.pad(
        image,
        ((margin, margin), (margin, margin), (0, 0)),
        mode="constant",
    )


def prepare_indian_pines(
    mat_path: str | Path,
    patch_size: int,
    pca_components: int,
    pca_whiten: bool,
) -> PreparedData:
    mat_path = Path(mat_path)
    arrays = loadmat(str(mat_path), variable_names=["input", "TR", "TE"])
    missing = sorted({"input", "TR", "TE"} - set(arrays))
    if missing:
        raise ValueError(f"Dataset file is missing fields: {missing}")

    raw = np.asarray(arrays["input"])
    train_map = np.asarray(arrays["TR"])
    test_map = np.asarray(arrays["TE"])
    if raw.ndim != 3 or train_map.shape != raw.shape[:2]:
        raise ValueError("Unexpected image or label shape")
    if np.any((train_map > 0) & (test_map > 0)):
        raise ValueError("Training and test masks overlap")
    if patch_size % 2 != 1:
        raise ValueError("Patch size must be odd")

    labels = train_map + test_map
    classes = int(labels.max())
    train_counts = [
        int(np.count_nonzero(train_map == class_id))
        for class_id in range(1, classes + 1)
    ]
    if train_counts != [10] * classes:
        raise ValueError(
            "The release expects exactly ten training samples per class"
        )

    normalized_raw = _minmax_normalize(raw)
    flattened = normalized_raw.reshape(-1, normalized_raw.shape[-1])
    pca = PCA(
        n_components=int(pca_components),
        whiten=bool(pca_whiten),
    )
    pca_image = pca.fit_transform(flattened).reshape(
        raw.shape[0],
        raw.shape[1],
        int(pca_components),
    )
    centered_raw = normalized_raw - pca.mean_[None, None, :]

    margin = patch_size // 2
    train_positions = np.argwhere(train_map > 0)
    test_positions = np.argwhere(test_map > 0)
    audit = {
        "input_shape": list(raw.shape),
        "num_classes": classes,
        "train_samples": int(train_positions.shape[0]),
        "test_samples": int(test_positions.shape[0]),
        "train_counts": train_counts,
        "pca_components": int(pca_components),
        "pca_whiten": bool(pca_whiten),
        "pca_retained_variance": float(
            pca.explained_variance_ratio_.sum()
        ),
    }
    return PreparedData(
        pca_image=_pad(pca_image, margin),
        raw_image=_pad(centered_raw, margin),
        labels=labels,
        train_positions=train_positions,
        test_positions=test_positions,
        patch_size=int(patch_size),
        audit=audit,
    )


def build_loaders(
    prepared: PreparedData,
    train_batch_size: int,
    eval_batch_size: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    train_set = DualPathPatchDataset(
        prepared.pca_image,
        prepared.raw_image,
        prepared.labels,
        prepared.train_positions,
        prepared.patch_size,
    )
    test_set = DualPathPatchDataset(
        prepared.pca_image,
        prepared.raw_image,
        prepared.labels,
        prepared.test_positions,
        prepared.patch_size,
    )
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_batch_size),
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=int(eval_batch_size),
        shuffle=False,
        num_workers=0,
    )
    return train_loader, test_loader
