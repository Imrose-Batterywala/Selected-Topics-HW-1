from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sort_class_name(name: str) -> tuple[int, object]:
    return (0, int(name)) if name.isdigit() else (1, name)


def discover_class_names(root: Path) -> list[str]:
    classes = sorted(
        [entry.name for entry in os.scandir(root) if entry.is_dir()],
        key=sort_class_name,
    )
    if not classes:
        raise FileNotFoundError(f"No class directories found under {root}")
    return classes


def compute_class_counts(targets: list[int], num_classes: int) -> torch.Tensor:
    return torch.bincount(torch.tensor(targets, dtype=torch.long), minlength=num_classes)


def build_weighted_sampler(dataset: ImageFolder, power: float = 0.5) -> WeightedRandomSampler:
    class_counts = compute_class_counts(dataset.targets, len(dataset.classes)).to(torch.float32)
    class_weights = class_counts.clamp(min=1.0).pow(-power)
    sample_weights = class_weights[torch.tensor(dataset.targets, dtype=torch.long)]
    return WeightedRandomSampler(
        weights=sample_weights.to(torch.double),
        num_samples=len(sample_weights),
        replacement=True,
    )


class NumericImageFolder(ImageFolder):
    def find_classes(self, directory: str) -> tuple[list[str], dict[str, int]]:
        classes = discover_class_names(Path(directory))
        class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
        return classes, class_to_idx


class FlatImageDataset(Dataset):
    def __init__(self, root: Path, transform: Callable | None = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.image_paths = sorted(
            [
                path
                for path in self.root.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            ]
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No image files found under {self.root}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
        return image, image_path.name


def build_dataloaders(
    data_dir: Path,
    train_transform: Callable,
    eval_transform: Callable,
    batch_size: int,
    num_workers: int,
    drop_last_train: bool = False,
    train_sampler: str = "weighted",
    sampler_power: float = 0.5,
) -> tuple[DataLoader, DataLoader, list[str]]:
    train_dataset = NumericImageFolder(str(data_dir / "train"), transform=train_transform)
    val_dataset = NumericImageFolder(str(data_dir / "val"), transform=eval_transform)

    if train_dataset.classes != val_dataset.classes:
        raise RuntimeError("Train and validation class folders do not match.")

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }
    if train_sampler == "weighted":
        sampler = build_weighted_sampler(train_dataset, power=sampler_power)
        train_loader = DataLoader(
            train_dataset,
            sampler=sampler,
            shuffle=False,
            drop_last=drop_last_train,
            **loader_kwargs,
        )
    elif train_sampler == "shuffle":
        train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=drop_last_train,
            **loader_kwargs,
        )
    else:
        raise ValueError(f"Unsupported train sampler: {train_sampler}")
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_dataset.classes


def build_test_loader(
    test_dir: Path,
    transform: Callable,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    dataset = FlatImageDataset(test_dir, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
