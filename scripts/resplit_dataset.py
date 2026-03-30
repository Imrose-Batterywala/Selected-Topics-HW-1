#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import random
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge the labeled train/val pools and rebuild a stratified 90/10 split. "
            "The flat test directory is left untouched because it is unlabeled."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Dataset root containing train/, val/, and optionally test/.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when shuffling images before splitting.",
    )
    return parser.parse_args()


def numeric_key(value: str) -> tuple[int, object]:
    return (0, int(value)) if value.isdigit() else (1, value)


def collect_class_files(*roots: Path) -> dict[str, list[Path]]:
    classes = sorted(
        {
            child.name
            for root in roots
            if root.exists()
            for child in root.iterdir()
            if child.is_dir()
        },
        key=numeric_key,
    )
    if not classes:
        raise RuntimeError("No labeled class directories were found.")

    class_to_files: dict[str, list[Path]] = {}
    for class_name in classes:
        files: list[Path] = []
        seen_names: set[str] = set()
        for root in roots:
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for image_path in sorted(path for path in class_dir.iterdir() if path.is_file()):
                if image_path.name in seen_names:
                    raise RuntimeError(
                        f"Duplicate filename detected in class {class_name}: {image_path.name}"
                    )
                seen_names.add(image_path.name)
                files.append(image_path)

        if not files:
            raise RuntimeError(f"Class {class_name} does not contain any files.")
        class_to_files[class_name] = files

    return class_to_files


def compute_val_counts(class_to_files: dict[str, list[Path]]) -> tuple[dict[str, int], int]:
    total_images = sum(len(files) for files in class_to_files.values())
    target_train = round(total_images * 0.9)
    target_val = total_images - target_train

    val_counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned_val = 0

    for class_name, files in class_to_files.items():
        class_total = len(files)
        ideal_val = class_total * target_val / total_images
        base_val = math.floor(ideal_val)
        if class_total > 1:
            base_val = min(base_val, class_total - 1)
        else:
            base_val = 0

        val_counts[class_name] = base_val
        assigned_val += base_val
        remainders.append((ideal_val - base_val, class_name))

    remaining = target_val - assigned_val
    if remaining < 0:
        raise RuntimeError("Validation allocation overshot the target.")

    for _, class_name in sorted(remainders, key=lambda item: (-item[0], numeric_key(item[1]))):
        if remaining == 0:
            break
        class_total = len(class_to_files[class_name])
        max_val = class_total - 1 if class_total > 1 else 0
        if val_counts[class_name] >= max_val:
            continue
        val_counts[class_name] += 1
        remaining -= 1

    if remaining != 0:
        raise RuntimeError("Unable to assign the requested number of validation images.")

    return val_counts, target_val


def move_original_split(data_dir: Path) -> tuple[Path, Path, Path]:
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    backup_dir = data_dir / "_original_split_backup"

    if not train_dir.is_dir() or not val_dir.is_dir():
        raise RuntimeError("Expected both data/train and data/val to exist.")
    if backup_dir.exists():
        raise RuntimeError(f"Backup directory already exists: {backup_dir}")

    backup_dir.mkdir()
    archived_train = backup_dir / "train"
    archived_val = backup_dir / "val"
    shutil.move(str(train_dir), str(archived_train))
    shutil.move(str(val_dir), str(archived_val))

    train_dir.mkdir()
    val_dir.mkdir()
    return archived_train, archived_val, backup_dir


def rebuild_split(
    new_train_dir: Path,
    new_val_dir: Path,
    class_to_files: dict[str, list[Path]],
    val_counts: dict[str, int],
    seed: int,
) -> tuple[int, int]:
    rng = random.Random(seed)
    train_total = 0
    val_total = 0

    for class_name in sorted(class_to_files, key=numeric_key):
        files = class_to_files[class_name][:]
        rng.shuffle(files)
        val_count = val_counts[class_name]
        val_files = files[:val_count]
        train_files = files[val_count:]

        train_class_dir = new_train_dir / class_name
        val_class_dir = new_val_dir / class_name
        train_class_dir.mkdir(parents=True, exist_ok=True)
        val_class_dir.mkdir(parents=True, exist_ok=True)

        for image_path in train_files:
            image_path.rename(train_class_dir / image_path.name)
        for image_path in val_files:
            image_path.rename(val_class_dir / image_path.name)

        train_total += len(train_files)
        val_total += len(val_files)

    return train_total, val_total


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    archived_train, archived_val, backup_dir = move_original_split(data_dir)
    class_to_files = collect_class_files(archived_train, archived_val)
    val_counts, target_val = compute_val_counts(class_to_files)
    train_total, val_total = rebuild_split(
        new_train_dir=data_dir / "train",
        new_val_dir=data_dir / "val",
        class_to_files=class_to_files,
        val_counts=val_counts,
        seed=args.seed,
    )

    shutil.rmtree(backup_dir)

    total_images = train_total + val_total
    print(f"Rebuilt labeled split under {data_dir}")
    print(f"Seed: {args.seed}")
    print(f"Total labeled images: {total_images}")
    print(f"Train images: {train_total}")
    print(f"Validation images: {val_total}")
    print(f"Validation target: {target_val}")
    print("Left data/test untouched because it is unlabeled.")


if __name__ == "__main__":
    main()
