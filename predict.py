#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import torch
from PIL import ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw1_classifier.data import build_test_loader
from hw1_classifier.modeling import create_classifier, create_transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on the unlabeled HW1 test set.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        required=True,
        help="Checkpoint path. Pass multiple times to average model probabilities.",
    )
    parser.add_argument("--test-dir", type=Path, default=Path("data/test"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/predictions.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--checkpoint-weight",
        type=float,
        action="append",
        help="Optional checkpoint weight. Pass once per --checkpoint in the same order.",
    )
    parser.add_argument(
        "--weight-by-val-accuracy",
        action="store_true",
        help="Weight ensemble members by the val_accuracy stored in each checkpoint.",
    )
    parser.add_argument(
        "--tta",
        action="append",
        choices=["hflip", "vflip", "hvflip"],
        default=[],
        help="Optional test-time augmentation. Repeat to use multiple flip views alongside the original image.",
    )
    return parser.parse_args()


def load_ensemble_member(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = checkpoint["class_names"]
    backbone = checkpoint["backbone"]
    model = create_classifier(
        num_classes=len(class_names),
        backbone=backbone,
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()

    _, eval_transform, _ = create_transforms(model)
    return model, class_names, backbone, eval_transform, checkpoint.get("val_accuracy")


def resolve_tta_variants(requested_variants: list[str]) -> list[str]:
    variants = ["none"]
    for variant in requested_variants:
        if variant not in variants:
            variants.append(variant)
    return variants


def apply_tta(image, variant: str):
    if variant == "none":
        return image
    if variant == "hflip":
        return ImageOps.mirror(image)
    if variant == "vflip":
        return ImageOps.flip(image)
    if variant == "hvflip":
        return ImageOps.flip(ImageOps.mirror(image))
    raise ValueError(f"Unsupported TTA variant: {variant}")


class AugmentedTransform:
    def __init__(self, base_transform, variant: str) -> None:
        self.base_transform = base_transform
        self.variant = variant

    def __call__(self, image):
        return self.base_transform(apply_tta(image, self.variant))


def build_class_index(reference_class_names: list[str], class_names: list[str]) -> torch.Tensor | None:
    if reference_class_names == class_names:
        return None
    if len(reference_class_names) != len(class_names) or set(reference_class_names) != set(class_names):
        raise ValueError("All ensemble checkpoints must share the same label set.")

    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    return torch.tensor([class_to_idx[class_name] for class_name in reference_class_names], dtype=torch.long)


@torch.no_grad()
def collect_probabilities(
    model: torch.nn.Module,
    test_dir: Path,
    transform,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    test_loader = build_test_loader(
        test_dir=test_dir,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    file_names: list[str] = []
    probability_batches: list[torch.Tensor] = []
    for inputs, batch_file_names in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        probability_batches.append(torch.softmax(logits, dim=1).cpu())
        file_names.extend(batch_file_names)

    return file_names, torch.cat(probability_batches, dim=0)


@torch.no_grad()
def collect_tta_probabilities(
    model: torch.nn.Module,
    test_dir: Path,
    transform,
    tta_variants: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], torch.Tensor]:
    reference_file_names: list[str] | None = None
    ensembled_probabilities: torch.Tensor | None = None

    for variant in tta_variants:
        augmented_transform = (
            transform if variant == "none" else AugmentedTransform(transform, variant)
        )
        file_names, probabilities = collect_probabilities(
            model=model,
            test_dir=test_dir,
            transform=augmented_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        if reference_file_names is None:
            reference_file_names = file_names
            ensembled_probabilities = probabilities
        else:
            if file_names != reference_file_names:
                raise RuntimeError("TTA views produced different test-file orders.")
            ensembled_probabilities = ensembled_probabilities + probabilities

    if reference_file_names is None or ensembled_probabilities is None:
        raise RuntimeError("No TTA probabilities were collected.")

    return reference_file_names, ensembled_probabilities / len(tta_variants)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.checkpoint_weight is not None and len(args.checkpoint_weight) != len(args.checkpoint):
        raise ValueError("Pass --checkpoint-weight once per --checkpoint.")
    if args.checkpoint_weight is not None and args.weight_by_val_accuracy:
        raise ValueError("Choose either manual --checkpoint-weight values or --weight-by-val-accuracy.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tta_variants = resolve_tta_variants(args.tta)
    reference_class_names: list[str] | None = None
    reference_file_names: list[str] | None = None
    ensemble_probabilities: torch.Tensor | None = None
    total_weight = 0.0

    print(f"Device: {device}")
    print(f"TTA variants: {', '.join(tta_variants)}")

    for checkpoint_index, checkpoint_path in enumerate(args.checkpoint):
        checkpoint_path = checkpoint_path.resolve()
        model, class_names, backbone, eval_transform, val_accuracy = load_ensemble_member(
            checkpoint_path,
            device,
        )
        class_index = (
            None
            if reference_class_names is None
            else build_class_index(reference_class_names, class_names)
        )
        file_names, probabilities = collect_tta_probabilities(
            model=model,
            test_dir=args.test_dir,
            transform=eval_transform,
            tta_variants=tta_variants,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )

        if class_index is not None:
            probabilities = probabilities.index_select(dim=1, index=class_index)

        if args.checkpoint_weight is not None:
            checkpoint_weight = args.checkpoint_weight[checkpoint_index]
        elif args.weight_by_val_accuracy:
            if val_accuracy is None:
                raise KeyError(
                    f"Checkpoint {checkpoint_path} does not store val_accuracy for automatic weighting."
                )
            checkpoint_weight = float(val_accuracy)
        else:
            checkpoint_weight = 1.0

        weighted_probabilities = probabilities * checkpoint_weight
        if reference_class_names is None:
            reference_class_names = list(class_names)
            reference_file_names = file_names
            ensemble_probabilities = weighted_probabilities
        else:
            if file_names != reference_file_names:
                raise RuntimeError("Ensemble members produced different test-file orders.")
            ensemble_probabilities = ensemble_probabilities + weighted_probabilities

        total_weight += checkpoint_weight
        val_accuracy_text = "" if val_accuracy is None else f", val_acc={val_accuracy:.4f}"
        print(
            f"Loaded checkpoint: {checkpoint_path} ({backbone}) "
            f"weight={checkpoint_weight:.4f}{val_accuracy_text}"
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if reference_class_names is None or reference_file_names is None or ensemble_probabilities is None:
        raise RuntimeError("No checkpoints were provided.")
    if total_weight <= 0.0:
        raise RuntimeError("Ensemble weight must be positive.")

    ensemble_probabilities = ensemble_probabilities / total_weight

    predictions = ensemble_probabilities.argmax(dim=1).tolist()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_name", "pred_label"])

        for file_name, prediction in zip(reference_file_names, predictions):
            writer.writerow([Path(file_name).stem, reference_class_names[prediction]])

    print(f"Ensembled {len(args.checkpoint)} checkpoint(s)")
    print(f"Wrote predictions to {args.output}")


if __name__ == "__main__":
    main()
