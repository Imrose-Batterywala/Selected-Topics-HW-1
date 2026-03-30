#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw1_classifier.data import NumericImageFolder
from hw1_classifier.modeling import create_classifier, create_transforms
from hw1_classifier.utils import (
    balanced_accuracy_from_confusion_matrix,
    macro_f1_from_confusion_matrix,
    update_confusion_matrix,
)


BACKGROUND = "#f7f5ef"
PANEL_BG = "#fffdf9"
BORDER = "#d7d1c4"
TEXT = "#1f1f1f"
SUBTEXT = "#6b665e"
LOW_COLOR = (247, 245, 239)
HIGH_COLOR = (31, 122, 140)
DIAGONAL_OUTLINE = "#a32020"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a labeled split and create confusion-matrix artifacts."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Checkpoint path, for example artifacts/model1/best_model.pt",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for outputs. Defaults to the checkpoint directory.",
    )
    parser.add_argument(
        "--top-errors",
        type=int,
        default=25,
        help="How many off-diagonal confusions to include in the summary CSV.",
    )
    return parser.parse_args()


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def resolve_outputs(output_dir: Path, split: str) -> dict[str, Path]:
    prefix = f"confusion_matrix_{split}"
    return {
        "raw_csv": output_dir / f"{prefix}.csv",
        "normalized_csv": output_dir / f"{prefix}_normalized.csv",
        "heatmap_png": output_dir / f"{prefix}.png",
        "top_errors_csv": output_dir / f"{prefix}_top_errors.csv",
        "summary_txt": output_dir / f"{prefix}_summary.txt",
    }


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    class_names = list(checkpoint["class_names"])
    model = create_classifier(
        num_classes=len(class_names),
        backbone=checkpoint["backbone"],
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    _, eval_transform, _ = create_transforms(model)
    return model, class_names, checkpoint["backbone"]


def build_target_index(dataset_classes: Sequence[str], class_names: Sequence[str]) -> torch.Tensor:
    if list(dataset_classes) == list(class_names):
        return torch.arange(len(class_names), dtype=torch.long)
    if len(dataset_classes) != len(class_names) or set(dataset_classes) != set(class_names):
        raise ValueError("Dataset classes do not match the checkpoint label set.")
    class_to_idx = {class_name: idx for idx, class_name in enumerate(class_names)}
    return torch.tensor([class_to_idx[class_name] for class_name in dataset_classes], dtype=torch.long)


@torch.no_grad()
def evaluate_split(
    model: torch.nn.Module,
    dataset_root: Path,
    class_names: list[str],
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    dataset = NumericImageFolder(str(dataset_root), transform=create_transforms(model)[1])
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )
    target_index = build_target_index(dataset.classes, class_names)
    target_index_device = target_index.to(device=device)
    confusion = torch.zeros((len(class_names), len(class_names)), dtype=torch.int64)
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets = target_index_device[targets]
        logits = model(inputs)
        predictions = logits.argmax(dim=1)

        total_correct += (predictions == targets).sum().item()
        total_examples += targets.size(0)
        update_confusion_matrix(confusion, predictions, targets)

    if total_examples == 0:
        raise RuntimeError(f"No samples found in {dataset_root}")
    return confusion, total_correct / total_examples


def write_confusion_csv(path: Path, class_names: Sequence[str], matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *row])


def row_normalize(confusion: torch.Tensor) -> torch.Tensor:
    confusion = confusion.to(torch.float64)
    support = confusion.sum(dim=1, keepdim=True)
    return confusion / support.clamp(min=1.0)


def write_normalized_csv(path: Path, class_names: Sequence[str], matrix: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *class_names])
        for class_name, row in zip(class_names, matrix.tolist()):
            writer.writerow([class_name, *[f"{value:.6f}" for value in row]])


def blend_color(low: tuple[int, int, int], high: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    clamped = max(0.0, min(1.0, t))
    return tuple(
        int(round(low_value + (high_value - low_value) * clamped))
        for low_value, high_value in zip(low, high)
    )


def draw_heatmap(
    normalized: torch.Tensor,
    class_names: Sequence[str],
    summary_lines: Sequence[str],
    output_path: Path,
    split: str,
) -> None:
    count = len(class_names)
    cell_size = max(10, min(18, 1500 // max(count, 1)))
    matrix_size = cell_size * count
    left_margin = 140
    top_margin = 180
    right_margin = 80
    bottom_margin = 120
    width = left_margin + matrix_size + right_margin
    height = top_margin + matrix_size + bottom_margin

    image = Image.new("RGB", (width, height), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(28)
    subtitle_font = load_font(18)
    label_font = load_font(14)
    tiny_font = load_font(11)

    panel_box = (24, 24, width - 24, height - 24)
    draw.rounded_rectangle(panel_box, radius=18, fill=PANEL_BG, outline=BORDER, width=2)
    draw.text((48, 42), f"Confusion Matrix ({split})", fill=TEXT, font=title_font)
    subtitle_y = 82
    for line in summary_lines:
        draw.text((50, subtitle_y), line, fill=SUBTEXT, font=subtitle_font)
        subtitle_y += 26

    matrix_left = left_margin
    matrix_top = top_margin
    matrix_right = matrix_left + matrix_size
    matrix_bottom = matrix_top + matrix_size

    for row_index in range(count):
        for col_index in range(count):
            value = float(normalized[row_index, col_index])
            color = blend_color(LOW_COLOR, HIGH_COLOR, math.sqrt(value))
            x0 = matrix_left + col_index * cell_size
            y0 = matrix_top + row_index * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size
            draw.rectangle((x0, y0, x1, y1), fill=color)
            if row_index == col_index and value > 0:
                draw.rectangle((x0, y0, x1, y1), outline=DIAGONAL_OUTLINE, width=1)

    tick_step = max(1, math.ceil(count / 20))
    for index, class_name in enumerate(class_names):
        if index % tick_step != 0 and index != count - 1:
            continue
        x = matrix_left + index * cell_size + cell_size / 2
        y = matrix_top + index * cell_size + cell_size / 2

        x_label_width, x_label_height = text_size(draw, class_name, tiny_font)
        draw.text((x - x_label_width / 2, matrix_top - x_label_height - 10), class_name, fill=SUBTEXT, font=tiny_font)

        y_label_width, y_label_height = text_size(draw, class_name, tiny_font)
        draw.text((matrix_left - y_label_width - 10, y - y_label_height / 2), class_name, fill=SUBTEXT, font=tiny_font)

        draw.line((x, matrix_top - 4, x, matrix_bottom), fill=BORDER, width=1)
        draw.line((matrix_left - 4, y, matrix_right, y), fill=BORDER, width=1)

    draw.rectangle((matrix_left, matrix_top, matrix_right, matrix_bottom), outline=SUBTEXT, width=2)

    x_axis = "Predicted label"
    x_axis_width, _ = text_size(draw, x_axis, label_font)
    draw.text((matrix_left + (matrix_size - x_axis_width) / 2, height - 60), x_axis, fill=TEXT, font=label_font)
    draw.text((50, matrix_top - 30), "True label", fill=TEXT, font=label_font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def collect_top_errors(
    confusion: torch.Tensor,
    class_names: Sequence[str],
    limit: int,
) -> list[dict[str, str | int | float]]:
    support = confusion.sum(dim=1)
    rows: list[dict[str, str | int | float]] = []
    for true_index in range(confusion.size(0)):
        for pred_index in range(confusion.size(1)):
            if true_index == pred_index:
                continue
            count = int(confusion[true_index, pred_index].item())
            if count == 0:
                continue
            class_support = int(support[true_index].item())
            rows.append(
                {
                    "true_label": class_names[true_index],
                    "pred_label": class_names[pred_index],
                    "count": count,
                    "support": class_support,
                    "row_rate": 0.0 if class_support == 0 else count / class_support,
                }
            )

    rows.sort(
        key=lambda row: (
            int(row["count"]),
            float(row["row_rate"]),
            str(row["true_label"]),
            str(row["pred_label"]),
        ),
        reverse=True,
    )
    return rows[:limit]


def write_top_errors_csv(path: Path, rows: Sequence[dict[str, str | int | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true_label", "pred_label", "count", "support", "row_rate"])
        for row in rows:
            writer.writerow(
                [
                    row["true_label"],
                    row["pred_label"],
                    row["count"],
                    row["support"],
                    f"{float(row['row_rate']):.6f}",
                ]
            )


def write_summary(
    path: Path,
    checkpoint_path: Path,
    split: str,
    backbone: str,
    accuracy: float,
    balanced_accuracy: float,
    macro_f1: float,
    top_errors: Sequence[dict[str, str | int | float]],
) -> list[str]:
    lines = [
        f"checkpoint: {checkpoint_path}",
        f"split: {split}",
        f"backbone: {backbone}",
        f"accuracy: {accuracy:.4f}",
        f"balanced_accuracy: {balanced_accuracy:.4f}",
        f"macro_f1: {macro_f1:.4f}",
        "",
        "top confusions:",
    ]
    if top_errors:
        for row in top_errors:
            lines.append(
                f"{row['true_label']} -> {row['pred_label']}: "
                f"{row['count']} of {row['support']} ({float(row['row_rate']) * 100:.2f}%)"
            )
    else:
        lines.append("none")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return [
        f"Accuracy: {accuracy * 100:.2f}%",
        f"Balanced accuracy: {balanced_accuracy * 100:.2f}%    Macro F1: {macro_f1 * 100:.2f}%",
    ]


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = (args.output_dir or checkpoint_path.parent).resolve()
    dataset_root = (args.data_dir / args.split).resolve()
    outputs = resolve_outputs(output_dir, args.split)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, class_names, backbone = load_model(checkpoint_path, device)
    confusion, accuracy = evaluate_split(
        model=model,
        dataset_root=dataset_root,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )
    normalized = row_normalize(confusion)
    balanced_accuracy = balanced_accuracy_from_confusion_matrix(confusion)
    macro_f1 = macro_f1_from_confusion_matrix(confusion)
    top_errors = collect_top_errors(confusion, class_names, args.top_errors)

    write_confusion_csv(outputs["raw_csv"], class_names, confusion)
    write_normalized_csv(outputs["normalized_csv"], class_names, normalized)
    write_top_errors_csv(outputs["top_errors_csv"], top_errors)
    summary_lines = write_summary(
        path=outputs["summary_txt"],
        checkpoint_path=checkpoint_path,
        split=args.split,
        backbone=backbone,
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        top_errors=top_errors,
    )
    draw_heatmap(
        normalized=normalized,
        class_names=class_names,
        summary_lines=summary_lines,
        output_path=outputs["heatmap_png"],
        split=args.split,
    )

    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {dataset_root}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Balanced accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1 * 100:.2f}%")
    print(f"Saved raw matrix to {outputs['raw_csv']}")
    print(f"Saved normalized matrix to {outputs['normalized_csv']}")
    print(f"Saved heatmap to {outputs['heatmap_png']}")
    print(f"Saved top-error summary to {outputs['top_errors_csv']}")
    print(f"Saved text summary to {outputs['summary_txt']}")


if __name__ == "__main__":
    main()
