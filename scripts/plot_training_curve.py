#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


EPOCH_PATTERN = re.compile(
    r"^Epoch\s+(\d+)/(\d+)\s+\|\s+"
    r"train_loss=([0-9]*\.?[0-9]+)\s+train_acc=([0-9]*\.?[0-9]+)\s+\|\s+"
    r"val_loss=([0-9]*\.?[0-9]+)\s+val_acc=([0-9]*\.?[0-9]+)$"
)

IMAGE_WIDTH = 1600
IMAGE_HEIGHT = 900
BACKGROUND = "#f7f5ef"
PANEL_BG = "#fffdf9"
BORDER = "#d7d1c4"
GRID = "#e7e1d6"
AXIS = "#6b665e"
TEXT = "#1f1f1f"
TRAIN_COLOR = "#d06b1f"
VAL_COLOR = "#1f7a8c"
HIGHLIGHT = "#a32020"


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    total_epochs: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a training curve image from a train.log file."
    )
    parser.add_argument(
        "--log",
        type=Path,
        required=True,
        help="Path to the training log file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <log-dir>/training_curve.png.",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional title override for the generated image.",
    )
    return parser.parse_args()


def parse_metrics(log_path: Path) -> list[EpochMetrics]:
    metrics: list[EpochMetrics] = []
    for raw_line in log_path.read_text(encoding="utf-8").splitlines():
        match = EPOCH_PATTERN.match(raw_line.strip())
        if match is None:
            continue
        metrics.append(
            EpochMetrics(
                epoch=int(match.group(1)),
                total_epochs=int(match.group(2)),
                train_loss=float(match.group(3)),
                train_acc=float(match.group(4)),
                val_loss=float(match.group(5)),
                val_acc=float(match.group(6)),
            )
        )

    if not metrics:
        raise ValueError(f"No epoch metrics found in {log_path}")
    return metrics


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        "DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for candidate in font_candidates:
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


def relative_title(log_path: Path) -> str:
    try:
        return str(log_path.parent.relative_to(Path.cwd()))
    except ValueError:
        return str(log_path.parent)


def format_decimal(value: float) -> str:
    return f"{value:.2f}"


def format_percent(value: float) -> str:
    return f"{value:.1f}%"


def scaled_points(
    xs: list[int],
    ys: list[float],
    left: float,
    top: float,
    width: float,
    height: float,
    y_min: float,
    y_max: float,
) -> list[tuple[float, float]]:
    count = len(xs)
    points: list[tuple[float, float]] = []
    for index, (_, y_value) in enumerate(zip(xs, ys)):
        if count == 1:
            x = left + width / 2
        else:
            x = left + (width * index / (count - 1))
        y_ratio = 0.5 if y_max == y_min else (y_value - y_min) / (y_max - y_min)
        y = top + height - (y_ratio * height)
        points.append((x, y))
    return points


def draw_legend(
    draw: ImageDraw.ImageDraw,
    origin: tuple[float, float],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    x, y = origin
    for label, color in (("train", TRAIN_COLOR), ("val", VAL_COLOR)):
        draw.line((x, y + 8, x + 30, y + 8), fill=color, width=4)
        draw.text((x + 38, y), label, fill=TEXT, font=font)
        label_width, _ = text_size(draw, label, font)
        x += 38 + label_width + 28


def draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    epochs: list[int],
    train_values: list[float],
    val_values: list[float],
    y_label: str,
    value_formatter,
    main_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    small_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    highlight_index: int | None = None,
    highlight_label: str | None = None,
    y_floor: float | None = None,
    y_ceiling: float | None = None,
) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=20, fill=PANEL_BG, outline=BORDER, width=2)
    draw.text((x0 + 28, y0 + 20), title, fill=TEXT, font=main_font)

    legend_y = y0 + 20
    title_width, _ = text_size(draw, title, main_font)
    draw_legend(draw, (x0 + 40 + title_width, legend_y + 4), small_font)

    plot_left = x0 + 78
    plot_top = y0 + 88
    plot_right = x1 - 28
    plot_bottom = y1 - 70
    plot_width = plot_right - plot_left
    plot_height = plot_bottom - plot_top

    all_values = train_values + val_values
    y_min = min(all_values)
    y_max = max(all_values)
    if y_min == y_max:
        pad = max(0.1, abs(y_min) * 0.05)
    else:
        pad = (y_max - y_min) * 0.08
    y_min -= pad
    y_max += pad
    if y_floor is not None:
        y_min = max(y_floor, y_min)
    if y_ceiling is not None:
        y_max = min(y_ceiling, y_max)

    tick_count = 6
    for index in range(tick_count):
        tick_value = y_min + ((y_max - y_min) * index / (tick_count - 1))
        y = plot_bottom - (plot_height * index / (tick_count - 1))
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        label = value_formatter(tick_value)
        label_width, label_height = text_size(draw, label, small_font)
        draw.text(
            (plot_left - label_width - 12, y - label_height / 2),
            label,
            fill=AXIS,
            font=small_font,
        )

    max_x_ticks = 10
    step = max(1, math.ceil(len(epochs) / max_x_ticks))
    x_ticks = list(range(0, len(epochs), step))
    if x_ticks[-1] != len(epochs) - 1:
        x_ticks.append(len(epochs) - 1)
    for tick_index in x_ticks:
        if len(epochs) == 1:
            x = plot_left + plot_width / 2
        else:
            x = plot_left + (plot_width * tick_index / (len(epochs) - 1))
        draw.line((x, plot_top, x, plot_bottom), fill=GRID, width=1)
        label = str(epochs[tick_index])
        label_width, _ = text_size(draw, label, small_font)
        draw.text((x - label_width / 2, plot_bottom + 12), label, fill=AXIS, font=small_font)

    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=AXIS, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=AXIS, width=2)

    train_points = scaled_points(
        epochs, train_values, plot_left, plot_top, plot_width, plot_height, y_min, y_max
    )
    val_points = scaled_points(
        epochs, val_values, plot_left, plot_top, plot_width, plot_height, y_min, y_max
    )
    draw.line(train_points, fill=TRAIN_COLOR, width=4, joint="curve")
    draw.line(val_points, fill=VAL_COLOR, width=4, joint="curve")

    y_label_width, y_label_height = text_size(draw, y_label, small_font)
    draw.text(
        (x0 + 18, y0 + (y1 - y0 - y_label_height) / 2),
        y_label,
        fill=AXIS,
        font=small_font,
    )
    x_axis_label = "Epoch"
    x_axis_width, _ = text_size(draw, x_axis_label, small_font)
    draw.text(
        (plot_left + (plot_width - x_axis_width) / 2, y1 - 42),
        x_axis_label,
        fill=AXIS,
        font=small_font,
    )

    if highlight_index is not None and highlight_label is not None:
        hx, hy = val_points[highlight_index]
        radius = 6
        draw.ellipse(
            (hx - radius, hy - radius, hx + radius, hy + radius),
            fill=HIGHLIGHT,
            outline=PANEL_BG,
            width=2,
        )
        label_width, label_height = text_size(draw, highlight_label, small_font)
        bubble_x = min(max(hx + 16, plot_left + 12), plot_right - label_width - 18)
        bubble_y = max(hy - label_height - 18, plot_top + 8)
        bubble_box = (
            bubble_x - 8,
            bubble_y - 6,
            bubble_x + label_width + 8,
            bubble_y + label_height + 6,
        )
        draw.rounded_rectangle(bubble_box, radius=10, fill="#fff7ef", outline=BORDER, width=1)
        draw.text((bubble_x, bubble_y), highlight_label, fill=TEXT, font=small_font)
        draw.line((hx, hy, bubble_box[0], bubble_box[1] + 10), fill=HIGHLIGHT, width=2)


def create_training_curve(metrics: list[EpochMetrics], output_path: Path, title: str) -> None:
    image = Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), BACKGROUND)
    draw = ImageDraw.Draw(image)
    title_font = load_font(34)
    subtitle_font = load_font(20)
    panel_title_font = load_font(22)
    axis_font = load_font(18)

    draw.text((52, 38), title, fill=TEXT, font=title_font)

    best_accuracy = max(metrics, key=lambda item: item.val_acc)
    best_loss = min(metrics, key=lambda item: item.val_loss)
    final_epoch = metrics[-1]
    subtitle = (
        f"Best val acc: {best_accuracy.val_acc * 100:.2f}% @ epoch {best_accuracy.epoch}    "
        f"Best val loss: {best_loss.val_loss:.4f} @ epoch {best_loss.epoch}    "
        f"Final val acc: {final_epoch.val_acc * 100:.2f}%"
    )
    draw.text((54, 86), subtitle, fill=AXIS, font=subtitle_font)

    epochs = [item.epoch for item in metrics]
    train_loss = [item.train_loss for item in metrics]
    val_loss = [item.val_loss for item in metrics]
    train_acc = [item.train_acc * 100 for item in metrics]
    val_acc = [item.val_acc * 100 for item in metrics]

    top = 148
    left = 44
    gap = 28
    panel_width = (IMAGE_WIDTH - (left * 2) - gap) // 2
    panel_height = 690

    draw_panel(
        draw=draw,
        box=(left, top, left + panel_width, top + panel_height),
        title="Loss",
        epochs=epochs,
        train_values=train_loss,
        val_values=val_loss,
        y_label="Cross-Entropy",
        value_formatter=format_decimal,
        main_font=panel_title_font,
        small_font=axis_font,
        highlight_index=best_loss.epoch - 1,
        highlight_label=f"best val loss {best_loss.val_loss:.4f}",
        y_floor=0.0,
    )
    draw_panel(
        draw=draw,
        box=(left + panel_width + gap, top, IMAGE_WIDTH - left, top + panel_height),
        title="Accuracy",
        epochs=epochs,
        train_values=train_acc,
        val_values=val_acc,
        y_label="Accuracy",
        value_formatter=format_percent,
        main_font=panel_title_font,
        small_font=axis_font,
        highlight_index=best_accuracy.epoch - 1,
        highlight_label=f"best val acc {best_accuracy.val_acc * 100:.2f}%",
        y_floor=0.0,
        y_ceiling=100.0,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    args = parse_args()
    log_path = args.log.resolve()
    output_path = (args.output or (log_path.parent / "training_curve.png")).resolve()
    metrics = parse_metrics(log_path)
    title = args.title or f"Training Curves: {relative_title(log_path)}"
    create_training_curve(metrics, output_path, title)

    best_accuracy = max(metrics, key=lambda item: item.val_acc)
    print(f"Saved training curve to {output_path}")
    print(
        f"Best validation accuracy: {best_accuracy.val_acc * 100:.2f}% at epoch {best_accuracy.epoch}"
    )


if __name__ == "__main__":
    main()
