#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, TextIO

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw1_classifier.data import build_dataloaders, discover_class_names
from hw1_classifier.modeling import (
    AUTO_AUGMENT_POLICIES,
    BACKBONE_ALIASES,
    DEFAULT_BACKBONE,
    create_classifier,
    create_mixup,
    create_transforms,
    resolve_backbone_name,
)
from hw1_classifier.utils import accuracy, set_seed


class SoftTargetCrossEntropy(nn.Module):
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.sum(-targets * F.log_softmax(logits, dim=-1), dim=-1).mean()


class TeeStream:
    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        return any(getattr(stream, "isatty", lambda: False)() for stream in self.streams)


@contextmanager
def mirrored_output(log_path: Path) -> Iterator[None]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = TeeStream(original_stdout, log_handle)
    sys.stderr = TeeStream(original_stderr, log_handle)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a timm classifier on the HW1 dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        help="Path to the training log file. Defaults to <output-dir>/train.log.",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default=DEFAULT_BACKBONE,
        help=(
            "timm model name or alias "
            f"({', '.join(f'{alias}={name}' for alias, name in sorted(BACKBONE_ALIASES.items()))})"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augmentation-policy",
        choices=sorted(AUTO_AUGMENT_POLICIES),
        default="randaugment",
    )
    parser.add_argument("--hflip", type=float, default=0.5)
    parser.add_argument("--vflip", type=float, default=0.5)
    parser.add_argument("--color-jitter", type=float, default=0.4)
    parser.add_argument("--grayscale-prob", type=float, default=0.1)
    parser.add_argument("--gaussian-blur-prob", type=float, default=0.1)
    parser.add_argument("--random-erasing-prob", type=float, default=0.25)
    parser.add_argument("--random-erasing-mode", type=str, default="pixel")
    parser.add_argument("--random-erasing-count", type=int, default=1)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.2)
    parser.add_argument("--cutmix-alpha", type=float, default=1.0)
    parser.add_argument("--mixup-prob", type=float, default=1.0)
    parser.add_argument("--mixup-switch-prob", type=float, default=0.5)
    parser.add_argument("--mixup-mode", choices=["batch", "pair", "elem"], default="batch")
    return parser.parse_args()


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    train_criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    mixup_fn,
) -> tuple[float, float]:
    model.train()
    amp_enabled = device.type == "cuda"
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        targets_for_loss = targets
        if mixup_fn is not None:
            inputs, targets_for_loss = mixup_fn(inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logits = model(inputs)
            loss = train_criterion(logits, targets_for_loss)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy(logits.detach(), targets_for_loss) * batch_size
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    eval_criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(inputs)
        loss = eval_criterion(logits, targets)

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


def run_training(args: argparse.Namespace) -> tuple[float, int, Path, Path]:
    use_mixup = args.mixup_alpha > 0.0 or args.cutmix_alpha > 0.0
    if use_mixup and args.batch_size % 2 != 0:
        raise ValueError("MixUp/CutMix requires an even --batch-size.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved_backbone = resolve_backbone_name(args.backbone)
    class_names = discover_class_names(args.data_dir / "train")
    num_classes = len(class_names)
    bootstrap_model = create_classifier(
        num_classes=num_classes,
        backbone=resolved_backbone,
        pretrained=True,
    )
    train_transform, eval_transform, data_config = create_transforms(
        bootstrap_model,
        auto_augment_policy=args.augmentation_policy,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        random_erasing_prob=args.random_erasing_prob,
        random_erasing_mode=args.random_erasing_mode,
        random_erasing_count=args.random_erasing_count,
    )
    train_loader, val_loader, class_names = build_dataloaders(
        data_dir=args.data_dir,
        train_transform=train_transform,
        eval_transform=eval_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last_train=use_mixup,
    )

    model = bootstrap_model.to(device)
    mixup_fn = create_mixup(
        num_classes=num_classes,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        mix_prob=args.mixup_prob,
        switch_prob=args.mixup_switch_prob,
        mode=args.mixup_mode,
        label_smoothing=args.label_smoothing,
    )
    train_criterion = (
        SoftTargetCrossEntropy()
        if mixup_fn is not None
        else nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    )
    eval_criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint_path = output_dir / "best_model.pt"
    last_checkpoint_path = output_dir / "last_model.pt"

    print(f"Backbone: {resolved_backbone}")
    if args.backbone != resolved_backbone:
        print(f"Backbone alias: {args.backbone}")
    print(f"Device: {device}")
    print(f"Classes: {num_classes}")
    print(f"Train images: {len(train_loader.dataset)}")
    print(f"Validation images: {len(val_loader.dataset)}")
    print(f"Input config: {data_config}")
    print(
        "Augmentations: "
        f"policy={args.augmentation_policy}, hflip={args.hflip}, vflip={args.vflip}, "
        f"color_jitter={args.color_jitter}, grayscale_prob={args.grayscale_prob}, "
        f"gaussian_blur_prob={args.gaussian_blur_prob}, re_prob={args.random_erasing_prob}"
    )
    print(
        "Label mixing: "
        f"mixup_alpha={args.mixup_alpha}, cutmix_alpha={args.cutmix_alpha}, "
        f"mix_prob={args.mixup_prob}, switch_prob={args.mixup_switch_prob}, mode={args.mixup_mode}"
    )

    best_val_accuracy = float("-inf")
    best_epoch = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            train_criterion=train_criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            mixup_fn=mixup_fn,
        )
        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            eval_criterion=eval_criterion,
            device=device,
        )
        scheduler.step()

        checkpoint = {
            "model_state": model.state_dict(),
            "class_names": class_names,
            "backbone": resolved_backbone,
            "backbone_alias": args.backbone,
            "epoch": epoch,
            "val_accuracy": val_acc,
            "data_config": data_config,
            "augmentations": {
                "policy": args.augmentation_policy,
                "hflip": args.hflip,
                "vflip": args.vflip,
                "color_jitter": args.color_jitter,
                "grayscale_prob": args.grayscale_prob,
                "gaussian_blur_prob": args.gaussian_blur_prob,
                "random_erasing_prob": args.random_erasing_prob,
                "random_erasing_mode": args.random_erasing_mode,
                "random_erasing_count": args.random_erasing_count,
            },
            "label_mixing": {
                "label_smoothing": args.label_smoothing,
                "mixup_alpha": args.mixup_alpha,
                "cutmix_alpha": args.cutmix_alpha,
                "mixup_prob": args.mixup_prob,
                "mixup_switch_prob": args.mixup_switch_prob,
                "mixup_mode": args.mixup_mode,
            },
        }
        torch.save(checkpoint, last_checkpoint_path)
        if val_acc >= best_val_accuracy:
            best_val_accuracy = val_acc
            best_epoch = epoch
            torch.save(checkpoint, best_checkpoint_path)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    return best_val_accuracy, best_epoch, best_checkpoint_path, last_checkpoint_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = (args.log_file or (output_dir / "train.log")).resolve()

    with mirrored_output(log_path):
        print(f"Training log: {log_path}")
        best_val_accuracy, best_epoch, best_checkpoint_path, last_checkpoint_path = run_training(args)

    print(f"Best validation accuracy: {best_val_accuracy:.4f} (epoch {best_epoch})")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"Last checkpoint: {last_checkpoint_path}")


if __name__ == "__main__":
    main()
