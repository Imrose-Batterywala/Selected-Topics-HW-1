from __future__ import annotations

import random

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    if targets.ndim > 1:
        targets = targets.argmax(dim=1)
    predictions = logits.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def update_confusion_matrix(
    confusion: torch.Tensor,
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    labels = targets.detach().to(device="cpu", dtype=torch.int64)
    preds = predictions.detach().to(device="cpu", dtype=torch.int64)
    encoded = labels * confusion.size(1) + preds
    confusion += torch.bincount(encoded, minlength=confusion.numel()).reshape(confusion.shape)


def balanced_accuracy_from_confusion_matrix(confusion: torch.Tensor) -> float:
    confusion = confusion.to(torch.float64)
    support = confusion.sum(dim=1)
    recall = confusion.diag() / support.clamp(min=1.0)
    valid = support > 0
    return recall[valid].mean().item()


def macro_f1_from_confusion_matrix(confusion: torch.Tensor) -> float:
    confusion = confusion.to(torch.float64)
    true_positives = confusion.diag()
    predicted = confusion.sum(dim=0)
    support = confusion.sum(dim=1)
    precision = true_positives / predicted.clamp(min=1.0)
    recall = true_positives / support.clamp(min=1.0)
    f1 = 2.0 * precision * recall / (precision + recall).clamp(min=1e-12)
    valid = support > 0
    return f1[valid].mean().item()
