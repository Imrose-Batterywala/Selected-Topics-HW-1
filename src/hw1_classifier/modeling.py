from __future__ import annotations

from typing import Any

import timm
import torch.nn as nn


DEFAULT_BACKBONE = "seresnextaa101d_32x8d.sw_in12k_ft_in1k"
BACKBONE_ALIASES: dict[str, str] = {
    "resnet152": DEFAULT_BACKBONE,
    "seresnext101": "seresnext101_32x4d.gluon_in1k",
    "seresnext101_32x4d": "seresnext101_32x4d.gluon_in1k",
    "seresnextaa101d": "seresnextaa101d_32x8d.sw_in12k_ft_in1k",
    "seresnextaa101d_32x8d": "seresnextaa101d_32x8d.sw_in12k_ft_in1k",
}
AUTO_AUGMENT_POLICIES = {
    "none": None,
    "autoaugment": "original-mstd0.5",
    "randaugment": "rand-m9-mstd0.5-inc1",
    "augmix": "augmix-m5-w4-d2",
}


def resolve_backbone_name(backbone: str) -> str:
    return BACKBONE_ALIASES.get(backbone, backbone)


def create_classifier(
    num_classes: int,
    backbone: str = DEFAULT_BACKBONE,
    pretrained: bool = True,
) -> nn.Module:
    backbone = resolve_backbone_name(backbone)
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


def create_transforms(
    model: nn.Module,
    *,
    auto_augment_policy: str = "randaugment",
    hflip: float = 0.5,
    vflip: float = 0.5,
    color_jitter: float = 0.4,
    grayscale_prob: float = 0.1,
    gaussian_blur_prob: float = 0.1,
    random_erasing_prob: float = 0.25,
    random_erasing_mode: str = "pixel",
    random_erasing_count: int = 1,
) -> tuple[Any, Any, dict[str, Any]]:
    data_config = timm.data.resolve_model_data_config(model)
    train_transform = timm.data.create_transform(
        **data_config,
        is_training=True,
        auto_augment=AUTO_AUGMENT_POLICIES[auto_augment_policy],
        hflip=hflip,
        vflip=vflip,
        color_jitter=color_jitter,
        grayscale_prob=grayscale_prob,
        gaussian_blur_prob=gaussian_blur_prob,
        re_prob=random_erasing_prob,
        re_mode=random_erasing_mode,
        re_count=random_erasing_count,
    )
    eval_transform = timm.data.create_transform(**data_config, is_training=False)
    return train_transform, eval_transform, data_config


def create_mixup(
    *,
    num_classes: int,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    mix_prob: float = 1.0,
    switch_prob: float = 0.5,
    mode: str = "batch",
    label_smoothing: float = 0.1,
):
    if mixup_alpha <= 0.0 and cutmix_alpha <= 0.0:
        return None
    return timm.data.Mixup(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        prob=mix_prob,
        switch_prob=switch_prob,
        mode=mode,
        label_smoothing=label_smoothing,
        num_classes=num_classes,
    )
