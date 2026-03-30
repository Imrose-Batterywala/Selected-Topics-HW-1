#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import urlopen

import torch
from PIL import Image
import timm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from hw1_classifier.modeling import DEFAULT_BACKBONE, resolve_backbone_name


DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/"
    "beignets-task-guide.png"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run top-k ImageNet inference with a pretrained timm model.",
    )
    parser.add_argument("--image-url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--backbone", default=DEFAULT_BACKBONE)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def load_image(image_url: str) -> Image.Image:
    with urlopen(image_url) as response:
        image = Image.open(response)
        return image.convert("RGB")


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    backbone = resolve_backbone_name(args.backbone)

    image = load_image(args.image_url)
    model = timm.create_model(backbone, pretrained=True).to(device).eval()

    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    logits = model(transform(image).unsqueeze(0).to(device))
    probabilities = logits.softmax(dim=1)[0] * 100
    top_probabilities, top_indices = torch.topk(probabilities, k=args.topk)

    print(f"Backbone: {backbone}")
    print(f"Image: {args.image_url}")
    print(f"Device: {device}")
    for rank, (probability, class_index) in enumerate(zip(top_probabilities, top_indices), start=1):
        print(f"{rank}. class_index={class_index.item()} probability={probability.item():.2f}%")


if __name__ == "__main__":
    main()
