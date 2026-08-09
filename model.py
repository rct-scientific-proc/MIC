"""ResNet builders with offline-friendly pretrained weight loading.

Weight sources, in order of precedence:
1. --weights-path <file.pth>  — an ImageNet state dict copied onto the machine
   (produced by download_weights.py on a connected box).
2. torchvision's normal cache (TORCH_HOME), downloading if online.
3. Random init (pretrained disabled).

The final fc layer is always replaced with a fresh K-class head after any
pretrained weights are loaded.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    weights_path: str | None = None,
) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {sorted(ARCHS)}, got '{arch}'")
    ctor, default_weights = ARCHS[arch]

    if weights_path is not None:
        model = ctor(weights=None)  # 1000-class ImageNet shape
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    elif pretrained:
        model = ctor(weights=default_weights)  # TORCH_HOME cache / download
    else:
        model = ctor(weights=None)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def weight_url(arch: str) -> str:
    """Download URL of the pretrained weights used by build_model."""
    return ARCHS[arch][1].url
