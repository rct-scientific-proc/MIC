"""ResNet builders with offline-friendly pretrained weight loading.

Weight sources, in order of precedence:
1. --weights-path <file.pth>  — an ImageNet state dict copied onto the machine
   (produced by download_weights.py on a connected box).
2. torchvision's normal cache (TORCH_HOME), downloading if online.
3. Random init (pretrained disabled).

The final fc layer is always replaced with a fresh K-class head after any
pretrained weights are loaded. For datasets that are not 1- or 3-channel,
the stem's first convolution is rebuilt for `in_channels` bands and
initialized from the pretrained RGB kernel (see adapt_input_conv), so
2-band or 4+-band imagery still benefits from ImageNet features.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torchvision import models

ARCHS = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, models.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
}


def adapt_input_conv(conv: nn.Conv2d, in_channels: int,
                     pretrained: bool = True) -> nn.Conv2d:
    """A copy of `conv` that accepts `in_channels` inputs. With pretrained
    weights the new kernel comes from the RGB one: 1 channel sums the three
    RGB kernels; otherwise they are tiled across the new channels and
    scaled by 3/in_channels, so an input of similar magnitude produces
    activations of similar magnitude (the timm recipe). Without pretrained
    weights the convolution is simply re-initialized at the new width."""
    new = nn.Conv2d(in_channels, conv.out_channels, conv.kernel_size,
                    conv.stride, conv.padding, conv.dilation, conv.groups,
                    bias=conv.bias is not None)
    if not pretrained:
        return new
    with torch.no_grad():
        w = conv.weight
        if in_channels == 1:
            new.weight.copy_(w.sum(dim=1, keepdim=True))
        else:
            reps = -(-in_channels // w.shape[1])
            new.weight.copy_(w.repeat(1, reps, 1, 1)[:, :in_channels]
                             * (w.shape[1] / in_channels))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_model(
    arch: str,
    num_classes: int,
    pretrained: bool = True,
    weights_path: str | None = None,
    in_channels: int = 3,
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
    if in_channels != 3:
        model.conv1 = adapt_input_conv(
            model.conv1, in_channels,
            pretrained=pretrained or weights_path is not None)
    return model


def init_classifier_prior(model: nn.Module, class_counts) -> np.ndarray:
    """Set the fresh head's bias to log(prior) so the untrained network
    predicts the training-time class frequencies instead of a uniform
    distribution (the RetinaNet / focal-loss prior initialisation). On a
    heavily skewed draw a uniform start means every hard negative is
    confidently wrong at step 0; the loss is then dominated by the majority
    class and the first epochs are spent learning the prior, during which
    validation recall sits at chance (the BatchNorm "collapse" transient).
    Starting at the prior removes that phase. Zero-count classes get a
    floor of one sample. Returns the prior that was applied."""
    counts = np.asarray(class_counts, dtype=np.float64)
    counts = np.maximum(counts, 1.0)
    prior = counts / counts.sum()
    with torch.no_grad():
        model.fc.bias.copy_(torch.as_tensor(np.log(prior), dtype=model.fc.bias.dtype))
    return prior


def weight_url(arch: str) -> str:
    """Download URL of the pretrained weights used by build_model."""
    return ARCHS[arch][1].url
