"""Shared constants and helpers for pruning group construction."""

from __future__ import annotations

from math import gcd
from typing import Optional

import torch.nn as nn

PRUNABLE_TYPES = {
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "Linear",
}

PASS_THROUGH_TYPES = {
    "BatchNorm1d",
    "BatchNorm2d",
    "BatchNorm3d",
    "ReLU",
    "ReLU6",
    "SiLU",
    "GELU",
    "Identity",
    "Dropout",
    "MaxPool2d",
    "AvgPool2d",
    "AdaptiveAvgPool2d",
}


def flow_out_channels(module: nn.Module, conv_type: Optional[str]) -> Optional[int]:
    """Return the number of channels that flow out of a producer module."""
    if isinstance(module, nn.Linear):
        return module.out_features
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if conv_type == "depthwise":
            return module.in_channels
        return module.out_channels
    if isinstance(module, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return module.out_channels
    return None


def lcm(a: int, b: int) -> int:
    """Least common multiple that supports zeros gracefully."""
    return abs(a * b) // gcd(a, b) if a and b else max(a, b)
