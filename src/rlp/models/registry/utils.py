"""Shared helpers for torchvision model adaptation."""

from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn
from torchvision import models as torchvision_models


def check_in_channels(model_name: str, in_channels: int) -> None:
    if in_channels != 3:
        print(
            f"Warning: torchvision model '{model_name}' is designed for 3 input channels. "
            f"The 'in_channels={in_channels}' argument will be ignored."
        )


def replace_linear_layer(
    parent: nn.Module, attr_path: Sequence, num_classes: int
) -> None:
    module = parent
    for attr in attr_path[:-1]:
        module = module[attr] if isinstance(attr, int) else getattr(module, attr)
    last = attr_path[-1]
    target = module[last] if isinstance(last, int) else getattr(module, last)
    if not isinstance(target, nn.Linear):
        raise TypeError("Expected a nn.Linear module to replace classifier head.")
    in_features = target.in_features
    new_layer = nn.Linear(in_features, num_classes)
    if isinstance(last, int):
        module[last] = new_layer
    else:
        setattr(module, last, new_layer)


def adapt_vgg_for_small_input(
    model: nn.Module, num_classes: int, input_size: Iterable[int] | None
) -> nn.Module:
    if not input_size:
        return model
    dims = list(input_size)
    if len(dims) < 3:
        return model
    _, height, width = dims
    if height >= 64 and width >= 64:
        return model

    model.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    with torch.no_grad():
        dummy = torch.zeros(1, *dims)
        features_out = model.features(dummy)
        pooled = model.avgpool(features_out)
        flat_dim = pooled.view(1, -1).size(1)

    hidden_dim = max(256, flat_dim)
    model.classifier = nn.Sequential(
        nn.Linear(flat_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Linear(hidden_dim, num_classes),
    )
    return model


__all__ = [
    "torchvision_models",
    "check_in_channels",
    "replace_linear_layer",
    "adapt_vgg_for_small_input",
]
