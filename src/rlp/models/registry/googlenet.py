"""Factory for GoogLeNet."""

from __future__ import annotations

from .utils import check_in_channels, replace_linear_layer, torchvision_models


def googlenet(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("googlenet", in_channels)
    model = torchvision_models.googlenet(
        weights=torchvision_models.GoogLeNet_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("fc",), num_classes)
    return model


__all__ = ["googlenet"]
