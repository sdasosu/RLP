"""Factories for MobileNet variants."""

from __future__ import annotations

from .utils import check_in_channels, replace_linear_layer, torchvision_models


def mobilenet_v3_small(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("mobilenet_v3_small", in_channels)
    model = torchvision_models.mobilenet_v3_small(
        weights=torchvision_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return model


def mobilenet_v3_large(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("mobilenet_v3_large", in_channels)
    model = torchvision_models.mobilenet_v3_large(
        weights=torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return model


__all__ = ["mobilenet_v3_small", "mobilenet_v3_large"]
