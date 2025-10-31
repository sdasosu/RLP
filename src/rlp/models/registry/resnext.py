"""Factories for ResNeXt variants."""

from __future__ import annotations

from .utils import check_in_channels, replace_linear_layer, torchvision_models


def resnext50_32x4d(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("resnext50_32x4d", in_channels)
    model = torchvision_models.resnext50_32x4d(
        weights=torchvision_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("fc",), num_classes)
    return model


def resnext101_32x8d(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("resnext101_32x8d", in_channels)
    model = torchvision_models.resnext101_32x8d(
        weights=torchvision_models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("fc",), num_classes)
    return model


__all__ = ["resnext50_32x4d", "resnext101_32x8d"]
