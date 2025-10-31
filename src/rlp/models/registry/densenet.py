"""Factories for DenseNet variants."""

from __future__ import annotations

from .utils import check_in_channels, replace_linear_layer, torchvision_models


def _build_densenet(model_name: str, weight_enum, num_classes: int):
    model_fn = getattr(torchvision_models, model_name)
    model = model_fn(weights=weight_enum)
    replace_linear_layer(model, ("classifier",), num_classes)
    return model


def densenet121(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("densenet121", in_channels)
    return _build_densenet(
        "densenet121",
        torchvision_models.DenseNet121_Weights.IMAGENET1K_V1,
        num_classes,
    )


def densenet161(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("densenet161", in_channels)
    return _build_densenet(
        "densenet161",
        torchvision_models.DenseNet161_Weights.IMAGENET1K_V1,
        num_classes,
    )


def densenet169(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("densenet169", in_channels)
    return _build_densenet(
        "densenet169",
        torchvision_models.DenseNet169_Weights.IMAGENET1K_V1,
        num_classes,
    )


def densenet201(in_channels: int, num_classes: int, input_size=None):
    check_in_channels("densenet201", in_channels)
    return _build_densenet(
        "densenet201",
        torchvision_models.DenseNet201_Weights.IMAGENET1K_V1,
        num_classes,
    )


__all__ = ["densenet121", "densenet161", "densenet169", "densenet201"]
