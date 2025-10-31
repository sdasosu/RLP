"""Factories for ResNet backbone variants."""

from __future__ import annotations

from .utils import check_in_channels, replace_linear_layer, torchvision_models


def _build_resnet(model_name: str, weight_enum, num_classes: int) -> object:
    model_fn = getattr(torchvision_models, model_name)
    model = model_fn(weights=weight_enum)
    replace_linear_layer(model, ("fc",), num_classes)
    return model


def resnet18(in_channels: int, num_classes: int, input_size=None) -> object:
    check_in_channels("resnet18", in_channels)
    return _build_resnet(
        "resnet18", torchvision_models.ResNet18_Weights.IMAGENET1K_V1, num_classes
    )


def resnet34(in_channels: int, num_classes: int, input_size=None) -> object:
    check_in_channels("resnet34", in_channels)
    return _build_resnet(
        "resnet34", torchvision_models.ResNet34_Weights.IMAGENET1K_V1, num_classes
    )


def resnet50(in_channels: int, num_classes: int, input_size=None) -> object:
    check_in_channels("resnet50", in_channels)
    return _build_resnet(
        "resnet50", torchvision_models.ResNet50_Weights.IMAGENET1K_V1, num_classes
    )


def resnet101(in_channels: int, num_classes: int, input_size=None) -> object:
    check_in_channels("resnet101", in_channels)
    return _build_resnet(
        "resnet101", torchvision_models.ResNet101_Weights.IMAGENET1K_V1, num_classes
    )


def resnet152(in_channels: int, num_classes: int, input_size=None) -> object:
    check_in_channels("resnet152", in_channels)
    return _build_resnet(
        "resnet152", torchvision_models.ResNet152_Weights.IMAGENET1K_V1, num_classes
    )


__all__ = ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"]
