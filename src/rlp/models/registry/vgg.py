"""Factories for VGG family models."""

from __future__ import annotations

from .utils import (
    adapt_vgg_for_small_input,
    check_in_channels,
    replace_linear_layer,
    torchvision_models,
)


def vgg11(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg11", in_channels)
    model = torchvision_models.vgg11(
        weights=torchvision_models.VGG11_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


def vgg11_bn(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg11_bn", in_channels)
    model = torchvision_models.vgg11_bn(
        weights=torchvision_models.VGG11_BN_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


def vgg16(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg16", in_channels)
    model = torchvision_models.vgg16(
        weights=torchvision_models.VGG16_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


def vgg16_bn(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg16_bn", in_channels)
    model = torchvision_models.vgg16_bn(
        weights=torchvision_models.VGG16_BN_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


def vgg19(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg19", in_channels)
    model = torchvision_models.vgg19(
        weights=torchvision_models.VGG19_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


def vgg19_bn(in_channels: int, num_classes: int, input_size=None) -> nn.Module:
    check_in_channels("vgg19_bn", in_channels)
    model = torchvision_models.vgg19_bn(
        weights=torchvision_models.VGG19_BN_Weights.IMAGENET1K_V1
    )
    replace_linear_layer(model, ("classifier", -1), num_classes)
    return adapt_vgg_for_small_input(model, num_classes, input_size)


__all__ = ["vgg11", "vgg11_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn"]
