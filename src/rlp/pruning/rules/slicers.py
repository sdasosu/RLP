"""Utilities to slice module weights and activations after pruning decisions."""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from .helpers import infer_conv_type_from_module, sanitize_idx


def slice_conv_out(m: nn.Conv2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    m.out_channels = int(keep.numel())


def slice_depthwise_conv(m: nn.Conv2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    channels = int(keep.numel())
    m.in_channels = channels
    m.out_channels = channels
    m.groups = channels


def slice_conv_transpose_out(m: nn.ConvTranspose2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.out_channels = int(keep.numel())


def slice_linear_out(m: nn.Linear, keep_idx: List[int]) -> None:
    keep = torch.tensor(sanitize_idx(keep_idx, m.out_features), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    m.out_features = int(keep.numel())


def slice_batchnorm(m: nn.Module, keep_idx: List[int]) -> None:
    if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return
    c = m.num_features
    if len(keep_idx) == c:
        return
    keep = torch.tensor(sanitize_idx(keep_idx, c), dtype=torch.long)
    if hasattr(m, "weight") and m.weight is not None:
        m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if hasattr(m, "bias") and m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    if hasattr(m, "running_mean") and m.running_mean is not None:
        m.running_mean = m.running_mean.data.index_select(0, keep).contiguous()
    if hasattr(m, "running_var") and m.running_var is not None:
        m.running_var = m.running_var.data.index_select(0, keep).contiguous()
    m.num_features = int(keep.numel())


def slice_conv_in_grouped(m: nn.Conv2d, keep_idx: List[int]) -> None:
    group = max(1, getattr(m, "groups", 1))
    in_per_group = m.in_channels // group if group else m.in_channels
    keep = sorted(sanitize_idx(keep_idx, m.in_channels))
    if not keep:
        keep = list(range(min(in_per_group, m.in_channels)))

    per_group: List[List[int]] = [[] for _ in range(group)]
    for idx in keep:
        g = min(idx // max(1, in_per_group), group - 1)
        per_group[g].append(idx % max(1, in_per_group))

    target_len = max(1, len(keep) // max(1, group))
    target_len = max(1, min(target_len, in_per_group))

    sanitized_groups = []
    for lst in per_group:
        if len(lst) < target_len:
            base = list(range(in_per_group))
            lst = (lst + base)[:target_len]
        sanitized_groups.append(sorted(lst[:target_len]))

    weight = m.weight.data.view(
        group, m.out_channels // group, in_per_group, *m.weight.shape[2:]
    )
    selected = []
    for g, idxs in enumerate(sanitized_groups):
        idx_tensor = torch.tensor(idxs, dtype=torch.long)
        selected.append(weight[g].index_select(1, idx_tensor))
    new_weight = torch.stack(selected, dim=0).contiguous()
    m.weight = nn.Parameter(
        new_weight.view(m.out_channels, target_len, *m.weight.shape[2:])
    )
    m.in_channels = int(target_len * group)


def slice_conv_in(m: nn.Conv2d, keep_idx: List[int]) -> None:
    assert not (m.groups == m.in_channels == m.out_channels and m.groups > 1), (
        "Depthwise conv reached slice_conv_in; handle via slice_depthwise_conv"
    )
    if m.groups > 1:
        slice_conv_in_grouped(m, keep_idx)
        return
    keep = torch.tensor(sanitize_idx(keep_idx, m.in_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.in_channels = int(keep.numel())


def slice_linear_in(m: nn.Linear, keep_idx: List[int]) -> None:
    keep = torch.tensor(sanitize_idx(keep_idx, m.in_features), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.in_features = int(keep.numel())


def slice_consumer_in(
    module: nn.Module,
    keep_idx: List[int],
    conv_type: Optional[str],
    producer_type: Optional[str] = None,
    producer_base_channels: Optional[int] = None,
) -> List[int]:
    if isinstance(module, nn.Linear):
        expanded = keep_idx
        if keep_idx and getattr(module, "in_features", None):
            if (
                producer_type
                in {
                    "Conv1d",
                    "Conv2d",
                    "Conv3d",
                    "ConvTranspose1d",
                    "ConvTranspose2d",
                    "ConvTranspose3d",
                }
                and producer_base_channels
            ):
                stride = module.in_features // producer_base_channels
                expected = stride * len(keep_idx)
                if stride > 0 and expected <= module.in_features:
                    expanded = []
                    for idx in keep_idx:
                        base = idx * stride
                        expanded.extend(range(base, base + stride))
            if not expanded:
                expanded = keep_idx
        slice_linear_in(module, expanded)
        return list(expanded)

    if isinstance(module, nn.ConvTranspose2d):
        sanitized = sanitize_idx(keep_idx, module.out_channels)
        keep = torch.tensor(sanitized, dtype=torch.long)
        module.weight = nn.Parameter(
            module.weight.data.index_select(0, keep).contiguous()
        )
        module.out_channels = int(keep.numel())
        return sanitized

    if isinstance(module, nn.Conv2d):
        conv_type = conv_type or infer_conv_type_from_module(module)
        if conv_type == "depthwise" or (
            module.groups == module.in_channels == module.out_channels
            and module.groups > 1
        ):
            sanitized = sanitize_idx(keep_idx, module.out_channels)
            slice_depthwise_conv(module, sanitized)
            return sanitized
        if module.groups > 1:
            sanitized = sanitize_idx(keep_idx, module.in_channels)
            slice_conv_in_grouped(module, sanitized)
            return sanitized
        sanitized = sanitize_idx(keep_idx, module.in_channels)
        slice_conv_in(module, sanitized)
        return sanitized

    return list(keep_idx)


__all__ = [
    "slice_conv_out",
    "slice_depthwise_conv",
    "slice_conv_transpose_out",
    "slice_linear_out",
    "slice_batchnorm",
    "slice_conv_in_grouped",
    "slice_conv_in",
    "slice_linear_in",
    "slice_consumer_in",
]
