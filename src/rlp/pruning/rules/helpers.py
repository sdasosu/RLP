"""Shared helper functions for pruning operations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch.nn as nn

PASS_THRU_TYPES = {
    "view",
    "reshape",
    "flatten",
    "permute",
    "transpose",
    "relu",
    "relu_",
    "gelu",
    "silu",
    "hardswish",
    "Hardswish",
    "sigmoid",
    "tanh",
    "softmax",
    "avg_pool2d",
    "max_pool2d",
    "Dropout",
    "dropout",
    "AdaptiveAvgPool2d",
    "adaptive_avg_pool2d",
    "interpolate",
    "identity",
    "add",
    "add_",
}


def infer_conv_type_from_module(m: nn.Module) -> Optional[str]:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        groups = getattr(m, "groups", 1)
        if groups == m.in_channels == m.out_channels and groups > 1:
            return "depthwise"
        kernel = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        if all(k == 1 for k in kernel):
            return "pointwise"
        if groups > 1:
            return "grouped"
        return "standard"
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return "transpose"
    return None


def get_out_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return m.out_channels
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return m.out_channels
    if isinstance(m, nn.Linear):
        return m.out_features
    return None


def sanitize_idx(indices: List[int], upper: int) -> List[int]:
    cleaned = [i for i in indices if 0 <= i < upper]
    if not cleaned:
        k = min(upper, max(1, len(indices)))
        cleaned = list(range(k))
    return cleaned


def expand_linear_indices(
    linear: nn.Linear,
    indices: List[int],
    src_names: List[str],
    name2mod: Dict[str, nn.Module],
    pg,
) -> List[int]:
    if not indices:
        return indices
    in_features = getattr(linear, "in_features", None)
    if not in_features or in_features == len(indices):
        return indices

    if len(src_names) != 1:
        return indices

    src_name = src_names[0]
    src_module = name2mod.get(src_name)
    if src_module is None:
        return indices

    conv_like = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        nn.ConvTranspose1d,
        nn.ConvTranspose2d,
        nn.ConvTranspose3d,
    )
    if not isinstance(src_module, conv_like):
        return indices

    gid = pg.layer_to_group.get(src_name)
    if gid is None:
        return indices
    base_channels = pg.groups[gid]["base_channels"]
    if not base_channels:
        return indices

    stride = in_features // base_channels
    expected = stride * len(indices)
    if stride > 0 and expected <= in_features:
        expanded: List[int] = []
        for idx in indices:
            base = idx * stride
            expanded.extend(range(base, base + stride))
        return expanded
    return indices


def nearest_prunable_producer(
    pg, id2node: Dict[int, Dict[str, Any]], start_name: str
) -> Optional[str]:
    layer2id = {n["name"]: n["id"] for n in id2node.values() if n.get("is_module")}
    if start_name not in layer2id:
        return None
    start = layer2id[start_name]
    visited = set()
    queue: List[int] = [start]
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = id2node[node_id]
        for in_id in node.get("inputs", []):
            parent = id2node[in_id]
            if parent.get("is_module"):
                name = parent["name"]
                if name in pg.layer_to_group:
                    return name
                queue.append(in_id)
            else:
                if parent["type"] in PASS_THROUGH_TYPES:
                    queue.append(in_id)
    return None


def is_se_expand(dg, start_name: str, id2node: Dict[int, Dict[str, Any]]) -> bool:
    name2id = {n["name"]: n["id"] for n in dg.nodes if n.get("is_module")}
    if start_name not in name2id:
        return False
    start = name2id[start_name]
    visited = set()
    queue: List[int] = [start]
    steps = 0
    while queue and steps < 64:
        node_id = queue.pop(0)
        steps += 1
        if node_id in visited:
            continue
        visited.add(node_id)
        node = id2node[node_id]
        for out_id in node.get("outputs", []):
            child = id2node[out_id]
            if child.get("is_operation"):
                if child["type"] in ("mul", "*"):
                    return True
                queue.append(out_id)
                continue
            if child.get("is_module"):
                node_type = child["type"]
                if node_type in (
                    "Hardsigmoid",
                    "Sigmoid",
                    "ReLU",
                    "ReLU6",
                    "SiLU",
                    "GELU",
                    "Identity",
                    "AdaptiveAvgPool1d",
                    "AdaptiveAvgPool2d",
                    "AdaptiveAvgPool3d",
                    "AvgPool1d",
                    "AvgPool2d",
                    "AvgPool3d",
                ):
                    queue.append(out_id)
    return False


__all__ = [
    "infer_conv_type_from_module",
    "get_out_channels",
    "sanitize_idx",
    "expand_linear_indices",
    "nearest_prunable_producer",
    "is_se_expand",
    "PASS_THROUGH_TYPES",
]
