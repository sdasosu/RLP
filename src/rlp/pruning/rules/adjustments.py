"""BatchNorm adjustments used during pruning."""

from __future__ import annotations

from typing import Dict, List, Set

import torch.nn as nn

from .slicers import slice_batchnorm


def adjust_bn_chain(
    dg, name2mod, id2node, start_name: str, keep_idx: List[int]
) -> None:
    name2id = {n["name"]: n["id"] for n in dg.nodes if n.get("is_module")}
    if start_name not in name2id:
        return
    start = name2id[start_name]
    visited: Set[int] = set()
    queue: List[int] = [start]
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = id2node[node_id]
        for out_id in node.get("outputs", []):
            child = id2node[out_id]
            if child.get("is_operation"):
                if child["type"] in ("cat", "concat", "stack"):
                    continue
                queue.append(out_id)
                continue
            if child.get("is_module"):
                module = name2mod.get(child["name"])
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    if "denseblock" in child["name"] and ".norm1" in child["name"]:
                        continue
                    slice_batchnorm(module, keep_idx)
                    queue.append(out_id)
                    continue
                if isinstance(
                    module,
                    (
                        nn.ReLU,
                        nn.ReLU6,
                        nn.SiLU,
                        nn.GELU,
                        nn.Identity,
                        nn.Dropout,
                        nn.MaxPool2d,
                        nn.AvgPool2d,
                        nn.AdaptiveAvgPool2d,
                    ),
                ):
                    queue.append(out_id)
                    continue
            else:
                queue.append(out_id)


def adjust_concat_pre_norms(
    dg,
    id2node: Dict[int, Dict[str, object]],
    name2mod: Dict[str, nn.Module],
    consumer_name: str,
    keep_idx: List[int],
) -> None:
    if not keep_idx:
        return
    name2id = {n["name"]: n["id"] for n in dg.nodes if n.get("is_module")}
    if consumer_name not in name2id:
        return
    target_id = name2id[consumer_name]
    visited: Set[int] = set()
    queue: List[int] = [target_id]
    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)
        node = id2node[node_id]
        for in_id in node.get("inputs", []):
            src = id2node[in_id]
            if src.get("is_operation"):
                if src["type"] in ("cat", "concat", "stack"):
                    continue
                queue.append(in_id)
                continue
            if not src.get("is_module"):
                queue.append(in_id)
                continue
            module = name2mod.get(src["name"])
            if module is None:
                queue.append(in_id)
                continue
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                slice_batchnorm(module, keep_idx)
                queue.append(in_id)
                continue
            if isinstance(
                module,
                (
                    nn.ReLU,
                    nn.ReLU6,
                    nn.LeakyReLU,
                    nn.SiLU,
                    nn.GELU,
                    nn.Identity,
                    nn.Dropout,
                    nn.MaxPool2d,
                    nn.AvgPool2d,
                    nn.AdaptiveAvgPool2d,
                    nn.Hardswish,
                    nn.Hardsigmoid,
                ),
            ):
                queue.append(in_id)


__all__ = ["adjust_bn_chain", "adjust_concat_pre_norms"]
