"""Concatenation traversal helpers for pruning."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple, Set

import torch.nn as nn

from .helpers import get_out_channels, expand_linear_indices, nearest_prunable_producer


def concat_sources(
    dg, target_name: str, id2node: Dict[int, Dict[str, Any]], pg
) -> List[Tuple[int, str, str]]:
    name2id = {n["name"]: n["id"] for n in dg.nodes if n.get("is_module")}
    if target_name not in name2id:
        return []
    start = name2id[target_name]
    visited: Set[Tuple[int, bool, bool]] = set()
    queue: List[Tuple[int, bool, bool]] = [(start, False, False)]
    outputs: List[Tuple[int, str, str]] = []
    prunable_names = set(pg.layer_to_group.keys())
    while queue:
        node_id, after_cat, after_add = queue.pop(0)
        state = (node_id, after_cat, after_add)
        if state in visited:
            continue
        visited.add(state)
        node = id2node[node_id]
        for in_id in node.get("inputs", []):
            parent = id2node[in_id]
            if parent.get("is_operation") and parent["type"] in (
                "cat",
                "concat",
                "stack",
            ):
                for src_id in parent.get("inputs", []):
                    module_node = id2node[src_id]
                    if module_node.get("is_module"):
                        if module_node["name"] in prunable_names:
                            outputs.append(
                                (module_node["id"], module_node["name"], "concat")
                            )
                        else:
                            queue.append((module_node["id"], True, after_add))
                    else:
                        queue.append((src_id, True, after_add))
                continue
            if parent.get("is_module"):
                if parent["name"] in prunable_names:
                    combiner = (
                        "concat" if after_cat else "add" if after_add else "direct"
                    )
                    outputs.append((parent["id"], parent["name"], combiner))
                else:
                    queue.append((in_id, after_cat, after_add))
            else:
                queue.append(
                    (
                        in_id,
                        after_cat,
                        after_add or parent.get("type") in ("add", "add_"),
                    )
                )
    return outputs


def direct_prunable_consumers(
    dg,
    producer_name: str,
    id2node: Dict[int, Dict[str, Any]],
    name2mod: Dict[str, nn.Module],
) -> List[Dict[str, Any]]:
    name2id = {n["name"]: n["id"] for n in dg.nodes if n.get("is_module")}
    if producer_name not in name2id:
        return []
    start = name2id[producer_name]
    visited = set()
    outputs: Dict[str, Dict[str, Any]] = {}
    queue: deque[Tuple[int, bool]] = deque([(start, False)])
    while queue:
        node_id, after_concat = queue.popleft()
        state = (node_id, after_concat)
        if state in visited:
            continue
        visited.add(state)
        node = id2node[node_id]
        for out_id in node.get("outputs", []):
            child = id2node[out_id]
            if child.get("is_module"):
                module = name2mod.get(child["name"])
                if module is None:
                    queue.append((out_id, after_concat))
                    continue
                if child["type"] in {
                    "Conv1d",
                    "Conv2d",
                    "Conv3d",
                    "ConvTranspose1d",
                    "ConvTranspose2d",
                    "ConvTranspose3d",
                    "Linear",
                }:
                    outputs.setdefault(
                        child["name"],
                        {
                            "name": child["name"],
                            "type": child["type"],
                            "conv_type": child.get("conv_type"),
                            "needs_concat": after_concat,
                        },
                    )
                    continue
                queue.append((out_id, after_concat))
                continue
            if child.get("is_operation"):
                next_concat = after_concat or child["type"] in (
                    "cat",
                    "concat",
                    "stack",
                )
                for grandchild_id in child.get("outputs", []):
                    queue.append((grandchild_id, next_concat))
                continue
            queue.append((out_id, after_concat))
    return list(outputs.values())


def compute_concat_indices(
    dg,
    pg,
    consumer_name: str,
    keep_map: Dict[int, List[int]],
    name2mod: Dict[str, nn.Module],
    id2node: Dict[int, Dict[str, Any]],
) -> Tuple[List[int], List[Dict[str, Any]]]:
    upstream = concat_sources(dg, consumer_name, id2node, pg)
    name2gid = pg.layer_to_group

    def resolve_indices(
        src_name: str, gid: int
    ) -> Tuple[List[int], int, Optional[str]]:
        if gid != -1:
            return keep_map.get(gid, []), gid, None
        module = name2mod.get(src_name)
        if module is not None:
            channels = get_out_channels(module)
            if channels is not None:
                return list(range(channels)), gid, None
        fallback = nearest_prunable_producer(pg, id2node, src_name)
        if fallback is None:
            raise RuntimeError(f"concat source '{src_name}' channels unknown")
        gid_fb = name2gid.get(fallback, -1)
        if gid_fb == -1:
            raise RuntimeError(f"concat source '{src_name}' channels unknown")
        return keep_map.get(gid_fb, []), gid_fb, fallback

    full: List[int] = []
    branch_info: List[Dict[str, Any]] = []
    offset = 0
    add_seen = False

    for entry in upstream:
        if len(entry) == 3:
            _, name, combiner = entry
        else:
            _, name = entry
            combiner = "direct"

        gid = name2gid.get(name, -1)
        indices, resolved_gid, alias = resolve_indices(name, gid)
        source_name = alias if alias is not None else name

        if combiner == "add":
            if add_seen:
                continue
            full = list(indices)
            branch_info.append(
                {
                    "name": source_name,
                    "gid": resolved_gid,
                    "offset": 0,
                    "indices": indices,
                    "length": len(indices),
                    "combiner": "add",
                }
            )
            add_seen = True
            continue

        local_range = list(range(len(indices)))
        full.extend(offset + i for i in local_range)
        branch_info.append(
            {
                "name": source_name,
                "gid": resolved_gid,
                "offset": offset,
                "indices": indices,
                "length": len(local_range),
                "combiner": combiner,
            }
        )
        offset += len(local_range)

    return full, branch_info


__all__ = ["concat_sources", "direct_prunable_consumers", "compute_concat_indices"]
