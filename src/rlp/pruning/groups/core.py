"""Core implementation of pruning group discovery."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set

import torch.nn as nn

from .common import PRUNABLE_TYPES, flow_out_channels, lcm
from .helpers import calculate_min_channels, create_action_space, pick_canonical_member
from .traversal import GraphTraversal


class PruningGroups:
    """Derive structured pruning groups from a dependency graph."""

    def __init__(self, dg, model: nn.Module):
        self.dg = dg
        self.model = model
        self.id2node = {n["id"]: n for n in dg.nodes}
        self.name2node = {n["name"]: n for n in dg.nodes if n.get("is_module")}
        self.name2module = dict(model.named_modules())

        self.groups: List[Dict[str, Any]] = []
        self.layer_to_group: Dict[str, int] = {}
        self.residual_components: List[Set[int]] = []

        self._traversal = GraphTraversal(self.id2node, self.name2node, self.name2module)

    # --------------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------------- #
    def build(self) -> List[Dict[str, Any]]:
        prunable_layers = self._find_prunable_layers()
        mirror_groups = self._find_mirror_groups(prunable_layers)
        self._build_groups(mirror_groups)
        self._build_residual_components()
        return self.groups

    def synchronize_decisions(self, decisions: Dict[int, int]) -> Dict[int, int]:
        if not self.residual_components:
            return decisions

        gid2group = {g["group_id"]: g for g in self.groups}

        for component in self.residual_components:
            upper_bound = min(
                decisions.get(gid, gid2group[gid]["max_channels"]) for gid in component
            )
            feasible: Optional[Set[int]] = None
            for gid in component:
                space = set(gid2group[gid]["action_space"])
                feasible = space if feasible is None else feasible & space
                if not feasible:
                    break

            if not feasible:
                target = min(gid2group[gid]["min_channels"] for gid in component)
            else:
                candidates = [value for value in feasible if value <= upper_bound]
                target = max(candidates) if candidates else min(feasible)

            for gid in component:
                decisions[gid] = target

        return decisions

    def print_groups(self, verbose: bool = False) -> None:
        print("\n" + "=" * 80)
        print(f"Pruning Groups: {len(self.groups)} groups")
        print("=" * 80)
        for group in self.groups:
            print(f"\n[Group {group['group_id']}] {len(group['members'])} layers")
            print(
                f"  Producer: {group['producer']} "
                f"({group['producer_type']}, {group['producer_conv_type']})"
            )
            print(f"  Base channels: {group['base_channels']}")
            print(
                f"  Range: [{group['min_channels']}, {group['max_channels']}]  "
                f"Div: {group['divisibility']}"
            )
            print(f"  Action space: {len(group['action_space'])} choices")
            if verbose:
                print(f"  Members: {group['members']}")
                if group["consumers"]:
                    print(f"  Consumers ({len(group['consumers'])}):")
                    for consumer in group["consumers"][:6]:
                        tag = (
                            "after_concat" if consumer.get("needs_concat") else "direct"
                        )
                        print(
                            f"    - {consumer['name']} "
                            f"({consumer['type']}, {consumer['conv_type']}, {tag})"
                        )
                    if len(group["consumers"]) > 6:
                        print(f"    ... and {len(group['consumers']) - 6} more")
        print("\n" + "=" * 80)
        print(f"Total prunable params: {self._count_total_params()}")
        print("=" * 80 + "\n")

    def export_for_rl(self) -> Dict[str, Any]:
        return {
            "num_groups": len(self.groups),
            "action_spaces": [group["action_space"] for group in self.groups],
            "group_info": [
                {
                    "group_id": group["group_id"],
                    "members": group["members"],
                    "producer": group["producer"],
                    "producer_type": group["producer_type"],
                    "producer_conv_type": group["producer_conv_type"],
                    "base_channels": group["base_channels"],
                    "min_channels": group["min_channels"],
                    "max_channels": group["max_channels"],
                    "divisibility": group["divisibility"],
                }
                for group in self.groups
            ],
            "layer_to_group": self.layer_to_group,
        }

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _find_prunable_layers(self) -> List[Dict[str, Any]]:
        layers: List[Dict[str, Any]] = []
        for node in self.dg.nodes:
            if (
                node.get("is_module")
                and node["type"] in PRUNABLE_TYPES
                and node["name"] in self.name2module
            ):
                module = self.name2module[node["name"]]
                layers.append(
                    {
                        "name": node["name"],
                        "type": node["type"],
                        "conv_type": node.get("conv_type"),
                        "node_id": node["id"],
                        "out_channels": getattr(
                            module,
                            "out_channels",
                            getattr(module, "out_features", None),
                        ),
                        "in_channels": getattr(
                            module, "in_channels", getattr(module, "in_features", None)
                        ),
                        "groups": getattr(module, "groups", 1),
                        "flow_channels": flow_out_channels(
                            module, node.get("conv_type")
                        ),
                    }
                )
        return layers

    def _find_mirror_groups(
        self, prunable_layers: List[Dict[str, Any]]
    ) -> List[Set[str]]:
        parent = {layer["name"]: layer["name"] for layer in prunable_layers}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: str, y: str) -> None:
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_x] = root_y

        mirror_ops = {"add", "add_", "mul", "mul_"}

        for node in self.dg.nodes:
            if node.get("is_operation") and node["type"] in mirror_ops:
                producers = self._traversal.trace_all_producers(node["id"])
                if len(producers) > 1:
                    base = producers[0]
                    for producer in producers[1:]:
                        union(base, producer)

        for layer in prunable_layers:
            if layer["conv_type"] != "depthwise":
                continue
            if layer["name"] not in self.name2node:
                continue
            node = self.name2node[layer["name"]]
            producers = self._traversal.trace_producers_for_inputs(node["id"])
            for producer in producers:
                union(layer["name"], producer)

        grouped: Dict[str, Set[str]] = {}
        for layer in prunable_layers:
            grouped.setdefault(find(layer["name"]), set()).add(layer["name"])
        return list(grouped.values())

    def _build_groups(self, mirror_groups: List[Set[str]]) -> None:
        for gid, member_names in enumerate(mirror_groups):
            members = [
                self.name2node[name] for name in member_names if name in self.name2node
            ]
            if not members:
                continue

            producer_name = pick_canonical_member(sorted(member_names), self.name2node)
            producer_node = self.name2node[producer_name]
            producer_module = self.name2module[producer_name]

            producer_conv_type = producer_node.get("conv_type")
            base_channels = flow_out_channels(producer_module, producer_conv_type)
            if base_channels is None:
                continue

            all_consumers: List[Dict[str, Any]] = []
            divisibility = 1
            has_depthwise_consumer = False
            has_concat_consumer = False

            for member in members:
                consumers, div, has_depthwise, has_concat = (
                    self._traversal.analyze_consumers(member["id"])
                )
                all_consumers.extend(consumers)
                divisibility = lcm(divisibility, div)
                has_depthwise_consumer = has_depthwise_consumer or has_depthwise
                has_concat_consumer = has_concat_consumer or has_concat

            unique_consumers: Dict[str, Dict[str, Any]] = {}
            for consumer in all_consumers:
                name = consumer["name"]
                if name in unique_consumers:
                    if consumer.get("needs_concat"):
                        unique_consumers[name]["needs_concat"] = True
                    continue
                unique_consumers[name] = dict(consumer)

            min_channels = calculate_min_channels(
                int(base_channels), int(divisibility), has_depthwise_consumer
            )
            min_channels = min(min_channels, int(base_channels))

            group = {
                "group_id": gid,
                "members": sorted(member_names),
                "producer": producer_name,
                "producer_type": producer_node["type"],
                "producer_conv_type": producer_conv_type,
                "base_channels": int(base_channels),
                "min_channels": int(min_channels),
                "max_channels": int(base_channels),
                "divisibility": int(divisibility),
                "has_depthwise_consumer": bool(has_depthwise_consumer),
                "has_cat_consumer": bool(has_concat_consumer),
                "consumers": list(unique_consumers.values()),
                "action_space": create_action_space(
                    int(min_channels), int(base_channels), int(divisibility)
                ),
            }

            self.groups.append(group)
            for name in member_names:
                self.layer_to_group[name] = gid

    def _build_residual_components(self) -> None:
        if not self.groups:
            self.residual_components = []
            return

        parent: Dict[int, int] = {
            group["group_id"]: group["group_id"] for group in self.groups
        }

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_a] = root_b

        coupling_ops = {"add", "add_", "mul", "mul_"}

        for node in self.dg.nodes:
            if not node.get("is_operation") or node["type"] not in coupling_ops:
                continue
            producer_names = self._traversal.trace_all_producers(node["id"])
            group_ids = [
                self.layer_to_group[name]
                for name in producer_names
                if name in self.layer_to_group
            ]
            if len(group_ids) <= 1:
                continue
            base = group_ids[0]
            for gid in group_ids[1:]:
                union(base, gid)

        components: Dict[int, Set[int]] = defaultdict(set)
        for gid in parent:
            components[find(gid)].add(gid)
        self.residual_components = [
            component for component in components.values() if len(component) > 1
        ]

    def _count_total_params(self) -> int:
        total = 0
        for group in self.groups:
            for name in group["members"]:
                module = self.name2module.get(name)
                if module is not None:
                    total += sum(param.numel() for param in module.parameters())
        return total
