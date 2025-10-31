"""Pure helper functions used by the pruning group builder."""

from __future__ import annotations

from typing import Dict, List, Sequence


def pick_canonical_member(
    member_names: Sequence[str], name2node: Dict[str, Dict[str, object]]
) -> str:
    """Pick a representative member to serve as the producer baseline."""
    candidates = []
    for name in member_names:
        node = name2node[name]
        conv_type = node.get("conv_type")
        node_type = node["type"]
        rank = (
            0
            if (
                node_type.startswith("Conv")
                and conv_type in (None, "standard", "pointwise")
            )
            else 1
            if (node_type.startswith("Conv") and conv_type == "grouped")
            else 2
            if (node_type.startswith("Conv") and conv_type == "depthwise")
            else 3
        )
        candidates.append((rank, name))
    candidates.sort()
    return candidates[0][1]


def calculate_min_channels(
    base_channels: int, divisibility: int, has_depthwise_consumer: bool
) -> int:
    """Derive a conservative lower bound on channels for a group."""
    min_channels = max(1, divisibility)
    if has_depthwise_consumer:
        min_channels = max(min_channels, 8)
    min_channels = max(min_channels, max(1, int(base_channels * 0.10)))
    if divisibility > 1 and (min_channels % divisibility != 0):
        min_channels = ((min_channels // divisibility) + 1) * divisibility
    return min(min_channels, base_channels)


def create_action_space(
    min_channels: int, max_channels: int, divisibility: int
) -> List[int]:
    """Enumerate admissible channel counts respecting divisibility."""
    if max_channels < min_channels:
        return [min_channels]
    if divisibility <= 1:
        return list(range(min_channels, max_channels + 1))
    values: List[int] = []
    current = ((min_channels + divisibility - 1) // divisibility) * divisibility
    while current <= max_channels:
        values.append(current)
        current += divisibility
    return values or [min_channels]
