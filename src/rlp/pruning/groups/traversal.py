"""Graph traversal helpers for building pruning groups."""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Set, Tuple

from .common import PASS_THROUGH_TYPES, PRUNABLE_TYPES, lcm


class GraphTraversal:
    """Encapsulates dependency-graph walks used during group construction."""

    def __init__(
        self,
        id2node: Dict[int, Dict[str, object]],
        name2node: Dict[str, Dict[str, object]],
        name2module: Dict[str, object],
    ) -> None:
        self.id2node = id2node
        self.name2node = name2node
        self.name2module = name2module
        self._bad_node_warned = False

    def trace_all_producers(self, start_id: int) -> List[str]:
        producers: List[str] = []
        visited: Set[int] = set()
        queue: deque[int] = deque([start_id])

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            node = self.id2node[current]
            if node.get("is_module") and node["name"] in self.name2node:
                node_type = self.name2node[node["name"]]["type"]
                if node_type in PRUNABLE_TYPES:
                    producers.append(node["name"])
                    continue
            for parent_id in node.get("inputs", []):
                queue.append(parent_id)
        return producers

    def trace_producers_for_inputs(self, consumer_id: int) -> List[str]:
        consumer = self.id2node[consumer_id]
        producers: List[str] = []
        for input_id in consumer.get("inputs", []):
            producers.extend(self.trace_all_producers(input_id))
        return sorted(set(producers))

    def analyze_consumers(
        self, producer_id: int
    ) -> Tuple[List[Dict[str, object]], int, bool, bool]:
        consumers: List[Dict[str, object]] = []
        divisibility = 1
        has_depthwise = False
        has_concat = False

        visited: Set[Tuple[int, bool]] = set()
        queue: deque[Tuple[int, bool]] = deque([(producer_id, False)])

        while queue:
            current, after_concat = queue.popleft()
            if (current, after_concat) in visited:
                continue
            visited.add((current, after_concat))
            node = self.id2node[current]

            for out_id in node.get("outputs", []):
                successor = self.id2node[out_id]

                if successor.get("is_module"):
                    module = self.name2module.get(successor["name"])
                    if module is None:
                        queue.append((out_id, after_concat))
                        continue

                    succ_type = successor["type"]
                    if succ_type in PRUNABLE_TYPES:
                        conv_type = successor.get("conv_type")
                        groups = getattr(module, "groups", 1)
                        consumers.append(
                            {
                                "name": successor["name"],
                                "type": succ_type,
                                "conv_type": conv_type,
                                "groups": groups,
                                "needs_concat": bool(after_concat),
                            }
                        )
                        if conv_type == "depthwise":
                            has_depthwise = True
                        if conv_type == "grouped" or groups > 1:
                            divisibility = lcm(divisibility, groups)
                        continue

                    if succ_type in PASS_THROUGH_TYPES:
                        queue.append((out_id, after_concat))
                    else:
                        queue.append((out_id, after_concat))
                    continue

                if successor.get("is_operation"):
                    op_type = successor["type"]
                    if op_type in ("add", "add_"):
                        for child_id in successor.get("outputs", []):
                            queue.append((child_id, after_concat))
                        continue
                    if op_type in ("cat", "concat", "stack"):
                        has_concat = True
                        for child_id in successor.get("outputs", []):
                            queue.append((child_id, True))
                        continue
                    for child_id in successor.get("outputs", []):
                        child = self.id2node[child_id]
                        if child.get("is_module"):
                            module = self.name2module.get(child["name"])
                            if module is not None and child["type"] in PRUNABLE_TYPES:
                                conv_type = child.get("conv_type")
                                groups = getattr(module, "groups", 1)
                                consumers.append(
                                    {
                                        "name": child["name"],
                                        "type": child["type"],
                                        "conv_type": conv_type,
                                        "groups": groups,
                                        "needs_concat": bool(after_concat),
                                    }
                                )
                                if conv_type == "depthwise":
                                    has_depthwise = True
                                if conv_type == "grouped":
                                    divisibility = lcm(divisibility, groups)
                                continue
                        queue.append((child_id, after_concat))
                    continue

                if not (successor.get("is_module") or successor.get("is_operation")):
                    if not self._bad_node_warned:
                        print(
                            "[group_creator] warning: node without is_module/is_operation; skipping"
                        )
                        self._bad_node_warned = True

        return consumers, divisibility, has_depthwise, has_concat
