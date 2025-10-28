# pruning_helper/group_creator.py
import torch
import torch.nn as nn
from typing import List, Dict, Set, Tuple, Optional, Any
from collections import deque

def _flow_out_channels(m: nn.Module, conv_type: Optional[str]) -> Optional[int]:
    if isinstance(m, nn.Linear): return m.out_features
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        if conv_type == 'depthwise': return m.in_channels
        return m.out_channels
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return m.out_channels
    return None

class PruningGroups:
    def __init__(self, dg, model: nn.Module):
        self.dg = dg
        self.model = model
        self.id2node = {n['id']: n for n in dg.nodes}
        self.name2node = {n['name']: n for n in dg.nodes if n.get('is_module')}
        self.name2module = dict(model.named_modules())
        self.groups: List[Dict] = []
        self.layer_to_group: Dict[str, int] = {}

    def build(self) -> List[Dict]:
        prunable_layers = self._find_prunable_layers()
        mirror_groups = self._find_mirror_groups(prunable_layers)
        self._build_groups(mirror_groups)
        return self.groups

    def _find_prunable_layers(self) -> List[Dict]:
        prunable_types = {'Conv1d','Conv2d','Conv3d','ConvTranspose1d','ConvTranspose2d','ConvTranspose3d','Linear'}
        layers = []
        for node in self.dg.nodes:
            if node.get('is_module') and node['type'] in prunable_types and node['name'] in self.name2module:
                module = self.name2module[node['name']]
                layers.append({
                    'name': node['name'],
                    'type': node['type'],
                    'conv_type': node.get('conv_type'),
                    'node_id': node['id'],
                    'out_channels': getattr(module, 'out_channels', getattr(module, 'out_features', None)),
                    'in_channels': getattr(module, 'in_channels', getattr(module, 'in_features', None)),
                    'groups': getattr(module, 'groups', 1),
                    'flow_channels': _flow_out_channels(module, node.get('conv_type')),
                })
        return layers

    def _find_mirror_groups(self, prunable_layers: List[Dict]) -> List[Set[str]]:
        parent = {layer['name']: layer['name'] for layer in prunable_layers}
        def find(x):
            if parent[x] != x: parent[x] = find(parent[x])
            return parent[x]
        def union(x, y):
            px, py = find(x), find(y)
            if px != py: parent[px] = py

        for node in self.dg.nodes:
            if node.get('is_operation') and node['type'] in ('add','add_'):
                producers = self._trace_all_producers(node['id'])
                if len(producers) > 1:
                    base = producers[0]
                    for p in producers[1:]:
                        union(base, p)

        for layer in prunable_layers:
            if layer['conv_type'] == 'depthwise':
                if layer['name'] not in self.name2node: continue
                node = self.name2node[layer['name']]
                producers = self._trace_producers_for_inputs(node['id'])
                for p in producers:
                    union(layer['name'], p)

        groups_dict: Dict[str, Set[str]] = {}
        for layer in prunable_layers:
            root = find(layer['name'])
            groups_dict.setdefault(root, set()).add(layer['name'])
        return list(groups_dict.values())

    def _trace_all_producers(self, start_id: int) -> List[str]:
        producers = []
        vis = set()
        q = deque([start_id])
        while q:
            cur = q.popleft()
            if cur in vis: continue
            vis.add(cur)
            node = self.id2node[cur]
            if node.get('is_module') and node['name'] in self.name2node:
                t = self.name2node[node['name']]['type']
                if t in {'Conv1d','Conv2d','Conv3d','ConvTranspose1d','ConvTranspose2d','ConvTranspose3d','Linear'}:
                    producers.append(node['name'])
                    continue
            for pid in node['inputs']:
                q.append(pid)
        return producers

    def _trace_producers_for_inputs(self, consumer_id: int) -> List[str]:
        consumer = self.id2node[consumer_id]
        out = []
        for in_id in consumer['inputs']:
            out.extend(self._trace_all_producers(in_id))
        return sorted(set(out))

    def _pick_canonical_member(self, member_names: List[str]) -> str:
        cand = []
        for n in member_names:
            node = self.name2node[n]
            ct = node.get('conv_type')
            t  = node['type']
            rank = 0 if (t.startswith('Conv') and (ct in (None,'standard','pointwise'))) else \
                   1 if (t.startswith('Conv') and ct=='grouped') else \
                   2 if (t.startswith('Conv') and ct=='depthwise') else \
                   3
            cand.append((rank, n))
        cand.sort()
        return cand[0][1]

    def _build_groups(self, mirror_groups: List[Set[str]]):
        for gid, member_names in enumerate(mirror_groups):
            members = [self.name2node[n] for n in member_names if n in self.name2node]
            if not members: continue
            first_name = self._pick_canonical_member(list(member_names))
            first_node = self.name2node[first_name]
            first_mod  = self.name2module[first_name]
            p_ct       = first_node.get('conv_type')
            base_channels = _flow_out_channels(first_mod, p_ct)
            if base_channels is None: continue

            all_consumers = []
            divisibility = 1
            has_depthwise_consumer = False
            has_cat_consumer = False
            for m in members:
                cons, div, has_dw, has_cat = self._analyze_consumers(m['id'])
                all_consumers.extend(cons)
                divisibility = self._lcm(divisibility, div)
                has_depthwise_consumer = has_depthwise_consumer or has_dw
                has_cat_consumer = has_cat_consumer or has_cat

            seen = set(); uniq_consumers = []
            for c in all_consumers:
                k = (c['name'], c.get('needs_concat', False))
                if k in seen: continue
                uniq_consumers.append(c); seen.add(k)

            min_channels = self._calculate_min_channels(int(base_channels), divisibility, has_depthwise_consumer)
            min_channels = min(min_channels, int(base_channels))
            max_channels = int(base_channels)

            group = {
                'group_id': gid,
                'members': sorted(list(member_names)),
                'producer': first_name,
                'producer_type': first_node['type'],
                'producer_conv_type': p_ct,
                'base_channels': int(base_channels),
                'min_channels': int(min_channels),
                'max_channels': int(max_channels),
                'divisibility': int(divisibility),
                'has_depthwise_consumer': bool(has_depthwise_consumer),
                'has_cat_consumer': bool(has_cat_consumer),
                'consumers': uniq_consumers,
                'action_space': self._create_action_space(int(min_channels), int(max_channels), int(divisibility)),
            }
            self.groups.append(group)
            for name in member_names:
                self.layer_to_group[name] = gid

    def _analyze_consumers(self, producer_id: int) -> Tuple[List[Dict], int, bool, bool]:
        consumers = []
        divisibility = 1
        has_depthwise = False
        has_cat = False
        vis = set()
        q = deque([(producer_id, False)])
        while q:
            cur, after_concat = q.popleft()
            if (cur, after_concat) in vis: continue
            vis.add((cur, after_concat))
            node = self.id2node[cur]

            for out_id in node['outputs']:
                v = self.id2node[out_id]

                if v.get('is_module'):
                    m = self.name2module.get(v['name'])
                    if m is None:
                        q.append((out_id, after_concat))
                        continue

                    if v['type'] in {'Conv1d','Conv2d','Conv3d','ConvTranspose1d','ConvTranspose2d','ConvTranspose3d','Linear'}:
                        ct = v.get('conv_type'); g = getattr(m,'groups',1)
                        consumers.append({
                            'name': v['name'],
                            'type': v['type'],
                            'conv_type': ct,
                            'groups': g,
                            'needs_concat': bool(after_concat)
                        })
                        if ct == 'depthwise': has_depthwise = True
                        if ct == 'grouped' or g > 1: divisibility = self._lcm(divisibility, g)
                        continue

                    pass_through = {'BatchNorm1d','BatchNorm2d','BatchNorm3d','ReLU','ReLU6','SiLU','GELU','Identity','Dropout','MaxPool2d','AvgPool2d','AdaptiveAvgPool2d'}
# Ensure your op traversal also treats 'view', 'reshape', 'flatten', 'permute', 'transpose' as pass-through ops
# so BFS continues until it meets the Linear consumer.

                    if v['type'] in pass_through:
                        q.append((out_id, after_concat))
                    else:
                        q.append((out_id, after_concat))
                    continue

                if v.get('is_operation'):
                    if v['type'] in ('add','add_'):
                        for w_id in v['outputs']:
                            q.append((w_id, after_concat))
                        continue
                    if v['type'] in ('cat','concat','stack'):
                        has_cat = True
                        for w_id in v['outputs']:
                            q.append((w_id, True))
                        continue
                    for w_id in v['outputs']:
                        q.append((w_id, after_concat))
                    continue

                if not (v.get('is_module') or v.get('is_operation')):
                    if not hasattr(self, "_bad_node_warned"):
                        print("[group_creator] warning: node without is_module/is_operation; skipping")
                        self._bad_node_warned = True
                    continue
        return consumers, divisibility, has_depthwise, has_cat

    def _calculate_min_channels(self, base_channels: int, divisibility: int, has_depthwise: bool) -> int:
        min_ch = max(1, divisibility)
        if has_depthwise: min_ch = max(min_ch, 8)
        min_ch = max(min_ch, max(1, int(base_channels * 0.10)))
        if divisibility > 1 and (min_ch % divisibility != 0):
            min_ch = ((min_ch // divisibility) + 1) * divisibility
        return min(min_ch, base_channels)

    def _create_action_space(self, min_ch: int, max_ch: int, divisibility: int) -> List[int]:
        if max_ch < min_ch: return [min_ch]
        if divisibility <= 1:
            return list(range(min_ch, max_ch + 1))
        out = []
        c = ((min_ch + divisibility - 1) // divisibility) * divisibility
        while c <= max_ch:
            out.append(c)
            c += divisibility
        return out or [min_ch]

    def _lcm(self, a: int, b: int) -> int:
        from math import gcd
        return abs(a*b)//gcd(a,b) if a and b else max(a,b)

    def print_groups(self, verbose: bool = False):
        print("\n" + "="*80)
        print(f"Pruning Groups: {len(self.groups)} groups")
        print("="*80)
        for g in self.groups:
            print(f"\n[Group {g['group_id']}] {len(g['members'])} layers")
            print(f"  Producer: {g['producer']} ({g['producer_type']}, {g['producer_conv_type']})")
            print(f"  Base channels: {g['base_channels']}")
            print(f"  Range: [{g['min_channels']}, {g['max_channels']}]  Div: {g['divisibility']}")
            print(f"  Action space: {len(g['action_space'])} choices")
            if verbose:
                print(f"  Members: {g['members']}")
                if g['consumers']:
                    print(f"  Consumers ({len(g['consumers'])}):")
                    for c in g['consumers'][:6]:
                        tag = "after_concat" if c.get('needs_concat') else "direct"
                        print(f"    - {c['name']} ({c['type']}, {c['conv_type']}, {tag})")
                    if len(g['consumers']) > 6:
                        print(f"    ... and {len(g['consumers'])-6} more")
        print("\n" + "="*80)
        print(f"Total prunable params: {self._count_total_params()}")
        print("="*80 + "\n")

    def _count_total_params(self) -> int:
        total = 0
        for g in self.groups:
            for n in g['members']:
                if n in self.name2module:
                    total += sum(p.numel() for p in self.name2module[n].parameters())
        return total

    def export_for_rl(self) -> Dict[str, Any]:
        return {
            'num_groups': len(self.groups),
            'action_spaces': [g['action_space'] for g in self.groups],
            'group_info': [
                {
                    'group_id': g['group_id'],
                    'members': g['members'],
                    'producer': g['producer'],
                    'producer_type': g['producer_type'],
                    'producer_conv_type': g['producer_conv_type'],
                    'base_channels': g['base_channels'],
                    'min_channels': g['min_channels'],
                    'max_channels': g['max_channels'],
                    'divisibility': g['divisibility'],
                } for g in self.groups
            ],
            'layer_to_group': self.layer_to_group,
        }

if __name__ == "__main__":
    class ResNetBlock(nn.Module):
        def __init__(self, c=64):
            super().__init__()
            self.conv1 = nn.Conv2d(c,c,3,padding=1)
            self.bn1 = nn.BatchNorm2d(c)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(c,c,3,padding=1)
            self.bn2 = nn.BatchNorm2d(c)
        def forward(self, x):
            y = self.relu(self.bn1(self.conv1(x)))
            y = self.bn2(self.conv2(y))
            y = y + x
            return self.relu(y)

    class MobileNetBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.dw = nn.Conv2d(32,32,3,padding=1,groups=32)
            self.bn1 = nn.BatchNorm2d(32)
            self.pw = nn.Conv2d(32,64,1)
            self.bn2 = nn.BatchNorm2d(64)
        def forward(self, x):
            x = self.bn1(self.dw(x))
            x = self.bn2(self.pw(x))
            return x

    #from pruning_helper.depgraph_min import DependencyGraph

    m1 = ResNetBlock().eval()
    x1 = torch.randn(1,64,32,32)
    g1 = DependencyGraph().build(m1, x1)
    pg1 = PruningGroups(g1, m1); pg1.build(); pg1.print_groups(verbose=True)

    m2 = MobileNetBlock().eval()
    x2 = torch.randn(1,32,32,32)
    g2 = DependencyGraph().build(m2, x2)
    pg2 = PruningGroups(g2, m2); pg2.build(); pg2.print_groups(verbose=True)
