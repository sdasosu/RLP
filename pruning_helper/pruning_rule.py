# pruning_helper/pruning_rule.py
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Set, Tuple

#========================
# Simple rankers
#========================
def _l2_scores_conv2d(m: nn.Conv2d, conv_type: Optional[str]) -> torch.Tensor:
    w = m.weight.data
    return w.view(w.size(0), -1).pow(2).sum(dim=1)

def _l2_scores_linear(m: nn.Linear) -> torch.Tensor:
    w = m.weight.data
    return w.pow(2).sum(dim=1)

def _select_keep_idx(scores: torch.Tensor, k: int) -> List[int]:
    k = max(1, min(k, scores.numel()))
    _, idx = torch.topk(scores, k, largest=True, sorted=True)
    idx, _ = torch.sort(idx)
    return idx.tolist()

#========================
# Tiny helpers
#========================
def _infer_conv_type_from_module(m: nn.Module) -> Optional[str]:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        g = getattr(m, 'groups', 1)
        if g == m.in_channels == m.out_channels and g > 1: return 'depthwise'
        ks = m.kernel_size if isinstance(m.kernel_size, tuple) else (m.kernel_size,)
        if all(k == 1 for k in ks): return 'pointwise'
        if g > 1: return 'grouped'
        return 'standard'
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return 'transpose'
    return None

def _get_out_channels(m: nn.Module) -> Optional[int]:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)): return m.out_channels
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)): return m.out_channels
    if isinstance(m, nn.Linear): return m.out_features
    return None

def _sanitize_idx(idx: List[int], upper: int) -> List[int]:
    v = [i for i in idx if 0 <= i < upper]
    if not v:
        k = min(upper, max(1, len(idx)))
        v = list(range(k))
    return v

#========================
# Output slicing
#========================
def _slice_conv_out(m: nn.Conv2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(_sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    m.out_channels = int(keep.numel())

def _slice_dwconv(m: nn.Conv2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(_sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    c = int(keep.numel())
    m.in_channels = c
    m.out_channels = c
    m.groups = c

def _slice_convT_out(m: nn.ConvTranspose2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(_sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.out_channels = int(keep.numel())

def _slice_linear_out(m: nn.Linear, keep_idx: List[int]) -> None:
    # No-op: we rarely prune classifier outputs here; handled by consumer slicing
    return

#-------------------------------------------------------------------------------------
def _slice_bn_out(m: nn.Module, keep_idx: List[int]) -> None:
    if not isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)): return
    c = m.num_features
    keep = torch.tensor(_sanitize_idx(keep_idx, c), dtype=torch.long)
    if hasattr(m, 'weight') and m.weight is not None:
        m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
    if hasattr(m, 'bias') and m.bias is not None:
        m.bias = nn.Parameter(m.bias.data.index_select(0, keep).contiguous())
    if hasattr(m, 'running_mean') and m.running_mean is not None:
        m.running_mean = m.running_mean.data.index_select(0, keep).contiguous()
    if hasattr(m, 'running_var') and m.running_var is not None:
        m.running_var = m.running_var.data.index_select(0, keep).contiguous()
    m.num_features = int(keep.numel())
#-----------------------------------------------------------------------------------

#-----------------------------------------------------------------------------------
def _adjust_bn_chain(dg, name2mod, id2node, start_name: str, keep_idx: List[int]) -> None:
    name2id = {n['name']: n['id'] for n in dg.nodes if n.get('is_module')}
    if start_name not in name2id: return
    start = name2id[start_name]
    vis: Set[int] = set()
    q: List[int] = [start]
    while q:
        u = q.pop(0)
        if u in vis: continue
        vis.add(u)
        v = id2node[u]
        for out_id in v.get('outputs', []):
            w = id2node[out_id]
            if w.get('is_module'):
                mm = name2mod.get(w['name'])
                if isinstance(mm, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    _slice_bn_out(mm, keep_idx)
                    q.append(out_id); continue
                if isinstance(mm, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.GELU, nn.Identity, nn.Dropout,
                                   nn.MaxPool2d, nn.AvgPool2d, nn.AdaptiveAvgPool2d)):
                    q.append(out_id); continue
            else:
                q.append(out_id)
#-----------------------------------------------------------------------------------

#========================
# Input slicing
#========================
def _slice_conv_in_grouped(m: nn.Conv2d, keep_idx: List[int]) -> None:
    keep = torch.tensor(_sanitize_idx(keep_idx, m.in_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.in_channels = int(keep.numel())

def _slice_conv_in(m: nn.Conv2d, keep_idx: List[int]) -> None:
    # depthwise must not arrive here
    assert not (m.groups == m.in_channels == m.out_channels and m.groups > 1), \
        "Depthwise conv reached _slice_conv_in; handle via _slice_dwconv"
    if m.groups > 1:
        _slice_conv_in_grouped(m, keep_idx); return
    keep = torch.tensor(_sanitize_idx(keep_idx, m.in_channels), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.in_channels = int(keep.numel())

def _slice_linear_in(m: nn.Linear, keep_idx: List[int]) -> None:
    keep = torch.tensor(_sanitize_idx(keep_idx, m.in_features), dtype=torch.long)
    m.weight = nn.Parameter(m.weight.data.index_select(1, keep).contiguous())
    m.in_features = int(keep.numel())

def _slice_consumer_in(m: nn.Module, keep_idx: List[int], conv_type: Optional[str]) -> None:
    if isinstance(m, nn.Linear):
        _slice_linear_in(m, keep_idx); return
    if isinstance(m, nn.ConvTranspose2d):
        keep = torch.tensor(_sanitize_idx(keep_idx, m.out_channels), dtype=torch.long)
        m.weight = nn.Parameter(m.weight.data.index_select(0, keep).contiguous())
        m.out_channels = int(keep.numel()); return
    if isinstance(m, nn.Conv2d):
        if conv_type is None: conv_type = _infer_conv_type_from_module(m)
        if conv_type == 'depthwise' or (m.groups == m.in_channels == m.out_channels and m.groups > 1):
            _slice_dwconv(m, keep_idx); return
        if m.groups > 1:
            _slice_conv_in_grouped(m, keep_idx); return
        _slice_conv_in(m, keep_idx); return

#========================
# Concat helpers
#========================
def _concat_sources(dg, target_name: str, id2node: Dict[int, Dict[str, Any]]) -> List[Tuple[int, str]]:
    name2id = {n['name']: n['id'] for n in dg.nodes if n.get('is_module')}
    if target_name not in name2id: return []
    start = name2id[target_name]
    vis: Set[int] = set()
    q: List[int] = [start]
    outs: List[Tuple[int, str]] = []
    while q:
        u = q.pop(0)
        if u in vis: continue
        vis.add(u)
        v = id2node[u]
        for in_id in v.get('inputs', []):
            w = id2node[in_id]
            if w.get('is_operation') and w['type'] in ('cat', 'concat', 'stack'):
                for src_id in w.get('inputs', []):
                    s = id2node[src_id]
                    if s.get('is_module'):
                        outs.append((s['id'], s['name']))
                    else:
                        q.append(src_id)
                continue
            if w.get('is_module'):
                outs.append((w['id'], w['name']))
            else:
                q.append(in_id)
    return outs

def _compute_concat_indices(dg, pg, consumer_name: str, keep_map: Dict[int, List[int]],
                            name2mod: Dict[str, nn.Module], id2node: Dict[int, Dict[str, Any]]) -> List[int]:
    upstream = _concat_sources(dg, consumer_name, id2node)
    name2gid = pg.layer_to_group
    full: List[int] = []
    offset = 0
    for _, name in upstream:
        gid = name2gid.get(name, -1)
        if gid == -1:
            mod = name2mod.get(name, None)
            if mod is None:
                raise RuntimeError(f"concat source '{name}' module not found")
            ch = _get_out_channels(mod)
            if ch is None:
                raise RuntimeError(f"concat source '{name}' channels unknown")
            full.extend(range(offset, offset + ch))
            offset += ch
        else:
            kidx = keep_map.get(gid, [])
            full.extend([offset + i for i in kidx])
            offset += len(kidx)
    return full

#========================
# Fallback: nearest producer (handles flatten/view→fc)
#========================
_PASS_THRU_TYPES = {
    'view', 'reshape', 'flatten', 'permute', 'transpose',
    'relu', 'relu_', 'gelu', 'silu', 'sigmoid', 'tanh', 'softmax',
    'avg_pool2d', 'max_pool2d', 'interpolate', 'identity', 'add', 'add_'
}

def _nearest_producer_name(pg, id2node: Dict[int, Dict[str, Any]], start_name: str) -> Optional[str]:
    layer2id = {n['name']: n['id'] for n in id2node.values() if n.get('is_module')}
    if start_name not in layer2id: return None
    start = layer2id[start_name]
    vis: Set[int] = set()
    q: List[int] = [start]
    while q:
        u = q.pop(0)
        if u in vis: continue
        vis.add(u)
        v = id2node[u]
        for in_id in v.get('inputs', []):
            w = id2node[in_id]
            if w.get('is_module'):
                nm = w['name']
                if nm in pg.layer_to_group:
                    return nm
                q.append(in_id)
            else:
                if w['type'] in _PASS_THRU_TYPES:
                    q.append(in_id)
    return None

#========================
# SE expand detector (MobileNetV3): does this conv feed a 'mul' scale op?
#========================
def _is_se_expand(dg, start_name: str, id2node: Dict[int, Dict[str, Any]]) -> bool:
    name2id = {n['name']: n['id'] for n in dg.nodes if n.get('is_module')}
    if start_name not in name2id:
        return False
    start = name2id[start_name]
    vis: Set[int] = set()
    q: List[int] = [start]
    steps = 0
    while q and steps < 64:
        u = q.pop(0); steps += 1
        if u in vis: 
            continue
        vis.add(u)
        v = id2node[u]
        for out_id in v.get('outputs', []):
            w = id2node[out_id]
            if w.get('is_operation'):
                if w['type'] in ('mul', '*'):
                    return True
                q.append(out_id)
                continue
            if w.get('is_module'):
                t = w['type']
                if t in ('Hardsigmoid','Sigmoid','ReLU','ReLU6','SiLU','GELU','Identity',
                         'AdaptiveAvgPool1d','AdaptiveAvgPool2d','AdaptiveAvgPool3d',
                         'AvgPool1d','AvgPool2d','AvgPool3d'):
                    q.append(out_id)
                # stop on other heavy modules
    return False

#========================
# Main API
#========================
def apply_pruning(model: nn.Module, dg, pg, decisions: Dict[int, int]) -> Dict[int, List[int]]:
    name2mod = dict(model.named_modules())
    id2node = {n['id']: n for n in dg.nodes}
    keep_map: Dict[int, List[int]] = {}

    # Phase 1: prune producers (and mirror members) on OUTPUT + adjust BN after each
    for g in pg.groups:
        gid = g['group_id']
        k = int(decisions.get(gid, g['max_channels']))
        if g['divisibility'] > 1:
            k = (k // g['divisibility']) * g['divisibility']
        k = max(g['min_channels'], min(k, g['max_channels']))
        if g['divisibility'] > 1 and (k % g['divisibility'] != 0):
            k = ((k // g['divisibility']) + 1) * g['divisibility']
            k = min(k, g['max_channels'])

        prod_name = g['producer']
        pm = name2mod.get(prod_name, None)
        if pm is None:
            continue

        if isinstance(pm, nn.Linear):
            scores = _l2_scores_linear(pm)
        else:
            scores = _l2_scores_conv2d(pm, g.get('producer_conv_type'))
        keep_idx = _select_keep_idx(scores, k)
        keep_map[gid] = keep_idx

        if isinstance(pm, nn.Linear):
            _slice_linear_out(pm, keep_idx)
        elif isinstance(pm, nn.ConvTranspose2d):
            _slice_convT_out(pm, keep_idx)
        elif isinstance(pm, nn.Conv2d) and g.get('producer_conv_type') == 'depthwise':
            _slice_dwconv(pm, keep_idx)
        elif isinstance(pm, nn.Conv2d):
            _slice_conv_out(pm, keep_idx)

        _adjust_bn_chain(dg, name2mod, id2node, prod_name, keep_idx)

        for member in g['members']:
            if member == prod_name: continue
            mm = name2mod.get(member)
            if mm is None: continue
            m_ct = _infer_conv_type_from_module(mm) if isinstance(mm, nn.Conv2d) else None
            if isinstance(mm, nn.Linear):
                _slice_linear_out(mm, keep_idx)
            elif isinstance(mm, nn.ConvTranspose2d):
                _slice_convT_out(mm, keep_idx)
            elif isinstance(mm, nn.Conv2d) and (m_ct == 'depthwise'):
                _slice_dwconv(mm, keep_idx)
            elif isinstance(mm, nn.Conv2d):
                _slice_conv_out(mm, keep_idx)
            _adjust_bn_chain(dg, name2mod, id2node, member, keep_idx)

    # Phase 2: adjust consumers' INPUTS (includes FC and depthwise/pointwise chains)
    processed_inputs: Set[str] = set()
    for g in pg.groups:
        gid = g['group_id']
        keep_idx = keep_map[gid]
        for c in g['consumers']:
            cname = c['name']
            if cname in processed_inputs:
                continue
            cm = name2mod.get(cname)
            if cm is None:
                continue

            # Detect MobileNetV3 SE expand 1x1 conv: slice its OUT channels instead of input
            is_se_expand = False
            if isinstance(cm, nn.Conv2d) and (c.get('conv_type') == 'pointwise'):
                is_se_expand = _is_se_expand(dg, cname, id2node)

            if c.get('needs_concat'):
                full_idx = _compute_concat_indices(dg, pg, cname, keep_map, name2mod, id2node)
                if is_se_expand:
                    _slice_conv_out(cm, full_idx)  # align scale channels to concat result
                else:
                    _slice_consumer_in(cm, full_idx, c.get('conv_type'))
            else:
                if is_se_expand:
                    _slice_conv_out(cm, keep_idx)  # scale's OUT must match producer
                else:
                    _slice_consumer_in(cm, keep_idx, c.get('conv_type'))

            processed_inputs.add(cname)

    # Phase 2B: guard for Linear after flatten/view (ResNet/DenseNet/GoogLeNet heads)
    for lname, lmod in name2mod.items():
        if not isinstance(lmod, nn.Linear):
            continue
        if lname in processed_inputs:
            continue
        # Try concat path first
        try:
            srcs = _concat_sources(dg, lname, id2node)
            if srcs:
                full_idx = _compute_concat_indices(dg, pg, lname, keep_map, name2mod, id2node)
                _slice_linear_in(lmod, full_idx)
                processed_inputs.add(lname)
                continue
        except Exception:
            pass
        # Else: walk back to nearest prunable producer (across flatten/view/etc.)
        prod = _nearest_producer_name(pg, id2node, lname)
        if prod is None:
            continue
        gid2 = pg.layer_to_group.get(prod, -1)
        if gid2 == -1:
            continue
        _slice_linear_in(lmod, keep_map[gid2])
        processed_inputs.add(lname)

    return keep_map
