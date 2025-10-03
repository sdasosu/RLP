#====================== table_generator =======================
import torch                                                  #
import torch.nn as nn                                         #
import torch.fx as fx                                         #
from typing import Any, Dict, List, Tuple, Set                #
#==============================================================



#============ Identification (Channel Alterer)=================
_CHANNEL_AFFECTING = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,  # Conv Family
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear )
#===============================================================

#==================== Pass Through==============================
_PASS_THROUGH = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
    nn.ReLU, nn.ReLU6, nn.LeakyReLU, nn.ELU, nn.SiLU, nn.GELU, nn.Hardswish, nn.Hardtanh,
    nn.Sigmoid, nn.Tanh, nn.Mish, nn.Softplus, nn.Softsign,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout,
    nn.Identity,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.Flatten,
)
#=============================================================




#================= Node type informer ========================
def _NODE_TYPE(n :fx.Node, name_to_mod: Dict[str, nn.Module]) -> str:
    
    #--------------------------------------------------------
    if n.op =='call_module':
        m=name_to_mod[n.target]
        if isinstance(m, _CHANNEL_AFFECTING): return "Channel_affecting"
        if isinstance(m, _PASS_THROUGH): return "pass"
        return "other"
    #---------------------------------------------------------

    #---------------------------------------------------------
    if n.op == 'call_function' :
        tgt = str(n.target)

        if "aten.add.tensor" in tgt or "add.tensor" in tgt :
            return "merge_add"
        if "aten.cat.default" in tgt or ".cat" in tgt : 
            return "merge_cat"
        
        if any(k in tgt for k in ("relu", "batch_norm", "dropout", "max_pool", "avg_pool", "adaptive_avg_pool")): 
            # lite functional pass-throughs
            return "pass"
        return "other"
    #--------------------------------------------------------


    #--------------------------------------------------------
    if n.op == "call_method":
        if n.target in ("add", "_add"):
            return "merge_add"
        return "other"
    
    return "other"
    #-------------------------------------------------------


#=========================== Name to mod =============================
def _typename(n: fx.node, name_to_mod: Dict[str, nn.Module]) ->str:
    if n.op =="call_module":
        return type(name_to_mod[n.target]).__name__
    if n.op in("call_function", "call_method"):
        return str(n.target)
    return n.op
#=====================================================================

#=====================================================================
def _weight_key_for(n:fx.node, name_to_mod: Dict[str, nn.MOdule]) -> str:
    if n.op == "call_module":
        m=name_to_mod[n.target]
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear )):
            return f"{n.target}.weight"
    return "-"
#=====================================================================


#==================== Graph walk to build the Rows ===================
def _build_edges(gm:fx.GraphModule) -> List[Dict[str, Any]]:
    name_to_mod =dict(gm.named_modules())
    nodes: List[fx.Node] =list(gm.graph.nodes)

    user: Dict[fx.Node, List[fx.Node]]= {n: [] for n in nodes}
    
    for n in nodes:
        for u in user:
            user[n].append(u)
    rows:List[Dict[str, Any]] =[]
#====================================================================            



#====================== BFS traversal of graph ======================
def bfs_from(src: fx.Node):
    """From a channel-affecting node, walk pass-through chain to next consumer."""
    q: List[Tuple[fx.Node, List[str]]] = [(src, [])]
    seen: Set[fx.Node] = {src}
    while q:
        cur, path = q.pop(0)
        for u in users.get(cur, []):
            if u in seen:
                continue
            seen.add(u)
            kind = _NODE_TYPE(u, name_to_mod)
            if kind == "pass":
                label = u.target if isinstance(u.target, str) else _typename(u, name_to_mod)
                q.append((u, path + [str(label)]))
            elif kind in ("channel_affect", "merge_add", "merge_cat"):
                rows.append({
                    "Master": src.name,
                    "type": _typename(src, name_to_mod),
                    "target": _weight_key_for(src, name_to_mod),
                    "Slave": u.name,
                    "type_slave": _typename(u, name_to_mod),
                    "target_slave": _weight_key_for(u, name_to_mod),
                    "is_mirror": None,             # filled later for residual adds
                    "pass_through": path,          # list[str]
                    "pruning_ratio": 0.1,
                })
            else:
                q.append((u, path))

for n in nodes:
    if _node_kind(n, name_to_mod) == "channel_affect":
        bfs_from(n)

# ---- set is_mirror for residual adds ----
# For every add node (Slave), choose the first incoming edge as canonical;
# other incoming edges mirror that master's weight key.
from collections import defaultdict
by_slave_add: Dict[str, List[int]] = defaultdict(list)
for i, r in enumerate(rows):
    if r["type_slave"] == "aten.add.Tensor" or "aten.add.Tensor" in str(r["type_slave"]):
        by_slave_add[r["Slave"]].append(i)

for _, idxs in by_slave_add.items():
    if len(idxs) < 2:
        continue
    first = rows[idxs[0]]
    master_key = first["target"] if first["target"] != "-" else first["Master"]
    for j in idxs[1:]:
        rows[j]["is_mirror"] = master_key

return rows

# ---- public API ----
def model_to_table(model: nn.Module) -> List[Dict[str, Any]]:
    """
    Returns a list of dict rows with columns:
    Master | type | target | Slave | type_slave | target_slave | is_mirror | pass_through | pruning_ratio
    """
    gm = fx.symbolic_trace(model)
    return _build_edges(gm)

    def pretty_print(rows: List[Dict[str, Any]]):
    headers = ["Master","type","target","Slave","type_slave","target_slave","is_mirror","pass_through","pruning_ratio"]
    colw = {h: max(len(h), max((len(str(r[h])) for r in rows), default=0)) for h in headers}
    print(" | ".join(h.ljust(colw[h]) for h in headers))
    print("-+-".join("-"*colw[h] for h in headers))
    for r in rows:
        print(" | ".join(str(r[h]).ljust(colw[h]) for h in headers))

    # ---- quick smoke test (comment out in prod) ----
    if __name__ == "__main__":
    import torchvision.models as tvm
    for name, ctor in [
        ("resnet18", tvm.resnet18),
        ("googlenet", tvm.googlenet),
        ("mobilenet_v3_large", tvm.mobilenet_v3_large),
    ]:
        print(f"\n=== {name} ===")
        m = ctor(weights=None)
        rows = model_to_table(m)
        # show a few interesting lines
        focus = [r for r in rows if ("downsample" in r["target"] or "add.Tensor" in str(r["type_slave"]) or "cat" in str(r["type_slave"]))]
        pretty_print(focus[:12] if focus else rows[:12])

    #======================================================================



