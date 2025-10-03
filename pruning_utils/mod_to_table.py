#========== imports ==========================================================
import os
import csv
import torch
import operator
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Tuple
#===========================================================================


#========== config: what’s a pass-through vs channel-modifier ================
CHANNEL_MODIFIERS = (
    nn.Conv1d, nn.Conv2d, nn.Conv3d,
    nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d,
    nn.Linear,  # consumes channel aggregate (C*H*W)
)

PASSTHROUGH_MODULES = (
    nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,   # mirror channels
    nn.ReLU, nn.ReLU6, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Hardtanh, nn.Hardswish, nn.GELU,
    nn.Sigmoid, nn.Tanh, nn.Mish, nn.Softplus, nn.Softsign,
    nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d,
    nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d,
    nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d,
    nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
    nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d,
    nn.Flatten, nn.Identity,
)
# add ops to treat as pass-through (residual adds)
PASSTHROUGH_FUNCTION_NAMES = {"add"}  # operator.add / torch.add / aten.add.Tensor all stringify to include 'add'
#===========================================================================


#========== helpers ==========================================================
def _is_channel_modifier(m: nn.Module) -> bool:
    return isinstance(m, CHANNEL_MODIFIERS)

def _is_passthrough_module(name: str, m: nn.Module) -> bool:
    # treat downsample containers as pass-through; inner conv will be explicit call_module anyway
    if "downsample" in name:
        return True
    return isinstance(m, PASSTHROUGH_MODULES)

def _is_passthrough_function(target) -> bool:
    # FX might give operator.add, torch.add, or torch.ops.aten.add.Tensor
    t = str(target)
    return any(k in t for k in PASSTHROUGH_FUNCTION_NAMES)

def _out_id(mod_name: str, m: nn.Module) -> str:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return f"{mod_name}.out_channels[:]"
    if isinstance(m, nn.Linear):
        return f"{mod_name}.out_features[:]"
    return f"{mod_name}[:]"

def _in_id(mod_name: str, m: nn.Module) -> str:
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return f"{mod_name}.in_channels[:]"
    if isinstance(m, nn.Linear):
        return f"{mod_name}.in_features[:]"  # aggregate (C*H*W)
    return f"{mod_name}[:]"
#=============================================================================


#========== core: pruning table builder (BN/act/pool/add pass-through) =======
def build_pruning_table(model: nn.Module, table_path : str, default_ratio: float = 0.0, ) -> List[Dict]:
    """
    Returns rows:
      {
        'master': master_name,
        'master_id': e.g. 'conv2.out_channels[:]',
        'slave': slave_name,
        'slave_id': e.g. 'conv3.in_channels[:]',
        'pass_through': [names we skipped like bn2, relu, maxpool, downsample.*, add]',
        'pruning_ratio': float,
      }
    Only connects channel-dimension modifiers (Conv/ConvT/Linear).
    """
    gm = fx.symbolic_trace(model)
    name_to_mod = dict(gm.named_modules())

    # collect call_module nodes
    callmods: List[Tuple[fx.Node, str, nn.Module]] = []
    for node in gm.graph.nodes:
        if node.op == "call_module":
            callmods.append((node, node.target, name_to_mod.get(node.target, None)))

    rows: List[Dict] = []

    # for each producer that modifies channels, find nearest downstream channel-modifier consumer(s)
    for node, name, mod in callmods:
        if mod is None or not _is_channel_modifier(mod):
            continue  # only treat channel-modifiers as masters

        # BFS over users, skipping passthroughs; accumulate pass_through names per path
        queue: List[Tuple[fx.Node, List[str]]] = [(u, []) for u in node.users.keys()]
        visited = set()

        while queue:
            nxt, trail = queue.pop(0)
            if nxt in visited:
                continue
            visited.add(nxt)

            if nxt.op == "call_module":
                cm_name = nxt.target
                cm_mod = name_to_mod.get(cm_name, None)
                if cm_mod is None:
                    continue
                if _is_passthrough_module(cm_name, cm_mod):
                    # record and continue walking
                    queue.extend([(uu, trail + [cm_name]) for uu in nxt.users.keys()])
                    continue
                if _is_channel_modifier(cm_mod):
                    # found true consumer
                    rows.append({
                        "master": name,
                        "master_id": _out_id(name, mod),
                        "slave": cm_name,
                        "slave_id": _in_id(cm_name, cm_mod),                        
                        "pass_through": trail.copy(),
                        "pruning_ratio": default_ratio,
                    })
                    # do not expand beyond first channel-modifier by default
                    continue
                # unknown module type: stop this path

            elif nxt.op in ("call_function", "call_method"):
                # treat add / view / reshape etc. as pass-through
                label = None
                if nxt.op == "call_function" and _is_passthrough_function(nxt.target):
                    label = "add"
                elif nxt.op == "call_method":
                    # many are 'view', 'reshape', 'flatten', 'permute'—safe to pass
                    label = str(nxt.target)
                queue.extend([(uu, trail + ([label] if label else [])) for uu in nxt.users.keys()])

            elif nxt.op == "output":
                # path ends
                pass

    #-----------------------------------------------------------------------
    fieldnames = list(rows[0].keys())
    try: 
        with open (table_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Successfully saved {len(rows)} rows to {table_path}")
    except IOError as e:
        print(f"I/O error while saving CSV: {e}")    

    # ----------------------------------------------------------------------        

    return rows
#===========================================================================


#========== pretty printer (optional) =======================================
def print_table(rows: List[Dict]):
    if not rows:
        print("(empty)")
        return
    cols = ["master", "master_id", "slave", "slave_id",  "pass_through", "pruning_ratio"]
    widths = {c: max(len(c), max(len(str(r[c])) for r in rows)) for c in cols}
    print(" | ".join(c.ljust(widths[c]) for c in cols))
    print("-+-".join("-" * widths[c] for c in cols))
    for r in rows:
        print(" | ".join(str(r[c]).ljust(widths[c]) for c in cols))
#===========================================================================


#========== tiny demo with your simple chain ================================
if __name__ == "__main__":
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 8, 3, padding=1)
            self.bn2   = nn.BatchNorm2d(8)
            self.relu  = nn.ReLU(inplace=True)
            self.pool  = nn.MaxPool2d(2)
            self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
            self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
            self.bn4   = nn.BatchNorm2d(16)
            self.relu4 = nn.ReLU(inplace=True)
            self.flat  = nn.Flatten()
            self.fc    = nn.Linear(16*16*16, 10)  # for 32x32 input

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.pool(x)
            x = self.conv3(x)
            x = self.conv4(x)
            x = self.bn4(x)
            x = self.relu4(x)
            x = self.flat(x)
            return self.fc(x)

    #net = TinyNet()
    table_dir       = os.path.join('./assets/table', 'TinyModel_table.csv')
    rows = build_pruning_table(net, table_path=table_dir, default_ratio=0.0, )
    print_table(rows)
