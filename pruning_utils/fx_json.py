#========== imports ==========================================================
import os, json, importlib
from typing import Any, Dict, List, Tuple, Union
import torch
import torch.nn as nn
import torch.fx as fx
#===========================================================================


#========== import resolvers & JSON (de)sanitizers ==========================
def _import_from_string(path: str) -> Any:
    if path.startswith("torch.ops."):
        obj = torch
        for part in path.split(".")[1:]:
            obj = getattr(obj, part)
        return obj
    mod_path, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), name)

def _json_sanitize(obj: Any) -> Any:
    if isinstance(obj, (list, tuple)):   return [_json_sanitize(v) for v in obj]
    if isinstance(obj, dict):            return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, slice):           return {"__pytype__":"slice","start":_json_sanitize(obj.start),
                                                 "stop":_json_sanitize(obj.stop),"step":_json_sanitize(obj.step)}
    if obj is Ellipsis:                  return {"__pytype__":"ellipsis"}
    if isinstance(obj, torch.Size):      return list(obj)
    if isinstance(obj, (torch.dtype, torch.device)): return str(obj)
    return obj

def _unsanitize_literals(x: Any) -> Any:
    if isinstance(x, dict) and "__pytype__" in x:
        t = x["__pytype__"]
        if t == "slice":    return slice(x.get("start", None), x.get("stop", None), x.get("step", None))
        if t == "ellipsis": return Ellipsis
    return x

def _deserialize_arg(x: Any, env: Dict[str, fx.Node]) -> Any:
    x = _unsanitize_literals(x)
    if isinstance(x, dict) and "type" in x:
        t = x["type"]
        if t == "node":       return env[x["name"]]
        if t == "primitive":  return x["value"]
        return x.get("value", x)
    if isinstance(x, list):   return [ _deserialize_arg(v, env) for v in x ]
    if isinstance(x, tuple):  return tuple(_deserialize_arg(v, env) for v in x)
    if isinstance(x, dict):   return {k: _deserialize_arg(v, env) for k, v in x.items()}
    return x
#===========================================================================


#========== module init-args extractor (extendable) ==========================
def _init_args(m: nn.Module) -> Dict[str, Any]:
    # Convs
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return dict(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size,
                    stride=m.stride, padding=m.padding, dilation=m.dilation, groups=m.groups,
                    bias=(m.bias is not None), padding_mode=m.padding_mode)
    if isinstance(m, (nn.ConvTranspose1d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
        return dict(in_channels=m.in_channels, out_channels=m.out_channels, kernel_size=m.kernel_size,
                    stride=m.stride, padding=m.padding, output_padding=m.output_padding, groups=m.groups,
                    bias=(m.bias is not None), dilation=m.dilation)

    # Linear
    if isinstance(m, nn.Linear):
        return dict(in_features=m.in_features, out_features=m.out_features, bias=(m.bias is not None))

    # Norms
    if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return dict(num_features=m.num_features, eps=m.eps, momentum=m.momentum,
                    affine=m.affine, track_running_stats=m.track_running_stats)
    if isinstance(m, nn.LayerNorm):
        return dict(normalized_shape=m.normalized_shape, eps=m.eps, elementwise_affine=m.elementwise_affine)
    if isinstance(m, nn.GroupNorm):
        return dict(num_groups=m.num_groups, num_channels=m.num_channels, eps=m.eps, affine=m.affine)
    if hasattr(nn, "InstanceNorm2d") and isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
        return dict(num_features=m.num_features, eps=m.eps, momentum=m.momentum, affine=m.affine,
                    track_running_stats=getattr(m, "track_running_stats", False))

    # Pooling
    if isinstance(m, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        return dict(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                    dilation=m.dilation, return_indices=m.return_indices, ceil_mode=m.ceil_mode)
    if isinstance(m, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        return dict(kernel_size=m.kernel_size, stride=m.stride, padding=m.padding,
                    ceil_mode=m.ceil_mode, count_include_pad=m.count_include_pad, divisor_override=m.divisor_override)
    if isinstance(m, (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d,
                      nn.AdaptiveMaxPool1d, nn.AdaptiveMaxPool2d, nn.AdaptiveMaxPool3d)):
        return dict(output_size=m.output_size)

    # Activations
    if isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.ELU, nn.LeakyReLU, nn.Hardswish, nn.Hardtanh)):
        d = {"inplace": getattr(m, "inplace", False)}
        if isinstance(m, nn.LeakyReLU): d["negative_slope"] = m.negative_slope
        if isinstance(m, nn.ELU):       d["alpha"] = m.alpha
        if isinstance(m, nn.Hardtanh):  d["min_val"], d["max_val"] = m.min_val, m.max_val
        return d
    if isinstance(m, nn.GELU):
        return {"approximate": getattr(m, "approximate", "none")}

    # Dropout / Shape
    if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
        return dict(p=m.p, inplace=getattr(m, "inplace", False))
    if isinstance(m, nn.Flatten):
        return dict(start_dim=m.start_dim, end_dim=m.end_dim)
    if isinstance(m, nn.Identity):
        return {}

    # Optional: a few torchvision composite blocks (defensive)
    try:
        from torchvision.models.mobilenetv2 import InvertedResidual as V2IR
        if isinstance(m, V2IR):
            d = {}
            for k in ("stride", "use_res_connect"):
                if hasattr(m, k): d[k] = getattr(m, k)
            first, last = None, None
            for subm in m.modules():
                if isinstance(subm, nn.Conv2d):
                    if first is None: first = subm
                    last = subm
            if first and last:
                d.update(inp=first.in_channels, oup=last.out_channels)
            return d
    except Exception:
        pass

    try:
        from torchvision.models.mobilenetv3 import InvertedResidual as V3IR, SqueezeExcitation as SE
        if isinstance(m, V3IR):
            d = {}
            if hasattr(m, "use_res_connect"): d["use_res_connect"] = m.use_res_connect
            first, last = None, None
            for subm in m.modules():
                if isinstance(subm, nn.Conv2d):
                    if first is None: first = subm
                    last = subm
            if first and last:
                d.update(inp=first.in_channels, oup=last.out_channels)
            return d
        if isinstance(m, SE):
            d = {}
            for k in ("input_channels", "squeeze_channels"):
                if hasattr(m, k): d[k] = getattr(m, k)
            if hasattr(m, "activation"):
                d["activation"] = type(m.activation).__module__ + "." + type(m.activation).__name__
            if hasattr(m, "scale_activation"):
                d["scale_activation"] = type(m.scale_activation).__module__ + "." + type(m.scale_activation).__name__
            return d
    except Exception:
        pass

    return {}
#===========================================================================


#========== exporter: nn.Module -> JSON + PT ================================
def export_fx_arch(model: nn.Module, arch_json_path: str, weights_pt_path: str):
    gm = fx.symbolic_trace(model)
    name_to_mod = dict(gm.named_modules())

    def ser_target(tgt: Any) -> str:
        if isinstance(tgt, str): return tgt
        if hasattr(tgt, "__module__") and hasattr(tgt, "__name__"):
            return f"{tgt.__module__}.{tgt.__name__}"
        s = str(tgt)  # OpOverload prints like aten.add.Tensor
        if "aten." in s and "Tensor" in s:
            pieces = s.strip().split(".")
            if len(pieces) >= 2:
                return f"torch.ops.aten.{pieces[1]}.Tensor"
        return s

    def ser_arg(a: Any) -> Any:
        if isinstance(a, fx.Node):         return {"type": "node", "name": a.name}
        if isinstance(a, (list, tuple)):   return [ser_arg(v) for v in a]
        if isinstance(a, dict):            return {k: ser_arg(v) for k, v in a.items()}
        return {"type": "primitive", "value": _json_sanitize(a)}

    nodes = []
    for n in gm.graph.nodes:
        rec = dict(
            name=n.name,
            op=n.op,
            target=ser_target(n.target),
            args=ser_arg(n.args),
            kwargs=ser_arg(n.kwargs),
            metadata={}
        )
        if n.op == "call_module":
            m = name_to_mod[n.target]
            mt = type(m)
            rec["metadata"] = {
                "module_type": f"{mt.__module__}.{mt.__name__}",
                "init_args": _json_sanitize(_init_args(m)),
            }
        nodes.append(rec)

    os.makedirs(os.path.dirname(arch_json_path) or ".", exist_ok=True)
    with open(arch_json_path, "w") as f:
        json.dump({"version": 1, "framework": "torch.fx", "nodes": nodes}, f, indent=2)

    os.makedirs(os.path.dirname(weights_pt_path) or ".", exist_ok=True)
    torch.save(model.state_dict(), weights_pt_path)

    leaf_types = sorted({rec["metadata"].get("module_type") for rec in nodes if rec["op"] == "call_module"})
    print("[export_fx_arch] leaf module types:", leaf_types)
#===========================================================================


#========== importer: JSON (+ optional PT) -> GraphModule ====================
def import_fx_arch(arch_json_path: str, weights_pt_path: str = None, strict_load: bool = False,
                   force_single_tensor_output: bool = False) -> Union[fx.GraphModule, nn.Module]:
    with open(arch_json_path, "r") as f:
        arch = json.load(f)
    assert arch.get("framework") == "torch.fx", "Unsupported framework in JSON"
    nodes = arch["nodes"]

    # pass 1: instantiate leaf modules on a fresh root with exact hierarchy
    root = nn.Module()
    for r in nodes:
        if r["op"] == "call_module":
            tgt = r["target"]
            meta = r.get("metadata", {})
            cls_path = meta.get("module_type")
            if not cls_path:
                raise ValueError(f"Missing module_type for target '{tgt}'")
            cls = _import_from_string(cls_path)
            mod = cls(**(meta.get("init_args", {}) or {}))
            parts = tgt.split(".")
            parent = root
            for p in parts[:-1]:
                if not hasattr(parent, p):
                    setattr(parent, p, nn.Module())
                parent = getattr(parent, p)
            setattr(parent, parts[-1], mod)

    # pass 2: rebuild graph
    graph, env = fx.Graph(), {}
    for r in nodes:
        op, name, target = r["op"], r["name"], r["target"]
        args   = _deserialize_arg(r["args"], env)
        kwargs = _deserialize_arg(r["kwargs"], env)

        if op == "placeholder":
            node = graph.placeholder(target); node.name = name
        elif op == "get_attr":
            node = graph.get_attr(target);    node.name = name
        elif op == "call_function":
            fn = _import_from_string(target)
            node = graph.call_function(fn, args=tuple(args) if isinstance(args, list) else args, kwargs=kwargs); node.name = name
        elif op == "call_method":
            node = graph.call_method(target, args=tuple(args) if isinstance(args, list) else args, kwargs=kwargs); node.name = name
        elif op == "call_module":
            node = graph.call_module(target, args=tuple(args) if isinstance(args, list) else args, kwargs=kwargs); node.name = name
        elif op == "output":
            # If output args is a single-element list/tuple, unwrap to the element to avoid list outputs.
            out_arg = args
            if isinstance(out_arg, (list, tuple)) and len(out_arg) == 1:
                out_arg = out_arg[0]
            node = graph.output(out_arg)
        else:
            raise RuntimeError(f"Unsupported FX op: {op}")
        env[name] = node

    graph.lint()
    gm = fx.GraphModule(root, graph)

    if weights_pt_path:
        sd = torch.load(weights_pt_path, map_location="cpu")
        missing, unexpected = gm.load_state_dict(sd, strict=strict_load)
        if missing or unexpected:
            print(f"[import_fx_arch] load_state_dict - missing: {len(missing)}, unexpected: {len(unexpected)}")

    if force_single_tensor_output:
        gm = LogitsAdapter(gm)
    return gm
#===========================================================================


#========== convenience: split wrapper into two calls ========================
def fx2json(model: nn.Module, out_dir: str, stem: str):
    os.makedirs(out_dir, exist_ok=True)
    arch_json = os.path.join(out_dir, f"{stem}.json")
    weights_pt = os.path.join(out_dir, f"{stem}.pt")
    export_fx_arch(model, arch_json, weights_pt)
    print('[OK] FX -> JSON creation successful')
    return arch_json, weights_pt

def json2arch(arch_json_path: str, weights_pt_path: str = None, strict_load: bool = False,
              force_single_tensor_output: bool = False) -> Union[fx.GraphModule, nn.Module]:
    return import_fx_arch(arch_json_path, weights_pt_path, strict_load=strict_load,
                          force_single_tensor_output=force_single_tensor_output)
#===========================================================================


#========== training-safe helpers ===========================================
def _unwrap_logits(preds: Any) -> torch.Tensor:
    """Return a Tensor logits from Tensor | list | tuple | dict outputs."""
    if torch.is_tensor(preds):
        return preds
    if isinstance(preds, (list, tuple)):
        # prefer the last tensor (common pattern: [aux, logits])
        for x in reversed(preds):
            if torch.is_tensor(x):
                return x
        raise TypeError("No tensor found inside list/tuple preds")
    if isinstance(preds, dict):
        for k in ("logits", "out", "pred", "y", "cls"):
            v = preds.get(k, None)
            if torch.is_tensor(v):
                return v
        last_t = None
        for v in preds.values():
            if torch.is_tensor(v):
                last_t = v
        if last_t is not None:
            return last_t
        raise TypeError("No tensor found inside dict preds")
    raise TypeError(f"Unsupported preds type: {type(preds)}")

def mixup_criterion(preds: Any, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy with soft labels; robust to non-tensor model outputs."""
    preds = _unwrap_logits(preds)
    return -(targets * torch.log_softmax(preds, dim=-1)).sum(dim=1).mean()

class LogitsAdapter(nn.Module):
    """Wrap any model and guarantee a single Tensor output for CE/mixup."""
    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core
    def forward(self, x):
        y = self.core(x)
        return _unwrap_logits(y)
#===========================================================================


#========== quick self-test (comment out when importing) =====================
if __name__ == "__main__":
    # Two tiny nets:
    #  - TinyNet returns pure logits Tensor
    #  - TinyNetWithAux returns (aux, logits) to test unwrapping / adapter
    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.bn1   = nn.BatchNorm2d(8)
            self.relu1 = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.pool  = nn.AdaptiveAvgPool2d((32,32))
            self.flat  = nn.Flatten()
            self.fc    = nn.Linear(16*32*32, 10)
        def forward(self, x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.pool(self.conv2(x))
            x = self.flat(x)
            return self.fc(x)

    class TinyNetWithAux(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d((8,8))
            self.flat = nn.Flatten()
            self.aux  = nn.Linear(8*8*8, 5)   # pretend aux head
            self.fc   = nn.Linear(8*8*8, 10)  # logits head
        def forward(self, x):
            h = self.pool(self.conv(x))
            f = self.flat(h)
            aux = self.aux(f)
            logits = self.fc(f)
            return aux, logits   # tuple output on purpose

    out_dir = "./assets/recons"
    stem    = "tinynet_cifar10"

    # i. export FX -> JSON/PT (using the aux model to test robustness)
    net = TinyNetWithAux()
    jpath, ppath = fx2json(net, out_dir, stem)
    print("[OK] Exported:", jpath, ppath)

    # ii. import JSON(+PT) -> GraphModule
    gm = json2arch(jpath, ppath, strict_load=False, force_single_tensor_output=False)
    print("[OK] Rebuilt:", type(gm).__name__)

    # sanity forwards
    gm.eval()
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        y_raw = gm(x)              # this will be a tuple (aux, logits)
        y_unwrapped = _unwrap_logits(y_raw)
        print("[OK] unwrap:", isinstance(y_unwrapped, torch.Tensor), y_unwrapped.shape)

    # adapter path
    wrapped = LogitsAdapter(gm).eval()
    with torch.no_grad():
        y = wrapped(x)             # guaranteed Tensor
        print("[OK] adapter:", isinstance(y, torch.Tensor), y.shape)

    # mock mixup (soft labels) loss test
    soft_targets = torch.softmax(torch.randn(2, 10), dim=-1)
    loss = mixup_criterion(y_raw, soft_targets)  # you can pass raw gm output
    print("[OK] mixup_criterion works. loss=", float(loss))
