# prune_and_morph.py
# Minimal utilities to (A) prune tensors in a state_dict and (B) morph a model to match.
# Supports Conv2d / ConvTranspose2d (standard, grouped, depthwise), BN/IN/GN/LN, Linear.

from typing import Dict, List, Optional, Tuple, Any
from math import gcd
import torch
import torch.nn as nn

Tensor = torch.Tensor
SDict = Dict[str, Tensor]

# =============================================================================
# ---- State dict tools ----
# =============================================================================

def _ensure_cpu_clone(x: Optional[Tensor]) -> Optional[Tensor]:
    if x is None: return None
    return x.detach().cpu().clone()

def _get(sd: SDict, key: str) -> Optional[Tensor]:
    return sd.get(key, None)

def _set(sd: SDict, key: str, t: Optional[Tensor]):
    if t is None:
        if key in sd:
            del sd[key]
    else:
        sd[key] = t

def _l2_scores_along(W: Tensor, dim: int) -> Tensor:
    # L2 per "channel" on 'dim'
    perm = [dim] + [i for i in range(W.ndim) if i != dim]
    V = W.permute(*perm).contiguous().flatten(1)
    return V.norm(p=2, dim=1)

def _group_slices(n: int, g: int) -> List[slice]:
    assert g > 0 and n % g == 0, f"channels {n} must be divisible by groups {g}"
    s = n // g
    return [slice(i*s, (i+1)*s) for i in range(g)]

def clone_state_dict(sd: SDict) -> SDict:
    return {k: _ensure_cpu_clone(v) for k, v in sd.items()}

# --------------------------
# Conv2d / ConvTranspose2d
# --------------------------

def prune_conv_out_sd(
    name: str,
    *,
    source_sd: SDict,
    target_sd: SDict,
    groups: int,
    pr: Optional[float] = None,
    keep_idx: Optional[List[int]] = None,
    is_transposed: bool = False,
) -> Dict[str, Any]:
    """
    Prune OUT channels of conv by L2 within each group (keeps group divisibility).
    - For Conv2d: weight [O, I_g, kH, kW], bias [O]
    - For ConvT2d: weight [I, O_g, kH, kW], bias (rare, kept as-is)
    Returns: {'kept_out_idx': [...], 'new_out_channels': int}
    """
    w_key, b_key = f"{name}.weight", f"{name}.bias"
    W = _get(source_sd, w_key);  B = _get(source_sd, b_key)
    if W is None: raise KeyError(f"Missing {w_key}")
    W = _ensure_cpu_clone(W); B = _ensure_cpu_clone(B)

    if not is_transposed:
        O = W.shape[0]
        if keep_idx is None:
            assert pr is not None, "provide pr or keep_idx"
            kept = []
            for s in _group_slices(O, groups):
                scores = _l2_scores_along(W[s], dim=0)
                n_prune = int(scores.numel() * pr)
                n_keep  = max(scores.numel() - n_prune, 1)
                topk = torch.argsort(scores, descending=True)[:n_keep].tolist()
                kept += [s.start + k for k in topk]
            keep_idx = kept
        # enforce divisibility
        assert len(keep_idx) % groups == 0, "new out_channels must be divisible by groups"
        W_new = W.index_select(0, torch.as_tensor(keep_idx, dtype=torch.long))
        B_new = B.index_select(0, torch.as_tensor(keep_idx, dtype=torch.long)) if B is not None else None
        _set(target_sd, w_key, W_new); _set(target_sd, b_key, B_new)
        return {"kept_out_idx": keep_idx, "new_out_channels": len(keep_idx)}
    else:
        # ConvTranspose2d: weight [I, O_g, kH, kW] — logical OUT is dim=1 * groups
        O = W.shape[1] * groups
        if keep_idx is None:
            assert pr is not None, "provide pr or keep_idx"
            kept_local = []
            # rank per-group along dim=1
            O_g = W.shape[1]
            scores = _l2_scores_along(W, dim=1)  # length O_g
            n_prune = int(scores.numel() * pr)
            n_keep  = max(scores.numel() - n_prune, 1)
            local_keep = torch.argsort(scores, descending=True)[:n_keep].tolist()
            kept_local = local_keep
            keep_idx = []
            for gi, s in enumerate(_group_slices(O, groups)):
                keep_idx += [s.start + k for k in kept_local]
        assert len(keep_idx) % groups == 0
        # Map global to local (dim=1)
        O_g = O // groups
        local_idx = [k % O_g for k in keep_idx]
        W_new = W.index_select(1, torch.as_tensor(local_idx, dtype=torch.long))
        _set(target_sd, w_key, W_new); _set(target_sd, b_key, B)  # bias untouched
        return {"kept_out_idx": keep_idx, "new_out_channels": len(keep_idx)}

def prune_conv_in_sd(
    name: str,
    *,
    source_sd: SDict,
    target_sd: SDict,
    groups: int,
    pr: Optional[float] = None,
    keep_in_idx: Optional[List[int]] = None,
    is_transposed: bool = False,
) -> Dict[str, Any]:
    """
    Prune IN channels of conv. Keeps per-group counts equal so new_in % groups == 0.
    Returns: {'kept_in_idx': [...], 'new_in_channels': int}
    """
    w_key, b_key = f"{name}.weight", f"{name}.bias"
    W = _get(source_sd, w_key);  B = _get(source_sd, b_key)
    if W is None: raise KeyError(f"Missing {w_key}")
    W = _ensure_cpu_clone(W); B = _ensure_cpu_clone(B)

    if not is_transposed:
        # Conv2d: W [O, I_g, kH, kW], select dim=1
        I_g = W.shape[1]
        I = I_g * groups
        if keep_in_idx is None:
            assert pr is not None, "provide pr or keep_in_idx"
            # rank per-group on dim=1
            scores = _l2_scores_along(W.permute(1,0,2,3).contiguous(), dim=0)  # over input slice
            # scores length is I_g; same for each group
            n_prune = int(scores.numel() * pr)
            n_keep  = max(scores.numel() - n_prune, 1)
            local_keep = torch.argsort(scores, descending=True)[:n_keep].tolist()
            keep_in_idx = []
            for gi, s in enumerate(_group_slices(I, groups)):
                base = s.start
                keep_in_idx += [base + k for k in local_keep]
        assert len(keep_in_idx) % groups == 0
        # map global -> local dim=1
        local = [k % I_g for k in keep_in_idx]
        W_new = W.index_select(1, torch.as_tensor(local, dtype=torch.long))
        _set(target_sd, w_key, W_new); _set(target_sd, b_key, B)
        return {"kept_in_idx": keep_in_idx, "new_in_channels": len(keep_in_idx)}
    else:
        # ConvT2d: W [I, O_g, kH, kW], input is dim=0
        I = W.shape[0]
        if keep_in_idx is None:
            assert pr is not None, "provide pr or keep_in_idx"
            scores = _l2_scores_along(W, dim=0)  # per input channel
            n_prune = int(scores.numel() * pr)
            n_keep  = max(scores.numel() - n_prune, 1)
            keep_in_idx = torch.argsort(scores, descending=True)[:n_keep].tolist()
        W_new = W.index_select(0, torch.as_tensor(keep_in_idx, dtype=torch.long))
        _set(target_sd, w_key, W_new); _set(target_sd, b_key, B)
        return {"kept_in_idx": keep_in_idx, "new_in_channels": len(keep_in_idx)}

# ---------------------------------------------------------------------------------------
# Normalizations
# ---------------------------------------------------------------------------------------

def prune_batchnorm_sd(name: str, *, keep_idx: List[int], source_sd: SDict, target_sd: SDict) -> Dict[str, Any]:
    keys = [".weight", ".bias", ".running_mean", ".running_var"]
    idx = torch.as_tensor(keep_idx, dtype=torch.long)
    for suf in keys:
        t = _get(source_sd, f"{name}{suf}")
        if t is None: raise KeyError(f"Missing {name}{suf}")
        _set(target_sd, f"{name}{suf}", _ensure_cpu_clone(t).index_select(0, idx))
    # num_batches_tracked stays (PyTorch handles size-1 buffer)
    nbt = _get(source_sd, f"{name}.num_batches_tracked")
    if nbt is not None: _set(target_sd, f"{name}.num_batches_tracked", _ensure_cpu_clone(nbt))
    return {"kept_features": keep_idx, "new_num_features": len(keep_idx)}

def prune_affine_norm_sd(name: str, *, keep_idx: List[int], source_sd: SDict, target_sd: SDict) -> Dict[str, Any]:
    # For GroupNorm/LayerNorm/InstanceNorm(affine): only affine params exist
    idx = torch.as_tensor(keep_idx, dtype=torch.long)
    for suf in [".weight", ".bias"]:
        t = _get(source_sd, f"{name}{suf}")
        if t is not None:
            _set(target_sd, f"{name}{suf}", _ensure_cpu_clone(t).index_select(0, idx))
    return {"kept_features": keep_idx, "new_num_features": len(keep_idx)}

# -------
# Linear
# -------

def prune_linear_out_sd(name: str, *, pr: Optional[float] = None, keep_idx: Optional[List[int]] = None,
                        source_sd: SDict, target_sd: SDict) -> Dict[str, Any]:
    W = _get(source_sd, f"{name}.weight")
    if W is None: raise KeyError(f"Missing {name}.weight")
    B = _get(source_sd, f"{name}.bias")
    W = _ensure_cpu_clone(W); B = _ensure_cpu_clone(B)
    if keep_idx is None:
        assert pr is not None
        scores = W.norm(p=2, dim=1)
        n_prune = int(scores.numel() * pr)
        n_keep  = max(scores.numel() - n_prune, 1)
        keep_idx = torch.argsort(scores, descending=True)[:n_keep].tolist()
    idx = torch.as_tensor(keep_idx, dtype=torch.long)
    _set(target_sd, f"{name}.weight", W.index_select(0, idx))
    if B is not None: _set(target_sd, f"{name}.bias", B.index_select(0, idx))
    return {"kept_out_idx": keep_idx, "new_out_features": len(keep_idx)}

def prune_linear_in_sd(name: str, *, keep_in_idx: List[int], source_sd: SDict, target_sd: SDict) -> Dict[str, Any]:
    W = _get(source_sd, f"{name}.weight")
    if W is None: raise KeyError(f"Missing {name}.weight")
    B = _get(source_sd, f"{name}.bias")  # untouched
    W = _ensure_cpu_clone(W)
    idx = torch.as_tensor(keep_in_idx, dtype=torch.long)
    _set(target_sd, f"{name}.weight", W.index_select(1, idx))
    if B is not None: _set(target_sd, f"{name}.bias", _ensure_cpu_clone(B))
    return {"kept_in_idx": keep_in_idx, "new_in_features": len(keep_in_idx)}

# ==================================
# ---- Model morphing to new sd ----
# ==================================

def _parent_and_leaf(model: nn.Module, qualname: str) -> Tuple[nn.Module, str]:
    parts = qualname.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]

def _device_dtype_like(mod: nn.Module):
    for p in list(mod.parameters()) + list(mod.buffers()):
        return p.device, p.dtype
    return torch.device("cpu"), torch.float32

def _bool_sd(sd: SDict, key: str) -> bool:
    return key in sd

def _conv2d_from_sd(old: nn.Conv2d, name: str, sd: SDict) -> nn.Conv2d:
    W = sd[f"{name}.weight"]  # [O, I_g, kH, kW]
    O = W.shape[0]; I_g = W.shape[1]; g = old.groups; I = I_g * g
    has_bias = _bool_sd(sd, f"{name}.bias")
    dev, dt = _device_dtype_like(old)
    # depthwise: groups must equal in_ch (and out==in)
    if g == old.in_channels == old.out_channels:
        g = I; O = I
    return nn.Conv2d(
        in_channels=I, out_channels=O, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, dilation=old.dilation,
        groups=g, bias=has_bias, padding_mode=old.padding_mode,
        device=dev, dtype=dt
    )

def _convt2d_from_sd(old: nn.ConvTranspose2d, name: str, sd: SDict) -> nn.ConvTranspose2d:
    W = sd[f"{name}.weight"]  # [I, O_g, kH, kW]
    I = W.shape[0]; O_g = W.shape[1]; g = old.groups; O = O_g * g
    has_bias = _bool_sd(sd, f"{name}.bias")
    dev, dt = _device_dtype_like(old)
    return nn.ConvTranspose2d(
        in_channels=I, out_channels=O, kernel_size=old.kernel_size,
        stride=old.stride, padding=old.padding, output_padding=old.output_padding,
        groups=g, bias=has_bias, dilation=old.dilation,
        device=dev, dtype=dt
    )

def _bn_from_sd(old: nn.BatchNorm2d, name: str, sd: SDict) -> nn.BatchNorm2d:
    C = sd[f"{name}.running_mean"].shape[0] if f"{name}.running_mean" in sd else sd[f"{name}.weight"].shape[0]
    dev, dt = _device_dtype_like(old)
    return nn.BatchNorm2d(
        num_features=C, eps=old.eps, momentum=old.momentum,
        affine=_bool_sd(sd, f"{name}.weight") or _bool_sd(sd, f"{name}.bias"),
        track_running_stats=old.track_running_stats, device=dev, dtype=dt
    )

def _in_from_sd(old: nn.InstanceNorm2d, name: str, sd: SDict) -> nn.InstanceNorm2d:
    # InstanceNorm may not have running stats; infer C from weight if affine, else keep old
    if f"{name}.weight" in sd: C = sd[f"{name}.weight"].shape[0]
    else: C = old.num_features
    dev, dt = _device_dtype_like(old)
    return nn.InstanceNorm2d(
        num_features=C, eps=old.eps, momentum=old.momentum,
        affine=old.affine, track_running_stats=old.track_running_stats,
        device=dev, dtype=dt
    )

def _gn_from_sd(old: nn.GroupNorm, name: str, sd: SDict) -> nn.GroupNorm:
    if f"{name}.weight" in sd: C = sd[f"{name}.weight"].shape[0]
    else: C = old.num_channels
    g = old.num_groups
    if C % g != 0:
        g = gcd(C, g) or 1
    dev, dt = _device_dtype_like(old)
    return nn.GroupNorm(num_groups=g, num_channels=C, eps=old.eps, affine=old.affine, device=dev, dtype=dt)

def _ln_from_sd(old: nn.LayerNorm, name: str, sd: SDict) -> nn.LayerNorm:
    if f"{name}.weight" in sd: C = sd[f"{name}.weight"].shape[0]
    else: C = old.normalized_shape[0]
    dev, dt = _device_dtype_like(old)
    return nn.LayerNorm(normalized_shape=(C,), eps=old.eps, elementwise_affine=old.elementwise_affine, device=dev, dtype=dt)

def _linear_from_sd(old: nn.Linear, name: str, sd: SDict) -> nn.Linear:
    W = sd[f"{name}.weight"]; out_f, in_f = W.shape
    has_bias = _bool_sd(sd, f"{name}.bias")
    dev, dt = _device_dtype_like(old)
    return nn.Linear(in_features=in_f, out_features=out_f, bias=has_bias, device=dev, dtype=dt)

def morph_model_to_state_dict(model: nn.Module, new_sd: SDict) -> nn.Module:
    name_to_mod = dict(model.named_modules())
    rebuilt = []
    for name, mod in list(name_to_mod.items()):
        has_w = f"{name}.weight" in new_sd
        has_rm = f"{name}.running_mean" in new_sd
        try:
            if isinstance(mod, nn.Conv2d) and has_w:
                new_mod = _conv2d_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.ConvTranspose2d) and has_w:
                new_mod = _convt2d_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.BatchNorm2d) and (has_w or has_rm):
                new_mod = _bn_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.InstanceNorm2d) and (has_w or mod.affine):
                new_mod = _in_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.GroupNorm) and (has_w or mod.affine):
                new_mod = _gn_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.LayerNorm) and (has_w or mod.elementwise_affine):
                new_mod = _ln_from_sd(mod, name, new_sd)
            elif isinstance(mod, nn.Linear) and has_w:
                new_mod = _linear_from_sd(mod, name, new_sd)
            else:
                continue
        except KeyError:
            continue

        # Compare shape signature; if changed, replace
        def sig(m: nn.Module):
            if isinstance(m, nn.Conv2d): return ("conv2d", m.in_channels, m.out_channels, m.groups, m.kernel_size)
            if isinstance(m, nn.ConvTranspose2d): return ("convt2d", m.in_channels, m.out_channels, m.groups, m.kernel_size)
            if isinstance(m, nn.BatchNorm2d): return ("bn", m.num_features)
            if isinstance(m, nn.InstanceNorm2d): return ("in", m.num_features, m.affine)
            if isinstance(m, nn.GroupNorm): return ("gn", m.num_groups, m.num_channels)
            if isinstance(m, nn.LayerNorm): return ("ln", m.normalized_shape)
            if isinstance(m, nn.Linear): return ("linear", m.in_features, m.out_features)
            return ("other",)
        if sig(mod) != sig(new_mod):
            parent, leaf = _parent_and_leaf(model, name)
            setattr(parent, leaf, new_mod)
            rebuilt.append(name)

    if rebuilt:
        print(f"[morph] rebuilt: {rebuilt}")
    else:
        print("[morph] no rebuild needed")
    return model

# ==========================================
# ---- Orchestrator: apply a prune plan ----
# ==========================================

def _module_info(model: nn.Module) -> Dict[str, Dict[str, Any]]:
    info = {}
    for n, m in model.named_modules():
        d: Dict[str, Any] = {"type": type(m).__name__, "module": m}
        if isinstance(m, nn.Conv2d):
            d.update({"groups": m.groups, "is_transposed": False})
        elif isinstance(m, nn.ConvTranspose2d):
            d.update({"groups": m.groups, "is_transposed": True})
        info[n] = d
    return info

def apply_prune_plan_on_state_dict(
    model: nn.Module,
    original_sd: SDict,
    plan: List[Dict[str, Any]],
) -> SDict:
    """
    Apply a list of actions on a *copy* of original_sd.
    Each action: {'type': 'conv_out'|'conv_in'|'bn'|'affine_norm'|'linear_out'|'linear_in',
                  'name': '<qualname>',
                  'pr': <ratio> OR 'keep_idx': [indices] (or 'keep_in_idx' for conv/linear in),
                  (conv only) 'groups': <int> [auto from model if omitted],
                  (conv only) 'is_transposed': bool [auto from model if omitted]}
    Returns the new (pruned) state_dict.
    """
    new_sd = clone_state_dict(original_sd)
    modinfo = _module_info(model)

    for act in plan:
        t = act["type"]; name = act["name"]
        src = new_sd  # read from latest (cumulative)
        if t in ("conv_out", "conv_in"):
            mi = modinfo.get(name, {})
            groups = act.get("groups", mi.get("groups", 1))
            isT    = act.get("is_transposed", mi.get("is_transposed", False))
            if t == "conv_out":
                prune_conv_out_sd(
                    name, source_sd=src, target_sd=new_sd,
                    groups=groups, pr=act.get("pr"), keep_idx=act.get("keep_idx"),
                    is_transposed=isT
                )
            else:
                prune_conv_in_sd(
                    name, source_sd=src, target_sd=new_sd,
                    groups=groups, pr=act.get("pr"), keep_in_idx=act.get("keep_in_idx"),
                    is_transposed=isT
                )

        elif t == "bn":
            prune_batchnorm_sd(name, keep_idx=act["keep_idx"], source_sd=src, target_sd=new_sd)

        elif t == "affine_norm":
            prune_affine_norm_sd(name, keep_idx=act["keep_idx"], source_sd=src, target_sd=new_sd)

        elif t == "linear_out":
            prune_linear_out_sd(name, pr=act.get("pr"), keep_idx=act.get("keep_idx"),
                                source_sd=src, target_sd=new_sd)

        elif t == "linear_in":
            prune_linear_in_sd(name, keep_in_idx=act["keep_in_idx"],
                               source_sd=src, target_sd=new_sd)
        else:
            raise ValueError(f"Unknown action type: {t}")

    return new_sd

# ==========================================
# ---- Top-level convenience function   ----
# ==========================================

def prune_and_save(
    model: nn.Module,
    original_state_dict: SDict,
    plan: List[Dict[str, Any]],
    pruned_dict_path: str,
    pruned_model_path: Optional[str] = None,
) -> Tuple[SDict, nn.Module]:
    """
    1) Applies pruning plan to state_dict (dependency-aware & group-safe)
    2) Morphs model modules to new shapes
    3) Loads pruned state_dict into morphed model (strict)
    4) Saves artifacts
    Returns: (new_state_dict, pruned_model)
    """
    # 1) prune sd
    new_sd = apply_prune_plan_on_state_dict(model, original_state_dict, plan)

    # 2) morph model
    pruned_model = morph_model_to_state_dict(model, new_sd)

    # 3) load strictly
    missing, unexpected = pruned_model.load_state_dict(new_sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State_dict load mismatch.\nMissing: {missing}\nUnexpected: {unexpected}")

    # 4) save
    torch.save(new_sd, pruned_dict_path)
    print(f"[save] pruned state_dict -> {pruned_dict_path}")
    if pruned_model_path is not None:
        torch.save(pruned_model, pruned_model_path)
        print(f"[save] pruned model (full object) -> {pruned_model_path}")

    return new_sd, pruned_model
