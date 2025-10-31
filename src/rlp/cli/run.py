"""Command-line entrypoints for running pruning experiments."""

import json
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

import torch
import torch.nn as nn

from rlp.config.loader import load_yaml_config
from rlp.models import registry as model_registry
from rlp.data import get_data
from rlp.training import _test, _training
from rlp.pruning import rules as PR, PruningGroups, build_depgraph
from rlp.utils import summary, set_seed, resolve_device


def _ensure_shape(shape: Iterable) -> tuple:
    try:
        dims = tuple(int(dim) for dim in shape)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "'model.example_input.shape' must be an iterable of integers"
        ) from exc
    if len(dims) == 0:
        raise ValueError(
            "'model.example_input.shape' must contain at least one dimension"
        )
    return dims


def _replace_module(root: nn.Module, path: str, new_module: nn.Module) -> None:
    parent = root
    parts = path.split(".")
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new_module
    else:
        setattr(parent, last, new_module)


def _run_with_shape_hooks(
    model: nn.Module, example: torch.Tensor, targets: Dict[str, nn.Module]
) -> Tuple[bool, RuntimeError | None, Dict[str, Tuple[int, ...]], List[str]]:
    recorded: Dict[str, Tuple[int, ...]] = {}
    call_order: List[str] = []
    hooks = []

    for name, module in targets.items():

        def _hook(mod, inputs, name=name):
            tensor = inputs[0]
            if isinstance(tensor, torch.Tensor):
                recorded[name] = tuple(tensor.shape)
                call_order.append(name)

        hooks.append(module.register_forward_pre_hook(_hook))

    try:
        model(example)
        return True, None, recorded, call_order
    except RuntimeError as exc:
        return False, exc, recorded, call_order
    finally:
        for hook in hooks:
            hook.remove()


def _reinit_linear(module: nn.Linear, in_features: int) -> nn.Linear:
    new_linear = nn.Linear(
        in_features, module.out_features, bias=module.bias is not None
    )
    nn.init.kaiming_normal_(new_linear.weight)
    if new_linear.bias is not None:
        nn.init.zeros_(new_linear.bias)
    return new_linear.to(module.weight.device, dtype=module.weight.dtype)


def _reinit_conv(module: nn.Module, in_channels: int) -> nn.Module:
    kwargs = {
        "kernel_size": module.kernel_size,
        "stride": module.stride,
        "padding": module.padding,
        "dilation": module.dilation,
        "bias": module.bias is not None,
        "padding_mode": getattr(module, "padding_mode", "zeros"),
    }
    groups = module.groups
    if groups == module.in_channels:
        groups = in_channels
    elif in_channels % groups != 0:
        groups = 1
    kwargs["groups"] = groups

    if isinstance(module, nn.Conv1d):
        new_conv = nn.Conv1d(in_channels, module.out_channels, **kwargs)
    elif isinstance(module, nn.Conv2d):
        new_conv = nn.Conv2d(in_channels, module.out_channels, **kwargs)
    elif isinstance(module, nn.Conv3d):
        new_conv = nn.Conv3d(in_channels, module.out_channels, **kwargs)
    else:
        raise TypeError("Unsupported convolution module for reinitialization.")

    return new_conv.to(module.weight.device, dtype=module.weight.dtype)


def _ensure_forward_ready(
    model: nn.Module, example: torch.Tensor, num_classes: int
) -> None:
    targets = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d))
    }

    while True:
        success, exc, recorded, order = _run_with_shape_hooks(model, example, targets)
        if success:
            return
        if not order:
            raise exc  # unable to diagnose
        failing_name = order[-1]
        module = targets[failing_name]
        shape = recorded.get(failing_name)
        if shape is None:
            raise exc

        if isinstance(module, nn.Linear):
            in_features = shape[1] if len(shape) > 1 else shape[0]
            new_linear = _reinit_linear(module, in_features)
            _replace_module(model, failing_name, new_linear)
            targets[failing_name] = new_linear
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            in_channels = shape[1] if len(shape) > 1 else module.in_channels
            new_conv = _reinit_conv(module, in_channels)
            _replace_module(model, failing_name, new_conv)
            targets[failing_name] = new_conv
        else:
            raise exc

        if isinstance(module, nn.Linear) and module.out_features == num_classes:
            continue


def _build_model(model_cfg: Dict[str, Any], example_input_shape: tuple):
    model_name = model_cfg.get("name", "resnet18")
    in_channels = model_cfg.get("in_channels", 3)
    num_classes = model_cfg.get("num_classes", 10)

    try:
        ctor = getattr(model_registry, model_name)
    except AttributeError as exc:
        raise ValueError(f"Unknown model name '{model_name}' in configuration") from exc

    spatial = example_input_shape[1:] if len(example_input_shape) >= 3 else None
    model = ctor(in_channels=in_channels, num_classes=num_classes, input_size=spatial)
    return model, model_name


def run_experiment(config_path: str) -> Dict[str, Any]:
    config = load_yaml_config(config_path)

    experiment_cfg = config.get("experiment", {})
    set_seed(int(experiment_cfg.get("seed", 42)))
    device = resolve_device(experiment_cfg.get("device", "auto"))

    model_cfg = config.get("model", {})
    example_input_shape = _ensure_shape(
        model_cfg.get("example_input", {}).get("shape", [1, 3, 32, 32])
    )

    model_, model_name = _build_model(model_cfg, example_input_shape)
    num_classes = int(model_cfg.get("num_classes", 10))

    paths_cfg = config.get("paths", {})
    checkpoint_dir = Path(
        paths_cfg.get("checkpoint_dir", "./assets/checkpoint/baselines")
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpt = checkpoint_dir / paths_cfg.get(
        "baseline_checkpoint", f"{model_name}_cifar10_best.pt"
    )

    pruned_dir = Path(paths_cfg.get("pruned_dir", "./assets/checkpoint/pruned_dict"))
    pruned_dir.mkdir(parents=True, exist_ok=True)
    pruned_ckpt = pruned_dir / paths_cfg.get(
        "pruned_checkpoint", f"{model_name}_cifar10_pruned.pt"
    )

    export_dir = Path(paths_cfg.get("export_dir", "./assets/pruning_maps"))
    export_dir.mkdir(parents=True, exist_ok=True)
    pruning_map_json = export_dir / paths_cfg.get("pruning_map", "pruning_map.json")
    keep_map_json = export_dir / paths_cfg.get("keep_map", "keep_map.json")

    data_cfg = config.get("data", {})
    train_loader, test_loader, cutmix_or_mixup = get_data(
        batch_size=int(data_cfg.get("batch_size", 128)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        num_classes=num_classes,
        augmentations=data_cfg.get("augmentations"),
    )

    summary_depth = int(model_cfg.get("summary_depth", 1))
    summary_input = torch.randn(*example_input_shape, device="cpu")
    summary(model_, input=summary_input, depth=summary_depth, device="cpu")

    x_demo = torch.randn(*example_input_shape, device="cpu")
    model_.eval().to("cpu")
    dg = build_depgraph(model_, example_input=x_demo)
    pg = PruningGroups(dg, model_)
    pg.build()
    pg.print_groups(verbose=True)

    with pruning_map_json.open("w", encoding="utf-8") as handle:
        json.dump(pg.export_for_rl(), handle, indent=2)

    pruning_cfg = config.get("pruning", {})
    keep_ratio = float(pruning_cfg.get("keep_ratio", 0.5))
    if not 0.0 < keep_ratio <= 1.0:
        raise ValueError("'pruning.keep_ratio' must be in the range (0, 1].")

    decisions = {}
    for g in pg.groups:
        k = int(g["base_channels"] * keep_ratio)
        if g["divisibility"] > 1:
            k = (k // g["divisibility"]) * g["divisibility"]
        k = max(g["min_channels"], min(k, g["max_channels"]))
        if g["divisibility"] > 1 and (k % g["divisibility"] != 0):
            k = ((k // g["divisibility"]) + 1) * g["divisibility"]
            k = min(k, g["max_channels"])
        decisions[g["group_id"]] = k

    decisions = pg.synchronize_decisions(decisions)

    preserve_linear_threshold = int(pruning_cfg.get("preserve_large_linear", 1024))
    reinit_large_linear = bool(pruning_cfg.get("reinit_large_linear", False))
    if preserve_linear_threshold > 0:
        for g in pg.groups:
            if (
                g["producer_type"] == "Linear"
                and g["base_channels"] >= preserve_linear_threshold
            ):
                decisions[g["group_id"]] = g["max_channels"]

    for g in pg.groups:
        if g["producer_type"] == "Linear" and g["base_channels"] == num_classes:
            decisions[g["group_id"]] = g["max_channels"]

    keep_map = PR.apply_pruning(model_, dg, pg, decisions)

    if reinit_large_linear and preserve_linear_threshold > 0:
        for mod in model_.modules():
            if (
                isinstance(mod, nn.Linear)
                and mod.out_features >= preserve_linear_threshold
            ):
                nn.init.kaiming_normal_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    with keep_map_json.open("w", encoding="utf-8") as handle:
        json.dump(
            {int(k): [int(i) for i in v] for k, v in keep_map.items()}, handle, indent=2
        )

    def _nparams(module: nn.Module) -> int:
        return sum(p.numel() for p in module.parameters())

    print(f"[controller] pruned params: {_nparams(model_)}")
    summary(
        model_,
        input=torch.randn(*example_input_shape, device="cpu"),
        depth=summary_depth,
        device="cpu",
    )

    example_cpu = torch.randn(*example_input_shape, device="cpu")
    _ensure_forward_ready(model_, example_cpu, num_classes)

    training_cfg = config.get("training", {})
    epochs = int(training_cfg.get("epochs", 5))
    lr = float(training_cfg.get("lr", 0.001))

    model_.to(device)
    _training(
        model=model_,
        train_loader=train_loader,
        test_loader=test_loader,
        cutmix_or_mixup=cutmix_or_mixup,
        epochs=epochs,
        lr=lr,
        device=device,
        state_dict_path=str(pruned_ckpt),
        dataname=training_cfg.get("dataname"),
    )

    test_acc = _test(model_, test_loader, device=device)
    print("\n[controller] pruned training finished")
    print(f"[controller] test_acc={float(test_acc):.4f}")
    print(f"[controller] pruned state_dict saved to: {pruned_ckpt}")

    return {
        "test_acc": float(test_acc),
        "pruned_checkpoint": str(pruned_ckpt),
        "keep_map": str(keep_map_json),
        "pruning_map": str(pruning_map_json),
        "baseline_checkpoint": str(baseline_ckpt),
    }
