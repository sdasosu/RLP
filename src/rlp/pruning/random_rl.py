# pruning_helper/rl_engine.py
import os
import json
import torch
import random

from rlp.pruning.tracer import build_depgraph
from rlp.pruning.groups import PruningGroups
from rlp.pruning import rules as PR


# ============ random score hooks ==============
def _rand_scores_conv2d(m, conv_type):
    return torch.rand(m.weight.size(0))


def _rand_scores_linear(m):
    return torch.rand(m.weight.size(0))


# ============ temp ranker ctx ==============
class _TmpRandomRankers:
    def __enter__(self):
        self._orig_conv = PR._l2_scores_conv2d
        self._orig_lin = PR._l2_scores_linear
        PR._l2_scores_conv2d = _rand_scores_conv2d
        PR._l2_scores_linear = _rand_scores_linear
        return self

    def __exit__(self, exc_type, exc, tb):
        PR._l2_scores_conv2d = self._orig_conv
        PR._l2_scores_linear = self._orig_lin


# ============ core ==============
def run_fake_rl(
    model_,
    train_loader,
    test_loader,
    cutmix_or_mixup,
    checkpoint_path=None,
    sparsity=0.5,
    seed=42,
    device=None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # ------------ trace on cpu ------------
    model_.eval().to("cpu")
    x_demo = torch.randn(1, 3, 32, 32, device="cpu")
    dg = build_depgraph(model_, example_input=x_demo)
    pg = PruningGroups(dg, model_)
    pg.build()

    # ------------ decisions ------------
    decisions = {}
    for g in pg.groups:
        k = int(g["base_channels"] * (1.0 - float(sparsity)))
        if g["divisibility"] > 1:
            k = (k // g["divisibility"]) * g["divisibility"]
        k = max(g["min_channels"], min(k, g["max_channels"]))
        if g["divisibility"] > 1 and (k % g["divisibility"] != 0):
            k = ((k // g["divisibility"]) + 1) * g["divisibility"]
            k = min(k, g["max_channels"])
        decisions[g["group_id"]] = k

    # ------------ apply pruning ------------
    with _TmpRandomRankers():
        keep_map = PR.apply_pruning(model_, dg, pg, decisions)

    # ------------ train + eval ------------
    model_.to(device)
    from rlp.training.engine import _training, _test

    train_result = _training(
        model=model_,
        train_loader=train_loader,
        test_loader=test_loader,
        cutmix_or_mixup=cutmix_or_mixup,
        epochs=2,
        lr=0.001,
        device=device,
        state_dict_path=None,
    )

    if checkpoint_path:
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        torch.save(model_.state_dict(), checkpoint_path)

    acc, loss = _test(model_, test_loader, device=device)

    return {
        "reward": float(acc),
        "test_acc": float(acc),
        "test_loss": float(loss),
        "keep_map": {int(k): [int(i) for i in v] for k, v in keep_map.items()},
        "decisions": decisions,
        "num_groups": pg.export_for_rl()["num_groups"],
        "train_result": train_result,
    }


# ============ quick cli =============
if __name__ == "__main__":
    from rlp.data.datamodule import get_data
    from rlp.models.registry import resnet18
    from rlp.utils import summary

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, test_loader, cutmix_or_mixup = get_data()
    model_ = resnet18(in_channels=3, num_classes=10)
    summary(model_, input=torch.randn(1, 3, 32, 32), depth=1, device="cpu")

    out = run_fake_rl(
        model_,
        train_loader,
        test_loader,
        cutmix_or_mixup,
        checkpoint_path="./assets/checkpoint/baselines/resnet18_cifar10_pruned.pt",
        sparsity=0.5,
        seed=42,
        device=device,
    )
    print(json.dumps(out, indent=2))
