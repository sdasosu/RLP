# controller.py
import os
import json
import torch

from helper.data import get_data
from helper.summary import summary
from helper.model_zoo import *
from helper.train_test import _training, _test
from helper.FakeNet import FakeModel

from pruning_helper.tracer import build_depgraph
from pruning_helper.group_creator import PruningGroups
from pruning_helper import pruning_rule as PR

#============ setup ==============
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_demo = torch.randn(1, 3, 32, 32, device='cpu')

#============ paths ==============
checkpoint_dir = './assets/checkpoint/baselines'
os.makedirs(checkpoint_dir, exist_ok=True)
baseline_ckpt = os.path.join(checkpoint_dir, 'DenseNet121_cifar10_best.pt')

pruned_dir = './assets/checkpoint/pruned_dict'
os.makedirs(pruned_dir, exist_ok=True)
pruned_ckpt = os.path.join(pruned_dir, 'resnet18_cifar10_pruned.pt')

export_dir = './assets/pruning_maps'
os.makedirs(export_dir, exist_ok=True)
pruning_map_json = os.path.join(export_dir, 'pruning_map.json')
keep_map_json = os.path.join(export_dir, 'keep_map.json')

#============ data + model ==============
train_loader, test_loader, cutmix_or_mixup = get_data()
model_ = resnet18(in_channels=3, num_classes=10)
summary(model_, input=torch.randn(1, 3, 32, 32), depth=1, device='cpu')

#============ trace (cpu) ==============
model_.eval().to('cpu')
dg = build_depgraph(model_, example_input=x_demo)
print(f"nodes: {len(dg.nodes)}")
dg.print_graph(80)

#============ pruning groups (RL map) ==============
pg = PruningGroups(dg, model_)
pg.build()
pg.print_groups(verbose=True)

with open(pruning_map_json, 'w') as f:
    json.dump(pg.export_for_rl(), f, indent=2)
print(f"[controller] pruning map saved to: {pruning_map_json}")

#============ decisions (50% keep) ==============
decisions = {}
for g in pg.groups:
    k = int(g['base_channels'] * 0.5)
    if g['divisibility'] > 1:
        k = (k // g['divisibility']) * g['divisibility']
    k = max(g['min_channels'], min(k, g['max_channels']))
    if g['divisibility'] > 1 and (k % g['divisibility'] != 0):
        k = ((k // g['divisibility']) + 1) * g['divisibility']
        k = min(k, g['max_channels'])
    decisions[g['group_id']] = k

#============ apply pruning (cpu) ==============
keep_map = PR.apply_pruning(model_, dg, pg, decisions)
with open(keep_map_json, 'w') as f:
    json.dump({int(k): [int(i) for i in v] for k, v in keep_map.items()}, f, indent=2)
print(f"[controller] keep_map saved to: {keep_map_json}")



#============ pruned summary (cpu) ==============
def _nparams(m): return sum(p.numel() for p in m.parameters())
print(f"[controller] pruned params: {_nparams(model_)}")
summary(model_, input=torch.randn(1, 3, 32, 32), depth=1, device='cpu')

#============ train pruned model (5 epochs) ==============
model_.to(device)
train_result = _training(
    model=model_,
    train_loader=train_loader,
    test_loader=test_loader,
    cutmix_or_mixup=cutmix_or_mixup,
    epochs=5,
    lr=0.001,
    device=device,
    state_dict_path=pruned_ckpt )

#============ evaluate ==============
test_acc = _test(model_, test_loader, device=device)
print("\n[controller] pruned training finished")
print(f"[controller] test_acc={float(test_acc):.4f}")
print(f"[controller] pruned state_dict saved to: {pruned_ckpt}")
