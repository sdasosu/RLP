import os
import csv
import pickle
import networkx as nx
import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

torch.manual_seed(0)
os.makedirs("output", exist_ok=True)
device = "cpu"
dummy = torch.randn(1, 3, 224, 224)

class FakeModel(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super(FakeModel, self).__init__()

        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Batch Normalization and ReLU
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        # Third convolutional layer
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # Initial convolution
        x1 = self.conv1(x)

        # Residual path
        residual = x1

        # Main path
        x2 = self.conv2(x1)
        x2 = self.bn(x2)
        x2 = self.relu(x2)
        x2 = self.conv3(x2)

        # Add the residual connection
        out = x2 + residual

        # Final activation
        out = self.relu(out)

        return out


model = FakeModel().to(device).eval()

trace = fx.symbolic_trace(model)
GRAPH = trace.graph
GRAPH.print_tabular()
ShapeProp(trace).propagate(dummy)

print("\n\n")
for nodes in GRAPH.nodes:
    if nodes.op == "call_module":
        for src in nodes.all_input_nodes:
            pass