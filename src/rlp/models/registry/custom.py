"""Custom lightweight model definitions exposed via the registry."""

from __future__ import annotations

import torch.nn as nn


class TinyNet(nn.Module):
    """Small CNN tailored for quick experiments."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        c1, c2, c3, c4, c5, c6, c7 = 32, 64, 128, 128, 256, 256, 384

        self.conv1 = nn.Conv2d(in_channels, c1, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        self.dw3 = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)
        self.pw3 = nn.Conv2d(c2, c3, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(c3, c4, 3, 1, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(c4)
        self.relu4 = nn.ReLU(inplace=True)

        self.conv5 = nn.Conv2d(c4, c5, 3, 1, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(c5)
        self.relu5 = nn.ReLU(inplace=True)

        self.conv6 = nn.Conv2d(c5, c6, 3, 1, 1, bias=False)
        self.bn6 = nn.BatchNorm2d(c6)
        self.relu6 = nn.ReLU(inplace=True)

        self.conv7 = nn.Conv2d(c6, c7, 3, 1, 1, bias=False)
        self.bn7 = nn.BatchNorm2d(c7)
        self.relu7 = nn.ReLU(inplace=True)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c7, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x):  # noqa: D401 - inherit docstring from nn.Module
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.pw3(self.dw3(x)))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))
        x = self.relu7(self.bn7(self.conv7(x)))
        x = self.gap(x).flatten(1)
        return self.fc(x)


def tinynet(in_channels: int, num_classes: int, input_size=None):
    return TinyNet(in_channels=in_channels, num_classes=num_classes)


class TinyResnet(nn.Module):
    """Moderate-sized CNN with residual connections."""

    def __init__(self, in_channels: int = 3, num_classes: int = 73):
        super().__init__()

        self.layer1 = nn.Conv2d(
            in_channels=in_channels, out_channels=61, kernel_size=3, padding=1
        )
        self.layer2 = nn.Conv2d(
            in_channels=61, out_channels=73, kernel_size=3, padding=1
        )
        self.layer3 = nn.Conv2d(
            in_channels=73, out_channels=61, kernel_size=3, padding=1
        )

        self.layer4 = nn.Conv2d(
            in_channels=61, out_channels=85, kernel_size=3, padding=0
        )
        self.layer5 = nn.Conv2d(
            in_channels=85, out_channels=97, kernel_size=3, padding=0
        )
        self.bn1 = nn.BatchNorm2d(97)
        self.mp1 = nn.MaxPool2d(2, 2)

        self.layer6 = nn.Conv2d(
            in_channels=97, out_channels=101, kernel_size=3, padding=1
        )
        self.layer7 = nn.Conv2d(
            in_channels=101, out_channels=113, kernel_size=3, padding=1
        )
        self.layer8 = nn.Conv2d(
            in_channels=113, out_channels=101, kernel_size=3, padding=1
        )

        self.layer9 = nn.Conv2d(
            in_channels=101, out_channels=127, kernel_size=3, padding=0
        )
        self.layer10 = nn.Conv2d(
            in_channels=127, out_channels=131, kernel_size=3, padding=0
        )
        self.bn2 = nn.BatchNorm2d(131)
        self.mp2 = nn.MaxPool2d(2, 2)

        final_spatial_dim = 53
        final_channels = 131
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                final_channels * final_spatial_dim * final_spatial_dim, num_classes
            ),
        )

    def forward(self, x):  # noqa: D401 - inherit docstring from nn.Module
        out = self.layer1(x)
        res1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = out + res1
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.bn1(out)
        out = self.mp1(out)

        out = self.layer6(out)
        res2 = out
        out = self.layer7(out)
        out = self.layer8(out)
        out = out + res2
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.bn2(out)
        out = self.mp2(out)

        return self.classifier(out)


def tinyresnet(in_channels: int, num_classes: int, input_size=None):
    return TinyResnet(in_channels=in_channels, num_classes=num_classes)


__all__ = ["tinynet", "tinyresnet"]
