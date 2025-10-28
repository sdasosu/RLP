import torch.nn as nn
# THE FIX IS HERE: Use an alias to avoid name collision.
from torchvision import models as torchvision_models

def _check_in_channels(model_name, in_channels):
    if in_channels != 3:
        print(f"Warning: torchvision model '{model_name}' is designed for 3 input channels. The 'in_channels={in_channels}' argument will be ignored.")




# ================= VGG FAMILY =================
def vgg11(in_channels, num_classes):
    _check_in_channels('vgg11', in_channels)
    model = torchvision_models.vgg11(weights=torchvision_models.VGG11_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def vgg11_bn(in_channels, num_classes):
    _check_in_channels('vgg11_bn', in_channels)
    model = torchvision_models.vgg11_bn(weights=torchvision_models.VGG11_BN_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def vgg16(in_channels, num_classes):
    _check_in_channels('vgg16', in_channels)
    model = torchvision_models.vgg16(weights=torchvision_models.VGG16_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
    
def vgg16_bn(in_channels, num_classes):
    _check_in_channels('vgg16_bn', in_channels)
    model = torchvision_models.vgg16_bn(weights=torchvision_models.VGG16_BN_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def vgg19(in_channels, num_classes):
    _check_in_channels('vgg19', in_channels)
    model = torchvision_models.vgg19(weights=torchvision_models.VGG19_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model

def vgg19_bn(in_channels, num_classes):
    _check_in_channels('vgg19_bn', in_channels)
    model = torchvision_models.vgg19_bn(weights=torchvision_models.VGG19_BN_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
# ==============================================







#================ RESNET FALILY =================
def resnet18(in_channels, num_classes):
    _check_in_channels('resnet18', in_channels)
    model = torchvision_models.resnet18(weights=torchvision_models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet34(in_channels, num_classes):
    _check_in_channels('resnet34', in_channels)
    model = torchvision_models.resnet34(weights=torchvision_models.ResNet34_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet50(in_channels, num_classes):
    _check_in_channels('resnet50', in_channels)
    model = torchvision_models.resnet50(weights=torchvision_models.ResNet50_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnet101(in_channels, num_classes):
    _check_in_channels('resnet101', in_channels)
    model = torchvision_models.resnet101(weights=torchvision_models.ResNet101_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
    
def resnet152(in_channels, num_classes):
    _check_in_channels('resnet152', in_channels)
    model = torchvision_models.resnet152(weights=torchvision_models.ResNet152_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
# ==================================================





    
#================= MOBILENET FAMILY ================
def mobilenet_v3_small(in_channels, num_classes):
    _check_in_channels('mobilenet_v3_small', in_channels)
    model = torchvision_models.mobilenet_v3_small(weights=torchvision_models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
    
def mobilenet_v3_large(in_channels, num_classes):
    _check_in_channels('mobilenet_v3_large', in_channels)
    model = torchvision_models.mobilenet_v3_large(weights=torchvision_models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model
# ==================================================





#===================== DenseNet ====================
def densenet121(in_channels, num_classes):
    _check_in_channels('densenet121', in_channels)
    model = torchvision_models.densenet121(weights=torchvision_models.DenseNet121_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def densenet161(in_channels, num_classes):
    _check_in_channels('densenet161', in_channels)
    model = torchvision_models.densenet161(weights=torchvision_models.DenseNet161_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def densenet169(in_channels, num_classes):
    _check_in_channels('densenet169', in_channels)
    model = torchvision_models.densenet169(weights=torchvision_models.DenseNet169_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model

def densenet201(in_channels, num_classes):
    _check_in_channels('densenet201', in_channels)
    model = torchvision_models.densenet201(weights=torchvision_models.DenseNet201_Weights.IMAGENET1K_V1)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)
    return model
# ==================================================





#==================== GoogLeNet ====================
def googlenet(in_channels, num_classes):
    _check_in_channels('googlenet', in_channels)
    model = torchvision_models.googlenet(weights=torchvision_models.GoogLeNet_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
# ==================================================






#===================== ResNext ======================
def resnext50_32x4d(in_channels, num_classes):
    _check_in_channels('resnext50_32x4d', in_channels)
    model = torchvision_models.resnext50_32x4d(weights=torchvision_models.ResNeXt50_32X4D_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def resnext101_32x8d(in_channels, num_classes):
    _check_in_channels('resnext101_32x8d', in_channels)
    model = torchvision_models.resnext101_32x8d(weights=torchvision_models.ResNeXt101_32X8D_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
# ==================================================


# ===================== TinyNet (expanded: +2 conv layers) =====================
class TinyNet(nn.Module):
    """
    Small CNN:
      Stem:      Conv-BN-ReLU
      Block 2:   Conv-BN-ReLU + MaxPool
      Block 3:   Depthwise(3x3)-Pointwise(1x1)-BN-ReLU + MaxPool
      Block 4:   Conv-BN-ReLU
      Block 5:   Conv-BN-ReLU
      Block 6:   Conv-BN-ReLU            <-- NEW
      Block 7:   Conv-BN-ReLU            <-- NEW
      Head:      GAP -> Linear
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        c1, c2, c3, c4, c5, c6, c7 = 32, 64, 128, 128, 256, 256, 384

        # Stem
        self.conv1 = nn.Conv2d(in_channels, c1, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(c1)
        self.relu1 = nn.ReLU(inplace=True)

        # Block 2
        self.conv2 = nn.Conv2d(c1, c2, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(c2)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)  # 32->16 for CIFAR

        # Block 3 (Depthwise separable)
        self.dw3   = nn.Conv2d(c2, c2, 3, 1, 1, groups=c2, bias=False)  # depthwise
        self.pw3   = nn.Conv2d(c2, c3, 1, 1, 0, bias=False)             # pointwise
        self.bn3   = nn.BatchNorm2d(c3)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2)  # 16->8 for CIFAR

        # Block 4
        self.conv4 = nn.Conv2d(c3, c4, 3, 1, 1, bias=False)
        self.bn4   = nn.BatchNorm2d(c4)
        self.relu4 = nn.ReLU(inplace=True)

        # Block 5
        self.conv5 = nn.Conv2d(c4, c5, 3, 1, 1, bias=False)
        self.bn5   = nn.BatchNorm2d(c5)
        self.relu5 = nn.ReLU(inplace=True)

        # Block 6 (NEW)
        self.conv6 = nn.Conv2d(c5, c6, 3, 1, 1, bias=False)
        self.bn6   = nn.BatchNorm2d(c6)
        self.relu6 = nn.ReLU(inplace=True)

        # Block 7 (NEW)
        self.conv7 = nn.Conv2d(c6, c7, 3, 1, 1, bias=False)
        self.bn7   = nn.BatchNorm2d(c7)
        self.relu7 = nn.ReLU(inplace=True)

        # Head
        self.gap   = nn.AdaptiveAvgPool2d((1, 1))
        self.fc    = nn.Linear(c7, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.pw3(self.dw3(x)))))
        x = self.relu4(self.bn4(self.conv4(x)))
        x = self.relu5(self.bn5(self.conv5(x)))
        x = self.relu6(self.bn6(self.conv6(x)))   # NEW
        x = self.relu7(self.bn7(self.conv7(x)))   # NEW
        x = self.gap(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x


def tinynet(in_channels, num_classes):
    """Factory matching your existing API."""
    return TinyNet(in_channels=in_channels, num_classes=num_classes)
# =========================================================================== 


# ===========================================================================
import torch
import torch.nn as nn
from torch import Tensor

class TinyResnet(nn.Module):
    """
    An extended convolutional neural network model with more layers and
    additional residual connections and downsampling.
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 73):
        super().__init__()
        
        self.layer1 = nn.Conv2d(in_channels=in_channels, out_channels=61, kernel_size=3, padding=1)
        self.layer2 = nn.Conv2d(in_channels=61, out_channels=73, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(in_channels=73, out_channels=61, kernel_size=3, padding=1)
        # Residual add 1
        self.layer4 = nn.Conv2d(in_channels=61, out_channels=85, kernel_size=3, padding=0)
        self.layer5 = nn.Conv2d(in_channels=85, out_channels=97, kernel_size=3, padding=0)
        self.bn1    = nn.BatchNorm2d(97)
        self.mp1    = nn.MaxPool2d(2, 2)

        # Additional layers
        self.layer6 = nn.Conv2d(in_channels=97, out_channels=101, kernel_size=3, padding=1)
        self.layer7 = nn.Conv2d(in_channels=101, out_channels=113, kernel_size=3, padding=1)
        self.layer8 = nn.Conv2d(in_channels=113, out_channels=101, kernel_size=3, padding=1)
        # Residual add 2
        self.layer9 = nn.Conv2d(in_channels=101, out_channels=127, kernel_size=3, padding=0)
        self.layer10= nn.Conv2d(in_channels=127, out_channels=131, kernel_size=3, padding=0)
        self.bn2    = nn.BatchNorm2d(131)
        self.mp2    = nn.MaxPool2d(2, 2)
        final_spatial_dim = 53
        final_channels = 131
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(final_channels * final_spatial_dim * final_spatial_dim, num_classes)
        )

    def forward(self, x):
        # Block 1
        out = self.layer1(x)
        res1 = out
        out = self.layer2(out)
        out = self.layer3(out)
        out = out + res1
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.bn1(out)
        out = self.mp1(out)
        
        # Block 2
        out = self.layer6(out)
        res2 = out
        out = self.layer7(out)
        out = self.layer8(out)
        out = out + res2
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.bn2(out)
        out = self.mp2(out)

        out = self.classifier(out)
        return out

def tinyresnet(in_channels: int, num_classes: int):
    return TinyResnet(in_channels=in_channels, num_classes=num_classes)
# ===========================================================================
