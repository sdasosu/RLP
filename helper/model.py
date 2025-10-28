import torch
import torch.nn as nn

class TinyModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(TinyModel, self).__init__()
        
        # Block 1: Conv -> BN -> ReLU -> MaxPool
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 2: Conv -> BN -> ReLU -> MaxPool
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 3: Conv -> BN -> ReLU -> MaxPool
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Block 4: Conv -> BN -> ReLU
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        # Block 5: Conv -> BN -> ReLU
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        
        # Global Average Pooling + Fully Connected
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Example usage
if __name__ == "__main__":
    model = TinyModel(in_channels=3, num_classes=10)
    #print(model)
    
    # Test with random input (batch_size=4, channels=3, height=224, width=224)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print(f"\nOutput shape: {output.shape}")  # Should be [4, 10]