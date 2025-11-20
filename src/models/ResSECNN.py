import torch
import torch.nn as nn
import torch.nn.functional as F
'''
class ResSECNN(nn.Module):
    """A standard convolutional neural network for spectrograms."""
    def __init__(self, num_classes=7):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ by 2

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ by 2

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # ↓ by 2
        )

        # Adaptive Pooling lets you use variable-sized spectrograms
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [B, 1, Freq, Time]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

'''
class SE2D(nn.Module):
    """Squeeze-and-Excitation block for 2D feature maps."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        b, c, _, _ = x.size()
        w = self.pool(x).view(b, c)
        w = F.relu(self.fc1(w))
        w = self.sigmoid(self.fc2(w)).view(b, c, 1, 1)
        return x * w


class ResidualConvBlock(nn.Module):
    """A simple residual block with Conv2d + BatchNorm + ReLU."""
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

        self.downsample = None
        if in_channels != out_channels or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return F.relu(out)


class ResSECNN(nn.Module):
    """ResSE-CNN model for spectrograms."""
    def __init__(self, num_classes=7):
        super().__init__()

        # Feature extractor
        self.layer1 = ResidualConvBlock(1, 32, stride=2)
        self.se1 = SE2D(32)
        self.layer2 = ResidualConvBlock(32, 64, stride=2)
        self.se2 = SE2D(64)
        self.layer3 = ResidualConvBlock(64, 128, stride=2)
        self.se3 = SE2D(128)

        # Adaptive pooling handles variable spectrogram lengths
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

        self._initialize_weights()

    def forward(self, x):
        # x: [B, 1, Freq, Time]
        # x = x.unsqueeze(1)
        x = self.layer1(x)
        x = self.se1(x)
        x = self.layer2(x)
        x = self.se2(x)
        x = self.layer3(x)
        x = self.se3(x)

        x = self.avgpool(x)        # [B, 128, 1, 1]
        x = x.view(x.size(0), -1)  # [B, 128]
        x = self.fc(x)
        return x, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
