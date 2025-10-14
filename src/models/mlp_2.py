import torch
import torch.nn as nn
import torch.nn.functional as F

class ResSEMLP(nn.Module):
    def __init__(self, input_size, num_classes=1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.se1 = SE1D(256)
        self.res1 = ResidualBlock(256)
        self.dropout = nn.Dropout(0.4)
        self.fc_out = nn.Linear(256, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        # x = x.mean(dim=2)  # [54, 65]
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.se1(x)
        x = self.res1(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x, None

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

class SE1D(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C]
        w = F.relu(self.fc1(x))
        w = self.sigmoid(self.fc2(w))
        return x * w

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)