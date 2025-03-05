import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxFeatureMap2D(nn.Module):
    def forward(self, x):
        channels = x.size(1)
        assert channels % 2 == 0
        return torch.max(x[:, :channels // 2, :, :],
                         x[:, channels // 2:, :, :])

class MaxFeatureMap1D(nn.Module):
    def forward(self, x):
        features = x.size(1)
        assert features % 2 == 0
        return torch.max(x[:, :features // 2], x[:, features // 2:])

class LCNNModel(nn.Module):
    def __init__(self, num_classes=2, dropout_p=0.75):
        super(LCNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.mfm1 = MaxFeatureMap2D()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm2 = MaxFeatureMap2D()
        self.bn1   = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1)
        self.mfm3 = MaxFeatureMap2D()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn2   = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, 96, kernel_size=1, stride=1)
        self.mfm4 = MaxFeatureMap2D()
        self.bn3   = nn.BatchNorm2d(48)
        self.conv5 = nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1)
        self.mfm5 = MaxFeatureMap2D()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(64, 128, kernel_size=1, stride=1)
        self.mfm6 = MaxFeatureMap2D()
        self.bn4   = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.mfm7 = MaxFeatureMap2D()
        self.bn5   = nn.BatchNorm2d(32)
        self.conv8 = nn.Conv2d(32, 64, kernel_size=1, stride=1)
        self.mfm8 = MaxFeatureMap2D()
        self.bn6   = nn.BatchNorm2d(32)
        self.conv9 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.mfm9 = MaxFeatureMap2D()
        self.pool4 = nn.AdaptiveMaxPool2d((1, 1))
        self.dropout_p = dropout_p
        self.fc1 = nn.Linear(32, 160)
        self.mfm_fc = MaxFeatureMap1D()
        self.bn_fc = nn.BatchNorm1d(80)
        self.fc2 = nn.Linear(80, num_classes)
        self._initialize_weights()
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.mfm1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.mfm2(x)
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.mfm3(x)
        x = self.pool2(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.mfm4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.mfm5(x)
        x = self.pool3(x)
        x = self.conv6(x)
        x = self.mfm6(x)
        x = self.bn4(x)
        x = self.conv7(x)
        x = self.mfm7(x)
        x = self.bn5(x)
        x = self.conv8(x)
        x = self.mfm8(x)
        x = self.bn6(x)
        x = self.conv9(x)
        x = self.mfm9(x)
        x = self.pool4(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.mfm_fc(x)
        x = self.bn_fc(x)
        logits = self.fc2(x)
        return logits, None

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
