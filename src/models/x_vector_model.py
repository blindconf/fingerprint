import torch
import torch.nn as nn
import torch.nn.functional as F

class TDNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, context_size, dilation=1):
        super(TDNNLayer, self).__init__()
        self.tdnn = nn.Conv1d(
            in_dim, out_dim, kernel_size=context_size, dilation=dilation, padding=0
        )
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        x = self.tdnn(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

class XVector(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(XVector, self).__init__()

        # Frame-level layers
        self.frame1 = TDNNLayer(input_dim, 512, context_size=5, dilation=1)
        self.frame2 = TDNNLayer(512, 512, context_size=3, dilation=2)
        self.frame3 = TDNNLayer(512, 512, context_size=3, dilation=3)
        self.frame4 = TDNNLayer(512, 512, context_size=1, dilation=1)
        self.frame5 = TDNNLayer(512, 1500, context_size=1, dilation=1)

        # Segment-level layers
        self.segment6 = nn.Linear(3000, 512)
        self.segment7 = nn.Linear(512, 512)
        self.softmax = nn.Linear(512, num_classes)

    def forward(self, x):
        # Frame-level layers
        
        print(f'shape: {x.shape}')
        x = self.frame1(x)
        x = self.frame2(x)
        x = self.frame3(x)
        x = self.frame4(x)
        x = self.frame5(x)

        # Statistical pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        stats = torch.cat([mean, std], dim=1)

        # Segment-level layers
        x = self.segment6(stats)
        x = F.relu(x)
        x = self.segment7(x)
        x = F.relu(x)

        # Softmax layer
        x = self.softmax(x)
        return x, None

# Instantiate the model
def create_xvector(input_dim=60, num_classes=7):

    return XVector(input_dim=input_dim, num_classes=num_classes)
