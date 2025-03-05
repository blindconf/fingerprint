import torch

class LogisticRegression(torch.nn.Module):
    def __init__(self, num_input, num_out):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(num_input, num_out)
    def forward(self, x):
        x = self.linear(x)
        return x