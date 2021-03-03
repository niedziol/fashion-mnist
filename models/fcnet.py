import torch.nn as nn
import torch.nn.functional as F


class FCNet(nn.Module):
    def __init__(self, hidden=64):
        super(FCNet, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x