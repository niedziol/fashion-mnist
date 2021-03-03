import torch.nn as nn


class LogReg(nn.Module):
    def __init__(self):
        super(LogReg, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
