import torch.nn as nn
import torch.nn.functional as F


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear1 = nn.Linear(10, 8)
        self.linear2 = nn.Linear(8, 1)

    def forward(self, input):
        x = F.relu(self.linear1(input))
        x = F.sigmoid(self.linear2(x))
        return x
