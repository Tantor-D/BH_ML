import torch
import torch.nn as nn

class our_CNN(nn.Module):
    def __init__(self):
        super(our_CNN, self).__init__()
        self.ch = 16  # 超参数，可以自己调整
        self.cnn = nn.Sequential(
            nn.Conv2d(3, self.ch, 4, 2, 1),  # 4*128*128 -> 16*64*64
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Conv2d(self.ch, 2 * self.ch, 4, 2, 1),  # 16*64*64 -> 32*32*32
            nn.BatchNorm2d(2 * self.ch),
            nn.ReLU(),
            nn.Conv2d(2 * self.ch, 2 * self.ch, 4, 2, 1),  # 32*32*32 -> 32*16*16
            nn.BatchNorm2d(2 * self.ch),
            nn.ReLU(),
            nn.Conv2d(2 * self.ch, self.ch, 4, 2, 1),  # 32*16*16 -> 16*8*8 此时感受野为67*67
            nn.BatchNorm2d(self.ch),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
            # nn.ReLU()      这一层不清楚到底需不需要
        )

    def forward(self, input):
        return self.cnn(input)
