import torch
import torch.nn as nn
from torchvision import models


class our_CNN(nn.Module):
    def __init__(self):
        super(our_CNN, self).__init__()
        self.VGG = models.vgg16(weights = 'VGG16_Weights.IMAGENET1K_V1')
        for param in self.VGG.parameters():
            param.requires_grad = False
        self.VGG.classifier = torch.nn.Sequential(
            nn.Linear(25088, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, input):
        return self.VGG(input)
