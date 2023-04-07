from efficientnet_pytorch import EfficientNet
from torch import nn

'''feature = model._fc.in_features
model._fc = nn.Linear(in_features=feature,out_features=45,bias=True)
print(model)'''


import torch
import torch.nn as nn
from torchvision import models


class our_CNN(nn.Module):
    def __init__(self):
        super(our_CNN, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b5')
        # effective net的参数好像没有非常多，那就训练整个模型看看
        #for param in self.model.parameters():
        #    param.requires_grad = False
        
        feature = self.model._fc.in_features    # 2048
        '''
        self.model._fc = torch.nn.Sequential(
            nn.Linear(feature, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        '''
        # 只要一层全连接层试试，有时间的话再换多层的
        self.model._fc = torch.nn.Linear(feature, 1)
        
    def forward(self, input):
        return self.model(input)

if __name__ == '__main__':
    print(xxx)