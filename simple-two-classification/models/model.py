import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


class MODEL(nn.Module):

  def __init__(self):
    self.inplanes = 64
    super(MODEL, self).__init__()

    self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
      )
    self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
      )

    self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
      )

    self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2), stride=(2, 2))
      )	

    self.fc = nn.Linear(3136, 2)
    



  def forward(self, x):
    c,_,_,_ = x.shape
    # print(x.shape)
    x = self.layer1(x)
    # print(x.shape)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)
    # print('x.shape', x.shape)
    x = x.view(c, -1)
    # print(x.shape)
    x = self.fc(x)

    return x
   




# class ResidualBlock(nn.Module):
    
#     '''
#     实现子module: Residual Block
#     '''
    
#     def __init__(self,inchannel,outchannel,stride=1,shortcut=None):
        
#         super(ResidualBlock,self).__init__()
        
#         self.left=nn.Sequential(
#             nn.Conv2d(inchannel,outchannel,3,stride,1,bias=False),
#             nn.BatchNorm2d(outchannel),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(outchannel,outchannel,3,1,1,bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         self.right=shortcut
    
#     def forward(self,x):
        
#         out=self.left(x)
#         residual=x if self.right is None else self.right(x)
#         out+=residual
#         return F.relu(out)
    
# class MODEL(nn.Module):

#     def __init__(self,num_classes=1000):
        
#         super(MODEL,self).__init__()
        
#         # 前几层图像转换
#         self.pre=nn.Sequential(
#             nn.Conv2d(3,64,7,2,3,bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(3,2,1)
#         )
        
#         # 重复的layer，分别有3，4，6，3个residual block
#         self.layer1=self._make_layer(64,64,3)
#         self.layer2=self._make_layer(64,128,4,stride=2)
#         self.layer3=self._make_layer(128,256,6,stride=2)
#         self.layer4=self._make_layer(256,512,3,stride=2)
        
#         #分类用的全连接
#         self.fc=nn.Linear(512,num_classes)
    
#     def _make_layer(self,inchannel,outchannel,bloch_num,stride=1):

#         shortcut=nn.Sequential(
#             nn.Conv2d(inchannel,outchannel,1,stride,bias=False),
#             nn.BatchNorm2d(outchannel)
#         )
#         layers=[]
#         layers.append(ResidualBlock(inchannel,outchannel,stride,shortcut))
#         for i in range(1,bloch_num):
#             layers.append(ResidualBlock(outchannel,outchannel))
#         return nn.Sequential(*layers)
    
#     def forward(self,x):
        
#         x=self.pre(x)
        
#         x=self.layer1(x)
#         x=self.layer2(x)
#         x=self.layer3(x)
#         x=self.layer4(x)

#         print('x.shape', x.shape)
        
#         x=F.avg_pool2d(x,7)
#         x=x.view(x.size(0),-1)
#         return self.fc(x)