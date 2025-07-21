import torch
import torch.nn as n
import numpy as np
from torch.nn import functional as f
from torchmetrics import Accuracy
from tqdm import tqdm


class Res_basic(n.Module):
    
    def __init__(self, input_chan, number_chan, strides=1, use1x1conv = False):
        super().__init__()
        # 3×3
        self.conv1 = n.Conv2d(input_chan, number_chan, kernel_size=3,stride=strides, padding=1 )
        self.bn1   = n.BatchNorm2d(number_chan)
        # 3×3
        self.conv2 = n.Conv2d(number_chan, number_chan, kernel_size=3,stride=1, padding=1 )
        self.bn2   = n.BatchNorm2d(number_chan)
        if use1x1conv:
            self.conv1x1 = n.Conv2d(input_chan, number_chan, kernel_size=1,stride=strides, bias=False)
        else:
            self.conv1x1 = None
    def forward(self, x):
        identity = x
        out = f.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.conv1x1 : #reshape
            identity = self.conv1x1(x)
        out += identity
        return f.relu(out)

class Bottleneck(n.Module):
    """ResNet-50 """
    expansion = 4          # out_channels = planes * expansion

    def __init__(self, in_channels, planes, stride=1):
        super().__init__()
        width = planes      

        # 1×1 c/4
        self.conv1 = n.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1   = n.BatchNorm2d(width)
        # 3×3 (downsampling)
        self.conv2 = n.Conv2d(width, width, kernel_size=3, stride=stride,padding=1, bias=False)
        self.bn2   = n.BatchNorm2d(width)
        # 1×1 c
        self.conv3 = n.Conv2d(width, planes * self.expansion,kernel_size=1, bias=False)
        self.bn3   = n.BatchNorm2d(planes * self.expansion)
        self.relu  = n.ReLU(inplace=True)
        
        if stride !=1 or in_channels != planes*self.expansion:
            self.downsample = n.Sequential(
                n.Conv2d(in_channels, planes*self.expansion,
                          kernel_size=1, stride=stride, bias=False),
                n.BatchNorm2d(planes*self.expansion),
            )
        else:
            self.downsample = n.Identity()
        

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)




class Accumulator:
    """Accumulate values in n variables. acc[i] indexes the accumulated result of each variable。"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        """Add each value to the corresponding slot of data"""
        if len(args) != len(self.data):
            raise ValueError(f"Expected {len(self.data)} values, got {len(args)}")
        for i, v in enumerate(args):
            self.data[i] += float(v)

    def reset(self):
        """reset all accumulated values to zero"""
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def __repr__(self):
        return f"Accumulator({self.data})"
    
 
def res_net(input_chan, number_chan, number_block, havefirst =False):# only onetimes downsampling
    blocks = []
    for i in range(number_block):
        
        if i == 0 :
            stride =1 if havefirst else 2
            use1x1 = (input_chan != number_chan) or (stride != 1)
            blocks.append(Res_basic(input_chan, number_chan, strides=stride, use1x1conv=use1x1))
        else: 
            blocks.append(Res_basic(number_chan, number_chan,strides=1,use1x1conv=False))
    return blocks

def resnet50 (input_chan, number_chan, number_block, first_stride):
    blocks = []
    for i in range(number_block):
        if i ==0 :
            blocks.append(Bottleneck(input_chan, number_chan, stride = first_stride)) 
        else :
             blocks.append(Bottleneck(number_chan * Bottleneck.expansion,
                                 number_chan, stride=1))
    return blocks
        




           


    