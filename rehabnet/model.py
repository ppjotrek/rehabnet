import torch
from torch import nn

class SEBlock(torch.nn.Module):

  def __init__(self,channel,reduction=16):
    super(SEBlock,self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Sequential(
        nn.Linear(channel,channel//reduction,bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel//reduction,channel,bias=False),
        nn.Sigmoid()
    )

  def forward(self,x):
    b,c,_= x.size()
    y = self.avg_pool(x).view(b,c)
    y = self.fc(y).view(b,c,1)
    return x*y.expand_as(x)
  
class ResSeBasicBlock(nn.Module):
  def __init__(self,in_channels,channels,stride=1,reduction=16,downsample = None) :
    super(ResSeBasicBlock,self).__init__()
    self.conv1 = nn.Conv1d(in_channels,channels,3,stride,padding=1,bias=False)
    self.bn1 = nn.BatchNorm1d(channels)
    self.elu = nn.ELU(inplace=True)
    self.conv2 = nn.Conv1d(channels,channels,3,1,padding=1,bias=False)
    self.bn2 = nn.BatchNorm1d(channels)
    self.se = SEBlock(channels,reduction)
    self.downsample = downsample

  def forward(self,x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.elu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.se(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out+=residual
    out = self.elu(out)

    return out
  
class SERes1d(nn.Module):

  def __init__(self, in_channels, num_classes, features):

    super(SERes1d,self).__init__()

    self.conv1 = nn.Conv1d(in_channels, features, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm1d(features)
    self.elu = nn.ELU()
    self.maxpool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)


    self.SE1_1 = ResSeBasicBlock(features, features)
    self.SE1_2 = ResSeBasicBlock(features, features)
    # downsample = None
    downsample = nn.Sequential(
        nn.Conv1d(features, features*2, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(features*2)
        )

    self.SE2_1 = ResSeBasicBlock(features, features*2, downsample=downsample, stride=2)
    self.SE2_2 = ResSeBasicBlock(features*2, features*2)

    downsample = nn.Sequential(
        nn.Conv1d(features*2, features*4, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(features*4)
        )
    self.SE3_1 = ResSeBasicBlock(features*2, features*4, downsample=downsample, stride=2)
    self.SE3_2 = ResSeBasicBlock(features*4, features*4)
    downsample = nn.Sequential(
        nn.Conv1d(features*4, features*8, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(features*8)
        )
    self.SE4_1 = ResSeBasicBlock(features*4, features*8, downsample=downsample, stride=2)
    self.SE4_2 = ResSeBasicBlock(features*8, features*8)

    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(features*8, num_classes)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.elu(x)
    x = self.maxpool(x)

    x = self.SE1_1(x)
    x = self.SE1_2(x)

    x = self.SE2_1(x)
    x = self.SE2_2(x)
    x = self.SE3_1(x)
    x = self.SE3_2(x)

    x = self.SE4_1(x)
    x = self.SE4_2(x)
    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.fc(x)

    return x