import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, ic, oc, stride, dilation):
        super(BasicBlock, self).__init__()
        self.expansion = 1
        self.conv1 = nn.Conv2d(ic, oc, 3, stride, dilation, dilation, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        
        if stride != 1 or ic != oc * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(ic, oc * self.expansion, 1, stride, 0, bias=False),
                nn.BatchNorm2d(oc * self.expansion)
            )
        else:
            self.shortcut = nn.Identity()
        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
        