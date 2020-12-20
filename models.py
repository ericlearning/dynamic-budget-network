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

class Bottleneck(nn.Module):
    def __init__(self, ic, oc, stride, dilation):
        super(Bottleneck, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(ic, oc, 1, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(oc)
        self.conv2 = nn.Conv2d(oc, oc, 3, stride, dilation, dilation, bias=False)
        self.bn2 = nn.BatchNorm2d(oc)
        self.conv3 = nn.Conv2d(oc, oc * self.expansion, 1, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(oc * self.expansion)
        
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
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

    
class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)
    

class ResNet(nn.Module):
    def __init__(self, ic, oc, block, num_blocks, strides=[1, 2, 2, 2], use_dilations=[False, False, False]):
        self.conv1 = nn.Conv2d(ic, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool1 = nn.MaxPool2d(3, 2, 1)
        
        self.dilation = 1

        expansion = block.expansion
        self.layer1 = self.make_layer(num_blocks[0], block, ic, 64, strides[0], False)
        self.layer2 = self.make_layer(num_blocks[1], block, 64 * expansion, 128, strides[1], dilations[1])
        self.layer3 = self.make_layer(num_blocks[2], block, 128 * expansion, 256, strides[2], dilations[2])
        self.layer4 = self.make_layer(num_blocks[3], block, 256 * expansion, 512, strides[3], dilations[3])
        
        self.cls = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(512 * expansion, oc)
        )
    
    def make_layer(self, num_blocks, block, ic, oc, stride, use_dilation):
        dilation = self.dilation
        if use_dilation:
            self.dilation *= stride
            stride = 1
        
        layer = [block(ic, oc, stride, dilation)]
        for _ in range(num_blocks-1):
            layer.append(block(oc * block.expansion, oc, 1, self.dilation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.maxpool1(out)
        
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        
        out = self.cls(out)
        return out