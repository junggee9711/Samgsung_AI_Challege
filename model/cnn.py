import torch
import torch.nn as nn

class Conv_Block(nn.Module):
    def __init__(self) :
        super(Conv_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class DepthModel(nn.Module):
    def __init__(self, height, width) :
        super(DepthModel, self).__init__()
        self.conv_block = self.make_layer(Conv_Block, 5)
        self.in_conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.out_conv = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU()

    def make_layer(self, block, num_layers):
        layers = []
        for _ in range(num_layers):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.in_conv(x))
        x = self.conv_block(x)
        x = self.out_conv(x)
        return x