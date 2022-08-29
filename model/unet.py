import torch
import torch.nn as nn
import torchvision.transforms.functional as F

##############################################################################################################
#2(Conv-BN-ReLU) block
class Double_Conv(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super(Double_Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

#Contracting path (Max-pooling, Double_Conv)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super(Down, self).__init__()
        self.double_conv = Double_Conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        return self.double_conv(self.pool(x))

#Expansive path (up_conv, Double_Conv)
class UP(nn.Module):
    def __init__(self, in_channels, out_channels) :
        super(UP, self).__init__()
        self.double_conv = Double_Conv(in_channels, out_channels)
        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)

    def forward(self, x, x_skipped):
        x = self.up_conv(x)
        if x.size() != x_skipped.size():
            x = F.resize(x, size=x_skipped.shape[2:])
        x = torch.cat([x_skipped, x], dim=1)
        return self.double_conv(x)

##############################################################################################################
class DepthModel(nn.Module):
    def __init__(self, height, width) :
        super(DepthModel, self).__init__()
        self.double_conv = Double_Conv(1, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = UP(1024, 512)
        self.up2 = UP(512, 256)
        self.up3 = UP(256, 128)
        self.up4 = UP(128, 64)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.double_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.up1(x5,x4)
        x7 = self.up2(x6,x3)
        x8 = self.up3(x7,x2)
        x9 = self.up4(x8,x1)
        return self.out_conv(x9)