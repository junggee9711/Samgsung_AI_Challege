import torch
from torch import nn

class Classification_Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.layer2 =  nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.fc1 = nn.Linear(72*3*64, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 4)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.fc1(x.view(x.size(0),-1))
        x = self.fc2(x)
        x = self.fc3(x)
        return x