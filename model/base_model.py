import torch
import torch.nn as nn

class DepthModel(nn.Module):
    def __init__(self, height, width):
        super(DepthModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(height * width, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(), 
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, height * width),
        )
        self.height = height
        self.width = width
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
    def forward(self, x):
        x = x.view(-1, self.height * self.width)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.view(-1, 1, self.height ,  self.width)
        return x