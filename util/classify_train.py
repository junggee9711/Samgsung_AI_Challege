import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np

def train_classify(model, optimizer, train_loader, device):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    for epoch in range(1, 5):
        model.train()
        train_loss = []
        for img, _, case, _ in tqdm(iter(train_loader)):
            img = img.float().to(device)
            optimizer.zero_grad()

            pred = model(img)
            label = torch.zeros(4).to(device)
            label[case-1] = 1
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(f'(Classification) Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}]')
    return model