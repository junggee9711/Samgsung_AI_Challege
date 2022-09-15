import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torch.nn.functional as F
import numpy as np

def train(model, optimizer, train_loader, scheduler, device):
    model.to(device)
    for epoch in range(1, 11):
        model.train()
        train_loss = []
        for img in tqdm(iter(train_loader)):
            img = img.float().to(device)
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(img)
            BCE, KLD = loss_function(recon_batch, img, mu, logvar)
            loss = BCE + KLD
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        print(f'(VAE) Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}]')
    return model

def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x.view(-1, 3456), x.view(-1, 3456), reduction = 'sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE, KLD