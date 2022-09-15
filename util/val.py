from unittest import case
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import numpy as np


def validation(model, criterion, val_loader, device):
    model.eval()
    rmse = nn.MSELoss().to(device)
    
    val_loss = []
    val_rmse = []
    with torch.no_grad():
        for sem, depth, _ in tqdm(iter(val_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            
            model_pred = model(sem)
            loss = criterion(model_pred, depth)
            
            pred = (model_pred*255.).type(torch.int8).float()
            true = (depth*255.).type(torch.int8).float()
            
            b_rmse = torch.sqrt(criterion(pred, true))
            
            val_loss.append(loss.item())
            val_rmse.append(b_rmse.item())

    return np.mean(val_loss), np.mean(val_rmse)