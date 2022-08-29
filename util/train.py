import torch
import torch.nn as nn
from tqdm.auto import tqdm
from util.val import validation
import numpy as np
import json

with open('./config/base_config.json', 'r') as f:
    config = json.load(f)

def train(model, optimizer, train_loader, val_loader, scheduler, device, epochs):
    model.to(device)
    criterion = nn.L1Loss().to(device)
    best_score = 999999
    best_model = None
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = []
        for sem, depth in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            
            optimizer.zero_grad()
            
            model_pred = model(sem)
            loss = criterion(model_pred, depth)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_rmse = validation(model, criterion, val_loader, device)
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}]')
        
        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model
        
        if scheduler is not None:
            scheduler.step()
            
    return best_model
    #choose best model based on val RMSE, and return the "best model" as output