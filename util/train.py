from distutils.command.config import config
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from util.val import validation
import numpy as np
import json
import wandb

def train(model, optimizer, train_loader, val_loader, scheduler, device, config):
    wandb.init(project="Samsung_AI_Challenge", entity="jnamq")
    wandb.config = {
        "epochs":config['EPOCHS'],
        "learning_rate":config['LEARNING_RATE'],
        "batch_size":config['BATCH_SIZE'],
    }
    wandb.watch(model)
    
    model.to(device)
    criterion = nn.L1Loss().to(device)
    best_score = 999999
    best_model = None

    for epoch in range(1, config['EPOCHS']+1):
        model.train()
        train_loss = []
        for sem, depth, case, _ in tqdm(iter(train_loader)):
            sem = sem.float().to(device)
            depth = depth.float().to(device)
            case = case.float().to(device)

            optimizer.zero_grad()
            model_pred = model(sem)
            #limit max depth by case
            max = 130 + 10*case
            for i in range(len(model_pred)):
                model_pred[i][0][model_pred[i][0]>max[i]] = max[i]

            loss = criterion(model_pred, depth)
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
        
        val_loss, val_rmse = validation(model, criterion, val_loader, device)
        wandb.log({'Train Loss': np.mean(train_loss), 'Val Loss': val_loss, 'Val RMSE': val_rmse})
        print(f'Epoch : [{epoch}] Train Loss : [{np.mean(train_loss):.5f}] Val Loss : [{val_loss:.5f}] Val RMSE : [{val_rmse:.5f}]')
        
        if best_score > val_rmse:
            best_score = val_rmse
            best_model = model
        
        if scheduler is not None:
            scheduler.step()
    return best_model
    #choose best model based on val RMSE, and return the "best model" as output