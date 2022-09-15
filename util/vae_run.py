from dataclasses import dataclass
from unittest import case
import pandas as pd
import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import glob
from model.vae import VAE
from util.vae_train import train

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

from util.dataset import trainDataset, VAEDataset
import json
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def generate_data(config, train_sem_paths, train_depth_paths, val_sem_paths, val_depth_paths):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    seed_everything(config['SEED']) # Seed fixed

    train_img_paths = sorted(glob.glob('./dataset/train/SEM/*/*/*.png'))
    train_img_dataset = trainDataset(train_img_paths)
    train_img_loader = DataLoader(train_img_dataset, batch_size = config['BATCH_SIZE'], shuffle=True, num_workers=0)
    
    model = VAE(config['HEIGHT'], config['WIDTH'], 512, 256, 128)
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = 1e-3)
    train_model = train(model, optimizer, train_img_loader, None, device)
    
    vae_train_dataset = VAEDataset(train_sem_paths, train_depth_paths, train_model, device)
    vae_val_dataset = VAEDataset(val_sem_paths, val_depth_paths, train_model, device)
    return vae_train_dataset, vae_val_dataset