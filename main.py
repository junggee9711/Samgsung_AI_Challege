from dataclasses import dataclass


import random
from this import d
import pandas as pd
import numpy as np
import os
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error

import warnings

from dataset import CustomDataset, data_split
from config import CFG
from model import BaseModel
from train import train
from inference import inference
from random_seed import seed_everything
warnings.filterwarnings(action='ignore') 
if __name__ == '__main__':
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    seed_everything(CFG['SEED']) # Seed 고정

    ###############################################################################################################################
    #set data paths
    simulation_sem_paths = sorted(glob.glob('./dataset/simulation_data/SEM/*/*/*.png'))
    simulation_depth_paths = sorted(glob.glob('./dataset/simulation_data/Depth/*/*/*.png')+glob.glob('./dataset/simulation_data/Depth/*/*/*.png'))
    
    #split train/eval data
    d_split = data_split(simulation_sem_paths, simulation_depth_paths)
    train_sem_paths, train_depth_paths, val_sem_paths, val_depth_paths = d_split.split(0.8)

    #define dataset
    train_dataset = CustomDataset(train_sem_paths, train_depth_paths)
    train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

    val_dataset = CustomDataset(val_sem_paths, val_depth_paths)
    val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)
    ###############################################################################################################################
    #run!!
    model = BaseModel(CFG['HEIGHT'], CFG['WIDTH'])
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG["LEARNING_RATE"])
    scheduler = None

    infer_model = train(model, optimizer, train_loader, val_loader, scheduler, device)
    ###############################################################################################################################
    #inference & submission
    test_sem_path_list = sorted(glob.glob('./dataset/test/SEM/*.png'))
    test_dataset = CustomDataset(test_sem_path_list, None)
    test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)

    inference(infer_model, test_loader, device)