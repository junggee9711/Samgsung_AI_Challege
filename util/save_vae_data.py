from importlib.resources import path
import zipfile
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import cv2

def save_imgs(data_loader, device):
    result_name_list = []
    result_list = []
    
    for img, _, _, name in tqdm(iter(data_loader)):
        img = img.float().to(device)
        for pred, img_name in zip(img, name):
            pred = pred.cpu().detach().numpy().transpose(1,2,0)*255.
            save_img_path = f'{img_name}'
            result_name_list.append(save_img_path)
            result_list.append(pred)
    os.makedirs('./vae_data', exist_ok=True)
    os.chdir("./vae_data/")
    for path, pred_img in zip(result_name_list, result_list):
        cv2.imwrite(path, pred_img)
    print("vae data saved")
    os.chdir("../")