import torch
import glob
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np


# separate train/val data
class data_split:
    def __init__(self, sem_paths, depth_paths):
        self.sem_paths = sem_paths
        self.depth_paths = depth_paths
        self.data_len = len(sem_paths)
    def split(self, ratio):
        train_sem_paths = self.sem_paths[:int(self.data_len*ratio)]
        train_depth_paths = self.depth_paths[:int(self.data_len*ratio)]
        val_sem_paths = self.sem_paths[int(self.data_len*ratio):]
        val_depth_paths = self.depth_paths[int(self.data_len*ratio):]
        return train_sem_paths, train_depth_paths, val_sem_paths, val_depth_paths

#Dataset class
class CustomDataset(Dataset):
    def __init__(self, sem_path_list, depth_path_list):
        super(CustomDataset, self).__init__()
        self.sem_path_list = sem_path_list
        self.depth_path_list = depth_path_list

    def __getitem__(self, index):
        sem_path = self.sem_path_list[index]
        sem_img = cv2.imread(sem_path, cv2.IMREAD_GRAYSCALE)
        sem_img =  np.expand_dims(sem_img, axis= -1).transpose(2,0,1)
        sem_img = sem_img / 255
        
        if self.depth_path_list is not None:
            depth_path = self.depth_path_list[index]
            depth_img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
            depth_img = np.expand_dims(depth_img, axis=-1).transpose(2,0,1)
            depth_img = depth_img / 255
            return torch.Tensor(sem_img), torch.Tensor(depth_img)
        else : 
            img_name = sem_path.split('/')[-1]
            img_name = img_name.split('\\')[-1] #local address
            #print(img_name)
            #######################
            return torch.Tensor(sem_img), img_name
    
    def __len__(self):
        return len(self.sem_path_list)