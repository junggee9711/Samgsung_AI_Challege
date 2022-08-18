from importlib.resources import path
import zipfile
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import cv2

def inference(model, test_loader, device):
    model.to(device)
    model.eval()

    result_name_list = []
    result_list = []
    with torch.no_grad():
        for sem, name in tqdm(iter(test_loader)):
            sem = sem.float().to(device)
            model_pred = model(sem)

            for pred, img_name in zip(model_pred, name):
                pred = pred.cpu().numpy().transpose(1,2,0)*255.
                save_img_path = f'{img_name}'
                result_name_list.append(save_img_path)
                result_list.append(pred)
    os.makedirs('./submission', exist_ok=True)
    os.chdir("./submission/")
    sub_imgs = []
    for path, pred_img in zip(result_name_list, result_list):
        cv2.imwrite(path, pred_img)
        sub_imgs.append(path)
    submissions = zipfile.ZipFile("../submission.zip", 'w')
    for path in sub_imgs:
        #print(path) ## check the path for error
        submissions.write(path)
    submissions.close()