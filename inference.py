from importlib.resources import path
import zipfile
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import os
import cv2
import pdb

def inference(model, class_model, test_loader, device):
    model.to(device)
    model.eval()

    result_name_list = []
    result_list = []
    with torch.no_grad():
        for sem, name in tqdm(iter(test_loader)):
            sem = sem.float().to(device)
            model_pred = model(sem)

            if class_model :
                case = (torch.argmax(class_model(sem), dim=1) +1).float().to(device)
                max = (130 + 10*case)/255
                
                for i in range(len(model_pred)):
                    model_pred[i][0][model_pred[i][0]>max[i]] = max[i]

            for pred, img_name in zip(model_pred, name):
                pred = pred.cpu().numpy().transpose(1,2,0)*255
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