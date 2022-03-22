import os
import cv2
import xml.dom.minidom as dom
import numpy as np
import torch
import torchvision.transforms as transforms
import shutil
import pandas as pd
from mmdet.apis import init_detector, inference_detector
import mmcv
from Anchor_generator import bbox_overlaps
from sklearn.metrics import roc_auc_score
from PIL import Image
from collections import defaultdict

os.environ["CUDA_VISIBLE_DEVICES"] = '1, 2, 3'

# MMdetection
config_file = '/data/ljm/mmdetection/configs/DBD/Faster_DBD_Config.py'
model_paths = ['/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/DBD/Faster_R-CNN_50/fold1.pth',
               '/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/DBD/Faster_R-CNN_50/fold2.pth',
               '/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/DBD/Faster_R-CNN_50/fold3.pth',
               '/data/ljm/LRCN/LITS2017肝脏肿瘤分割挑战数据集/DBD/Faster_R-CNN_50/fold4.pth']

val_case_dir = '/data/ljm/LRCN/case_CT_with_DBD'

val_cases = [['process_1', 'process_2', 'process_3', 'process_4'],
             ['process_5', 'process_6', 'process_7', 'process_8'],
             ['process_9', 'process_10', 'process_11', 'process_12'],
             ['process_13', 'process_14', 'process_15', 'process_16']]

val_control_dir = '/data/ljm/LRCN/new_control_CT_imgs_PSM'


## tpr, fpr means I-PP value for the case group and the matched control group separately

def tpr_fpr_cal(config_file, checkpoint_file, val_case_name=(), thr=0.3, mode='tpr', device='cuda:1'):

    model = init_detector(config_file, checkpoint_file, device=device)

    if mode == 'tpr':
        num_total_imgs = [0 for _ in val_case_name]  
        num_pos_imgs = [0 for _ in val_case_name]
        for i in range(4):
            patients_imgs_dir = os.path.join(val_case_dir, val_case_name[i])
            n_imgs = len(os.listdir(patients_imgs_dir))
            num_total_imgs[i] += n_imgs

            for img_name in os.listdir(patients_imgs_dir):
                img_path = os.path.join(patients_imgs_dir, img_name)
                result = inference_detector(model, img_path)[0]
                if len(result) > 0 and np.any(result[:, 4] >= thr):
                    num_pos_imgs[i] += 1

        tpr = [round(num_pos_imgs[j] / num_total_imgs[j], 2) for j in range(len(val_case_name))]

        return tpr
    elif mode == 'fpr':
        val_control_name = os.listdir(val_control_dir)
        val_control_name.sort()
        num_total_imgs = [0 for _ in val_control_name]
        num_pos_imgs = [0 for _ in val_control_name]
        
        for i, name in enumerate(val_control_name):
            imgs_dir = os.path.join(val_control_dir, name)
            imgs_list = os.listdir(imgs_dir)
            num_total_imgs[i] += len(imgs_list)
            for img_name in imgs_list:
                img_path = os.path.join(imgs_dir, img_name)
                result = inference_detector(model, img_path)[0]
                if len(result) > 0 and np.any(result[:, 4] >= thr):
                    num_pos_imgs[i] += 1

        fpr = [round(num_pos_imgs[j] / num_total_imgs[j], 3) for j in range(len(val_control_name))]
        return fpr
