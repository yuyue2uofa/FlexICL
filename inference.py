#!/usr/bin/env python
# coding: utf-8


# If there's a GPU available...

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd

import cv2 as cv


import pickle
#from sklearn.metrics import confusion_matrix
from config import get_config
from models import *
from image_dataloader import *


import random
from torch.utils.data import DataLoader



parser = argparse.ArgumentParser(description='SimICL')
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='train.csv', type=str)
parser.add_argument('--test_csv', default='test.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str) 
parser.add_argument('--batch_size', type=int, default=1,
                        help='batch size (default: 1)') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0.0, type=float,
                        help='Masking ratio (percentage of removed patches).')
parser.add_argument('--save_figure', action='store_true', help='save test prediction')
parser.add_argument('--model_name', default='latest_model.pth', type=str,
                        help='model name')
parser.add_argument('--softmask', action='store_true', help='use softmask')
parser.add_argument('--random_masking', action='store_true', help='random masking')
parser.add_argument('--random', action='store_true', help='random or most similar pairs during inference')
parser.add_argument('--max_performance', action='store_true', help='max performance during inference')
parser.add_argument('--device_gpu', default='cuda', type=str, help='specify the gpu you are using')
parser.add_argument('--cfg', default ='./configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml', type=str,  metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
parser.add_argument('--saved_image_folder', default ='test/', type=str, help='path to save image file', )

args = parser.parse_args() 

config = get_config(args)


if torch.cuda.is_available():
    device = args.device_gpu
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    cf = "cpu"
    print('We will use CPU')


def evaluation_metrics(y_true, y_pred, smooth = 0.0001):
    #y_true = y_true.detach().numpy()
    #y_pred = y_pred.detach().numpy()
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)

    dice = (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
    jaccard = (intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth - intersection)
    return(dice, jaccard)


# model
model = build_model(config) 
model.to(device)
print('save_dir:',args.save_dir)
print(args.model_name)
print('mask_ratio:',args.mask_ratio)
PATH = args.save_dir + args.model_name
if os.path.exists(PATH):
    state_dict = torch.load(PATH, map_location=device)
    model.load_state_dict(state_dict)
else:
    print('no pretrained model found')

test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]), 
        transforms.ToTensor()])
# Create test dataset
test_dataset = SegmentationDataSet_elbow_test_random(dataset_path=args.dataset_path, df_input1=args.train_csv, df_input2=args.test_csv, mask_ratio=args.mask_ratio,transform=test_transform, softmask=args.softmask, random_masking = args.random_masking)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                      batch_size=args.batch_size,
                                      shuffle = False, num_workers=4)


save_images = args.save_dir + args.saved_image_folder

if not os.path.exists(save_images):
    os.makedirs(save_images)
    print("Folder created")
else:
    print("Folder already exists")

def eval_model(dataloader, model, save_path):
    eval_loss = 0
    model.eval()
    dices = []
    jaccards = []

    with torch.no_grad():
        i = 0
        for images, masks, gts, image_filename1, image_filename2 in dataloader:  
            i += 1

            images = images.to(device) 
            masks = masks.to(device) 
            gts = gts.to(device) 
            loss, rec = model(images, masks, gts) 
            images = images.cpu() 
            masks = masks.cpu()  
            gts = gts.cpu()
            rec = rec.cpu() 
            #print(rec.shape, masks.shape)
            eval_loss =+ loss.item()       


            masks = masks.numpy()
            mask = masks.reshape(14, 14)
            mask = 1-np.repeat(mask, 16, axis=0).repeat(16, axis=1) 
            
            pred1 = rec.numpy()
            pred1 = (pred1 - np.min(pred1)) / np.ptp(pred1)
            images_input = images * mask
            
            
            # calculate dice and jaccard
            gt_temp = gts.numpy() #0-1, (1, 3, 224, 224)
            gt_temp = gt_temp[0,:, 112:,112:]
            pred_temp = pred1[0,:, 112:,112:]

            gt_temp[gt_temp>0.5] = 1
            gt_temp[gt_temp<=0.5] = 0
            pred_temp[pred_temp>0.5] = 1
            pred_temp[pred_temp<=0.5] = 0
            dice, jaccard = evaluation_metrics(gt_temp,pred_temp)
            dices.append(dice)
            jaccards.append(jaccard)
            filename = image_filename2[0]

            if i%1 == 0 and args.save_figure:  
                cv.imwrite(save_path+filename, pred_temp.transpose(1,2,0)*255) #save prediction
    return dices, jaccards


dices, jaccards= eval_model(test_dataloader, model, save_images) 
print('saved folder:', args.save_dir)
print(sum(dices)/len(dices), sum(jaccards)/len(jaccards))

