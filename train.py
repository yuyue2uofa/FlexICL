#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
import argparse
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'


import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import random

import cv2 as cv

import pickle
from config import get_config
from image_dataloader import *
from models import *

parser = argparse.ArgumentParser(description='SimICL')
parser.add_argument('--epochs', default=1200, type=int, metavar='N',
                    help='number of total epochs to run(default: 100)') 
parser.add_argument('--dataset_path', default = '/folder_path/', type=str) 
parser.add_argument('--train_csv', default='train.csv', type=str)
parser.add_argument('--save_dir', default = './results/', type=str)
parser.add_argument('--batch_size', type=int, default=256, help='batch size') 
parser.add_argument('--weight_decay', type=float, default=0.05,help='weight decay') 
parser.add_argument('--lr', type=float, default=0.0005,help='learning rate ') 
parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
parser.add_argument('--mask_ratio', default=0.6, type=float, help='Masking ratio (percentage of removed patches).')
parser.add_argument('--save_frequency', default=100, type=int, help='save trained model')
parser.add_argument('--augmentation', action='store_true', help='use image augmentation')
parser.add_argument('--augmentation_number', default=5, type=int, help='multiple number of training images')
parser.add_argument('--shuffle_training', action='store_true', help='shuffle support-query pair during training')
parser.add_argument('--softmask', action='store_true', help='use soft mask')
parser.add_argument('--frac_ratio', default=0.1, type=float, help='images selected percentage')
parser.add_argument('--random_state_seed', default=123, type=int, help='dataframe random state seed')
parser.add_argument('--random_masking', action='store_true', help='random masking')
parser.add_argument('--aug_threshold', default=0.5, type=float, help='imagewise augmentation threshold')
parser.add_argument('--device_gpu', default='cuda', type=str, help='specify the gpu you are using')
parser.add_argument('--cfg', default ='./configs/vit_base__800ep/simmim_pretrain__vit_base__img224__800ep.yaml', type=str,  metavar="FILE", help='path to config file', )
parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

args = parser.parse_args() 


config = get_config(args)

# If there's a GPU available...
if torch.cuda.is_available():
    device = args.device_gpu
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))
else:
    cf = "cpu"
    print('We will use CPU')

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
    print("Folder created")
else:
    print("Folder already exists")


# model
model = build_model(config) 
model.to(device)

PATH = args.save_dir + 'latest_model.pth' 
if os.path.exists(PATH): 
    model.load_state_dict(torch.load(PATH))


# Training function
def train_model(dataloader, optimizer, model, epoch):
    train_loss = 0
    model.train()
    i=0
    for images, masks, gts, image_filename1, image_filename2 in dataloader: 
        i += 1 
        images = images.to(device) 
        masks = masks.to(device)
        gts = gts.to(device)  

        
        loss, rec = model(images, masks, gts) 
        optimizer.zero_grad()
        loss.backward() # Calculate Gradients
        optimizer.step() # Update Weights


        train_loss += loss.item()

        if epoch % 100 == 0 and epoch > 400:
            if i == 1:
                images = images.cpu()[0] 
                masks = masks.cpu()[0]
                gts = gts.cpu()[0]
                rec = rec.cpu()[0] 
                masks = masks.numpy()
                mask = masks.reshape(14, 14)
                mask = 1-np.repeat(mask, 16, axis=0).repeat(16, axis=1)           
                pred1 = rec.detach().cpu().numpy()
                pred1 = (pred1 - np.min(pred1)) / np.ptp(pred1)
                images_input = images * mask

                filename = image_filename1[0].split('.')[0]+'_'+image_filename2[0].split('.')[0] + '_'+str(epoch)+'_pred.png'
                filename_gt = image_filename1[0].split('.')[0]+'_'+image_filename2[0].split('.')[0] + '_'+ str(epoch)+'_gt.png'
                filename_input = image_filename1[0].split('.')[0]+'_'+image_filename2[0].split('.')[0] + '_'+ str(epoch)+'_input.png'

                cv.imwrite(args.save_dir+filename, pred1.transpose(1,2,0)*255) #save prediction
                cv.imwrite(args.save_dir+filename_gt, gts.numpy().transpose(1,2,0)*255) #save gt image
                cv.imwrite(args.save_dir+filename_input, images_input.numpy().transpose(1,2,0)*255) #save input image

        
    return train_loss/len(dataloader), rec, masks, model

    
train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize([224, 224]), 
        transforms.ToTensor()])

# Run training and evaluation cycles
with open(args.save_dir + 'training_result.txt', 'a') as f:
    f.write('Batch size:'+str(args.batch_size)+'Learning rate:'+str(args.lr))

optimizer = optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
torch.set_grad_enabled(True)

result_path = args.save_dir + "training_result.pickle"


if not os.path.exists(result_path): #if results do not exit
    results = {'train_loss': []}
else:
    results_ = open(result_path,'rb')
    results = pickle.load(results_)
    results_.close()
start_epoch = len(results['train_loss'])

for epoch in range(start_epoch, args.epochs):
    # Create training dataset
    if args.shuffle_training:
        training_dataset = SegmentationDataSet_train(dataset_path = args.dataset_path, df_input = args.train_csv, mask_ratio = args.mask_ratio, transform = train_transform, augmentation=args.augmentation, augmentation_number=args.augmentation_number, frac_ratio=args.frac_ratio, random_state_seed=args.random_state_seed, random_shuffle = epoch, softmask=args.softmask, random_masking = args.random_masking, aug_threshold = args.aug_threshold)        
        print('test original concatenated images')
    else:
        training_dataset = SegmentationDataSet_train(dataset_path = args.dataset_path, df_input = args.train_csv, mask_ratio = args.mask_ratio, transform = train_transform, augmentation=args.augmentation, augmentation_number=args.augmentation_number, frac_ratio=args.frac_ratio, random_state_seed=args.random_state_seed, softmask=args.softmask)        
        print('test original concatenated images-unchanged pair: no shuffle training')
    training_dataloader = data.DataLoader(dataset=training_dataset, batch_size=args.batch_size, shuffle = True, num_workers=4)
    with open(args.save_dir + 'training_result.txt', 'a') as f:
        f.write("dataset successfully loaded \n")

    train_loss, pred, mask, model= train_model(training_dataloader, optimizer, model, epoch)
    # save model
    if epoch % 1 == 0:
        print("(epoch "+str(epoch)+")", 
              "\t"+"train loss: "+str(train_loss))
        torch.save(model.state_dict(),args.save_dir + 'latest_model.pth') 
        with open(args.save_dir + 'training_result.txt', 'a') as f:
            f.write("(epoch "+str(epoch)+")"+ 
                    "\t"+"train loss: "+str(train_loss)+"\n")
    results['train_loss'].append(train_loss)

    pickle.dump(results, open(args.save_dir + "training_result.pickle", "wb")) 
    if (epoch+1) % args.save_frequency == 0 and epoch >500:
        torch.save(model.state_dict(),args.save_dir + 'latest_model_'+str(epoch)+'.pth') 
    torch.cuda.empty_cache()        






