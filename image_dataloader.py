# dataloader

import random
import cv2 as cv
import pandas as pd
import numpy as np
import os
from torch.utils import data
import torch
import re

def sample_x_percent(group, frac_ratio = 0.05, random_state_seed = 123):
    return group.sample(frac=frac_ratio, random_state=random_state_seed)
    
class MaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=16, model_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1) 
        return mask

class SoftMaskGenerator:
    def __init__(self, input_size=224, mask_patch_size=16, model_patch_size=16, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=float)
        
        
        for idx in mask_idx:
            mask[idx] = np.random.rand()
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1) 
        return mask



class SegmentationDataSet_train(data.Dataset):
    def __init__(self, dataset_path: str, df_input:str, mask_ratio:float, transform=None, augmentation=False, augmentation_number=5, frac_ratio=0.1, random_state_seed = 123, random_shuffle=321, softmask=False, random_masking = False, aug_threshold = 0.5):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')  
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df = pd.read_csv(df_input)
        self.df = self.df.groupby('videoname', group_keys=False).apply(sample_x_percent, frac_ratio=frac_ratio, random_state_seed=random_state_seed) #select x% images from each video

        list_temp = list(self.df['filename'])
        random.seed(random_shuffle)

        if augmentation:
            self.images_list1 = list_temp[:len(list_temp)//2] * augmentation_number
            self.images_list2 = list_temp[len(list_temp)//2:] * augmentation_number
            random.shuffle(self.images_list1)
            random.shuffle(self.images_list2)
        else:
            self.images_list1 = list_temp[:len(list_temp)//2]
            self.images_list2 = list_temp[len(list_temp)//2:] 
            random.shuffle(self.images_list1)
            random.shuffle(self.images_list2)

        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transform = transform
        self.random_masking = random_masking
        self.aug_threshold = aug_threshold

        model_patch_size=16
         
        if softmask:
            self.mask_generator = SoftMaskGenerator(input_size=224,mask_patch_size=16, model_patch_size=16,mask_ratio=mask_ratio)
        else:            
            self.mask_generator = MaskGenerator(input_size=224,mask_patch_size=16, model_patch_size=16,mask_ratio=mask_ratio)
            
    def __len__(self):
        return len(self.images_list1)
    def __getitem__(self, index: int):
        # Select the sample
        image_filename1 = self.images_list1[index]
        image_filename2 = self.images_list2[index]

        # Load input and target: support pair
        image1 = cv.imread(os.path.join(self.input_path, image_filename1),0) # read concatenated input
        gt1  = cv.imread(os.path.join(self.output_path, image_filename1),0)  # read concatenated ground truth
        gt1[gt1>=1]=255
        width = max(image1.shape) - image1.shape[1]  # pad 0 on width
        height = max(image1.shape) - image1.shape[0]  # pad 0 on height
        image1 = np.pad(image1, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        gt1 = np.pad(gt1, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        image1 = cv.resize(image1, (224,224))
        gt1 = cv.resize(gt1, (224,224))
        gt1[gt1>=128]=255
        gt1[gt1<128]=0

        # Load input and target: query pair
        image2 = cv.imread(os.path.join(self.input_path, image_filename2),0) # read concatenated input
        gt2  = cv.imread(os.path.join(self.output_path, image_filename2),0)  # read concatenated ground truth
        gt2[gt2>=1]=255
        width = max(image2.shape) - image2.shape[1]  # pad 0 on width
        height = max(image2.shape) - image2.shape[0]  # pad 0 on height
        image2 = np.pad(image2, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        gt2 = np.pad(gt2, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        image2 = cv.resize(image2, (224,224))
        gt2 = cv.resize(gt2, (224,224))
        gt2[gt2>=128]=255
        gt2[gt2<128]=0

        current_random_state = random.getstate()   
        random.seed(None)

        if random.random() > self.aug_threshold:
            image1 = np.fliplr(image1)
            gt1 = np.fliplr(gt1)
            image2 = np.fliplr(image2)
            gt2 = np.fliplr(gt2)

            crop_size1 = random.randint(150, 224)   
            x1 = random.randint(0, 224 - crop_size1)
            y1 = random.randint(0, 224 - crop_size1)
            crop_size2 = random.randint(150, 224)  
            x2 = random.randint(0, 224 - crop_size2)
            y2 = random.randint(0, 224 - crop_size2)


            image1 = image1[y1:y1+crop_size1, x1:x1+crop_size1]
            gt1 = gt1[y1:y1+crop_size1, x1:x1+crop_size1]
            image2 = image2[y2:y2+crop_size2, x2:x2+crop_size2]
            gt2 = gt2[y2:y2+crop_size2, x2:x2+crop_size2]

 
        random.setstate(current_random_state)

        image1 = cv.resize(image1, (224,224))
        gt1 = cv.resize(gt1, (224,224))
        image2 = cv.resize(image2, (224,224))
        gt2 = cv.resize(gt2, (224,224))

        # concatenated output: GT
        output = np.zeros((448, 448))
        output[:224,:224] = image1
        output[:224,224:] = gt1
        output[224:,:224] = image2
        output[224:,224:] = gt2
        output = np.uint8(output)
    
        # concatenated input
        input = np.zeros((448, 448))
        input[:224,:224] = image1
        input[:224,224:] = gt1
        input[224:,:224] = image2
        input= np.uint8(input)
      
        # add: 3 channel
        input = np.repeat(input[None,...], 3, axis=0).transpose(1, 2, 0)
        output = np.repeat(output[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.transform:
            input = self.transform(np.uint8(input))
            output = self.transform(np.uint8(output)) 
        if self.random_masking:
            current_random_state = random.getstate()    
            random.seed(None)            
            mask = self.mask_generator()
            random.setstate(current_random_state)
        else:
            mask = self.mask_generator()            
        return input, mask, output, image_filename1, image_filename2

class SegmentationDataSet_test_random(data.Dataset):
    def __init__(self, dataset_path: str, df_input1:str, df_input2:str, mask_ratio:float, transform=None, softmask=False, random_masking=False):
        self.dataset_path = dataset_path
        self.input_path = os.path.join(self.dataset_path, 'Images/')  
        self.output_path = os.path.join(self.dataset_path, 'Masks/')
        self.df1 = pd.read_csv(df_input1) # training set
        self.df2 = pd.read_csv(df_input2) # test set

        
        self.images_list1 = list(self.df1['filename']) # use images from training set as support image
        random.shuffle(self.images_list1)
        self.images_list2 = list(self.df2['test_filename']) # use images from testing set as query image
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.float32
        self.transform = transform
        self.random_masking = random_masking

        model_patch_size=16
        if softmask:
            self.mask_generator = SoftMaskGenerator(input_size=224,mask_patch_size=16, model_patch_size=16,mask_ratio=mask_ratio)
        else:            
            self.mask_generator = MaskGenerator(input_size=224,mask_patch_size=16, model_patch_size=16,mask_ratio=mask_ratio)         
            
    def __len__(self):
        return len(self.images_list2)
    def __getitem__(self, index: int):
        # Select the sample
        image_filename2 = self.images_list2[index]
        image_filename1 = self.images_list1[index]

        
        # Load input and target: support pair
        image1 = cv.imread(os.path.join(self.input_path, image_filename1),0) # read concatenated input
        gt1  = cv.imread(os.path.join(self.output_path, image_filename1),0)  # read concatenated ground truth
        gt1[gt1>=1]=255
        width = max(image1.shape) - image1.shape[1]  # pad 0 on width
        height = max(image1.shape) - image1.shape[0]  # pad 0 on height
        image1 = np.pad(image1, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        gt1 = np.pad(gt1, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        image1 = cv.resize(image1, (224,224))
        gt1 = cv.resize(gt1, (224,224))
        gt1[gt1>=128]=255
        gt1[gt1<128]=0

        # Load input and target: query pair
        image2 = cv.imread(os.path.join(self.input_path, image_filename2),0) # read concatenated input
        gt2  = cv.imread(os.path.join(self.output_path, image_filename2),0)  # read concatenated ground truth
        gt2[gt2>=1]=255
        width = max(image2.shape) - image2.shape[1]  # pad 0 on width
        height = max(image2.shape) - image2.shape[0]  # pad 0 on height
        image2 = np.pad(image2, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        gt2 = np.pad(gt2, ((0, height), (0, width)), 'constant', constant_values=(0, 0))
        image2 = cv.resize(image2, (224,224))
        gt2 = cv.resize(gt2, (224,224))
        gt2[gt2>=128]=255
        gt2[gt2<128]=0

        # concatenated output: GT
        output = np.zeros((448, 448))
        output[:224,:224] = image1
        output[:224,224:] = gt1
        output[224:,:224] = image2
        output[224:,224:] = gt2
        output = np.uint8(output)
    
        # concatenated input
        input = np.zeros((448, 448))
        input[:224,:224] = image1
        input[:224,224:] = gt1
        input[224:,:224] = image2
        input= np.uint8(input)
      
        # add: 3 channel
        input = np.repeat(input[None,...], 3, axis=0).transpose(1, 2, 0)
        output = np.repeat(output[None,...], 3, axis=0).transpose(1, 2, 0)
        # add transform
        if self.transform:
            input = self.transform(np.uint8(input))
            output = self.transform(np.uint8(output)) 

            
        if self.random_masking:
            current_random_state = random.getstate()    
            random.seed(None)            
            mask = self.mask_generator()
            random.setstate(current_random_state)
        else:
            mask = self.mask_generator()   
            
        return input, mask, output, image_filename1, image_filename2
