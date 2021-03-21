from __future__ import print_function, division
import os.path as osp
import glob

import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms as tr
import cv2
import torchvision.transforms.functional as F
import random

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    gpu = 0
    seed=250
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class to_tensor():
    def __init__(self):
        self.ToTensor = tr.ToTensor()
    
    def crop_center(self, img,cropx,cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]
    
    def __call__(self,depth):
        depth = cv2.resize(depth, (1800, 2000), cv2.INTER_NEAREST)
        depth = self.crop_center(depth, 1600, 1024)
        depth = cv2.resize(depth, (640, 192), cv2.INTER_NEAREST)

        depth_tensor = torch.from_numpy(depth).unsqueeze(0).float()
        depth_tensor[depth_tensor<0.0] = 0.0
        depth_tensor[depth_tensor>80.0] = 80.0
        return depth_tensor / 80.0

class Make3D_dataset(Dataset):
    def __init__(self, root_dir='/vulcanscratch/koutilya/Make3D', depth_resize='bilinear'):
        self.root_dir = root_dir
        self.tensor_transform = [tr.ToPILImage(), tr.Resize((2000, 1800)), tr.CenterCrop((1024, 1600)), tr.Resize((192,640)), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.label_tensor_transform = [to_tensor()]
        self.depth_resize = depth_resize

        self.image_files = glob.glob(osp.join(self.root_dir,'Test134','*.jpg'))
        self.label_files = [f.replace('Test134','Gridlaserdata').replace('img','depth_sph_corr').replace('jpg','mat') for f in self.image_files]
        
        self.image_transform = tr.Compose(self.tensor_transform)
        self.label_transform = tr.Compose(self.label_tensor_transform)
        
        self.color_new_height = 1704 // 2
        self.depth_new_height = 21
        

    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self,idx):
        
        image = cv2.imread(self.image_files[idx])[:,:,::-1]
        depth_filename = self.label_files[idx]
        image = self.image_transform(image)
        return image, self.image_files[idx].split('/')[-1].split('.')[0], depth_filename
