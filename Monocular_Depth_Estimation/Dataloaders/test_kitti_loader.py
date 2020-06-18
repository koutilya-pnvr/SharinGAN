
import collections
import glob
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data
from Kitti_dataset_util import KITTI
import random
import cv2
from transform import *
from torchvision import transforms as tr

class KittiDataset(data.Dataset):
    def __init__(self, root='/vulcan/scratch/koutilya/projects/Domain_Adaptation/Common_Domain_Adaptation-Lighting/SharinGAN_results/fig5',
                 img_transform=None, joint_transform=None, depth_transform=None, complete_data=False):
      
        self.root = root
        self.files = []
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.depth_transform = depth_transform
        self.complete_data = complete_data

        self.files = os.listdir(self.root)

                                    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        filename = osp.join(self.root, self.files[idx])
        l_img = Image.open(filename).convert('RGB')
        
        l_img, _, _, _ = self.joint_transform((l_img, None, None, 'test', None))
            
        if self.img_transform is not None:
            l_img = self.img_transform(l_img)
    
        
        return l_img, filename.replace('img','ours')