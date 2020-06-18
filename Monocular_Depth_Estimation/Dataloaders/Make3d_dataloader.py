from __future__ import print_function, division
import os
import os.path as osp
import glob
import time

import torch
from skimage import io, transform
import scipy.io
from scipy.misc import imsave
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr, utils
import cv2
from tqdm import tqdm
import lycon
from skimage.morphology import binary_closing, disk
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
from PIL import Image

use_cuda = torch.cuda.is_available()
torch.manual_seed(1)
if use_cuda:
    gpu = 0
    seed=250
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

class Paired_transform():
    def __call__(self,image, depth):
        flip = random.random()
        if flip>0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)
        
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            if depth is not None:
                depth = F.rotate(depth, degree, Image.BILINEAR)
        return image, depth

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
    def __init__(self, root_dir='/vulcan/scratch/koutilya/Make3D', depth_resize='bilinear'):
        self.root_dir = root_dir
        self.tensor_transform = [tr.ToPILImage(), tr.Resize((2000, 1800)), tr.CenterCrop((1024, 1600)), tr.Resize((192,640)), tr.ToTensor(), tr.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        self.label_tensor_transform = [to_tensor()]#, tr.Normalize((0.5,), (0.5,))]
        self.depth_resize = depth_resize

        self.image_files = glob.glob(osp.join(self.root_dir,'Test134','*.jpg'))
        self.label_files = [f.replace('Test134','Gridlaserdata').replace('img','depth_sph_corr').replace('jpg','mat') for f in self.image_files]
        
        self.image_transform = tr.Compose(self.tensor_transform)
        self.label_transform = tr.Compose(self.label_tensor_transform)
        
        self.color_new_height = 1704 // 2
        self.depth_new_height = 21
        

    def __len__(self):
        return len(self.image_files)
    
    def crop_center(self, img,cropx,cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]


    def __getitem__(self,idx):
        
        image = cv2.imread(self.image_files[idx])[:,:,::-1]
        depth_filename = self.label_files[idx]
        # depth = scipy.io.loadmat(self.label_files[idx])
        # depth = depth['Position3DGrid']
        # depth = depth[:,:,3]
    
        # image = image[ (2272 - self.color_new_height)//2:(2272 + self.color_new_height)//2,:]
        # image = cv2.resize(image, (640,192), interpolation=cv2.INTER_LINEAR)
        # depth = cv2.resize(depth, (305, 407), interpolation=cv2.INTER_NEAREST)
        # depth = depth[(55 - 21)//2:(55 + 21)//2]

        # image = np.ascontiguousarray(image)
        # image = torch.from_numpy(image).permute(2,0,1).float() / 255.0
        
        image = self.image_transform(image)
        # depth = self.label_transform(depth)
        # image = image.contiguous()
        # return image, self.image_files[idx], depth_filename
        return image, self.image_files[idx].split('/')[-1].split('.')[0], depth_filename

# target_filenames = {'img1':'/vulcan/scratch/koutilya/Make3D/Test134/img-060705-17.10.14-p-080t000.jpg',
# 'img2': '/vulcan/scratch/koutilya/Make3D/Test134/img-060705-17.51.18-p-018t000.jpg',
# 'img3': '/vulcan/scratch/koutilya/Make3D/Test134/img-10.21op6-p-139t000.jpg',
# 'img4': '/vulcan/scratch/koutilya/Make3D/Test134/img-gatesback1-p-139t0.jpg',
# 'img5': '/vulcan/scratch/koutilya/Make3D/Test134/img-math13-p-139t0.jpg',
# 'img6': '/vulcan/scratch/koutilya/Make3D/Test134/img-op1-p-282t000.jpg',
# 'img7': '/vulcan/scratch/koutilya/Make3D/Test134/img-op2-p-015t000.jpg',
# 'img8': '/vulcan/scratch/koutilya/Make3D/Test134/img-op27-p-015t000.jpg',
# 'img9': '/vulcan/scratch/koutilya/Make3D/Test134/img-op33-p-139t000.jpg',
# 'img10': '/vulcan/scratch/koutilya/Make3D/Test134/img-op39-p-108t000',
# 'img11': '/vulcan/scratch/koutilya/Make3D/Test134/img-op52-p-108t000',
# 'img12': '/vulcan/scratch/koutilya/Make3D/Test134/img-stats11-p-281t0.jpg'}


target_filenames = {'img10': 'img-op39-p-108t000',
'img11': 'img-op52-p-108t000'}

# target_filenames = {'m3-img1.png':'/vulcan/scratch/koutilya/Make3D/Test134/img-statroad2-p-169t0.jpg',
# 'm3-img2.png':'/vulcan/scratch/koutilya/Make3D/Test134/img-stats10-p-108t0.jpg'}


test_dataset = Make3D_dataset()
print(len(test_dataset))
for idx in tqdm(range(len(test_dataset))):
    image, rgb_filename, depth_filename = test_dataset[idx]
    print(rgb_filename)
    if rgb_filename in target_filenames.values():
        print(idx, target_filenames.keys()[target_filenames.values().index(rgb_filename)], depth_filename)
    # print(torch.min(depth), torch.max(depth))
    # print(image.shape, depth.shape)
    # print(torch.min(image), torch.max(image))
    # image_numpy = (1.0+ image.numpy().transpose(1,2,0)) / 2.0
    # imsave('/vulcan/scratch/koutilya/projects/Domain_Adaptation/Common_Domain_Adaptation-Lighting/make3d_images/'+str(idx).zfill(5)+'_image.png',image_numpy)

# test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
# for i,data in enumerate(test_loader):
#     print(i)

# print(torch.min(depth), torch.max(depth))
# fig = plt.figure()
# fig.add_subplot(1,2,1).imshow(image.numpy().transpose(1,2,0))
# fig.add_subplot(1,2,2).imshow(depth.numpy().transpose(1,2,0)[:,:,0])
# plt.show()
