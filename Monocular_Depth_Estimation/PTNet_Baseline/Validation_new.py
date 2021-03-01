import os
import os.path as osp
import glob
import numpy as np
import random
from tqdm import tqdm
import imageio
import cv2

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tr
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import all_networks

from Dataloaders.Kitti_dataloader import KittiDataset as real_dataset
from Dataloaders.transform import *
from Dataloaders.Kitti_dataset_util import KITTI

class Solver():
    def __init__(self, exp):
        self.root_dir = '/vulcanscratch/koutilya/projects/Domain_Adaptation/Common_Domain_Adaptation-Lighting/PTNet_Baseline'
        
        # Seed
        self.seed = 1729
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Initialize networks
        self.netT = all_networks.define_G(3, 1, 64, 4, 'batch',
                                            'PReLU', 'UNet', 'kaiming', 0,
                                            False, [0], 0.1)
        self.netT.cuda()

        # Initialize Loss
        self.netT_loss_fn = nn.L1Loss()

        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Training Configuration details
        self.batch_size = 16
        joint_transform_list = [RandomImgAugment(no_flip=True, no_rotation=True, no_augment=True, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]

        self.joint_transform = tr.Compose(joint_transform_list)

        self.img_transform = tr.Compose(img_transform_list)

        self.depth_transform = tr.Compose([DepthToTensor()])

        self.exp = exp
        if self.exp == 'PTNet_Baseline_NEW':
            self.model_string = ''
        elif self.exp == 'PTNet_Baseline_bicubic_NEW':
            self.model_string = '_bicubic'

        self.writer = SummaryWriter(os.path.join(self.root_dir,'../tensorboard_logs/Vkitti-kitti/test/'+self.exp))

        # Initialize Data
        self.get_validation_data()

        self.garg_crop = True
        self.eigen_crop = False
        self.kitti = KITTI()

    def compute_errors(self, ground_truth, predication):

        # accuracy
        threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
        a1 = (threshold < 1.25 ).mean()
        a2 = (threshold < 1.25 ** 2 ).mean()
        a3 = (threshold < 1.25 ** 3 ).mean()

        #MSE
        rmse = (ground_truth - predication) ** 2
        rmse = np.sqrt(rmse.mean())

        #MSE(log)
        rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        # Abs Relative difference
        abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

        # Squared Relative difference
        sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

        return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


    def get_validation_data(self):
        self.real_val_dataset = real_dataset(data_file='test.txt',phase='test',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        
    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, 'saved_models', 'PTNet_baseline-'+str(self.iteration)+self.model_string+'.pth.tar' ))
        if len(saved_models)>0:
            model_state = torch.load(saved_models[0])
            self.netT.load_state_dict(model_state['netT_state_dict'])
            # self.iteration = model_state['iteration']
            return True
        return False

    def tensor2im(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy*80.0
        
    def get_depth_manually(self, depth_file):
        root_dir = '/vulcanscratch/koutilya/kitti/Depth_from_velodyne/'
        depth_split = depth_file.split('/')
        main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.png')
        
        depth = Image.open(main_file)
        depth = np.array(depth, dtype=np.float32) / 255.0
        return depth
    
    def Validate(self):
        self.netT.eval()
        saved_models_list = glob.glob(os.path.join(self.root_dir, 'saved_models', 'PTNet_baseline-*999'+self.model_string+'.pth.tar' ))
        for self.iteration in range(999,1000*len(saved_models_list),1000):
            self.load_prev_model()
            self.Validation()

    def Validation(self):
        num_samples = len(self.real_val_dataset)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples,np.float32)
        rmse = np.zeros(num_samples,np.float32)
        rmse_log = np.zeros(num_samples,np.float32)
        a1 = np.zeros(num_samples,np.float32)
        a2 = np.zeros(num_samples,np.float32)
        a3 = np.zeros(num_samples,np.float32)

        with torch.no_grad():
            for i,(data, depth_filenames) in tqdm(enumerate(self.real_val_dataloader)): 
                self.real_val_image = data['left_img']#, data['depth'] # self.real_depth is a numpy array 
                self.real_val_image = Variable(self.real_val_image.cuda())

                depth = self.netT(self.real_val_image)
                depth = depth[-1]
                depth_numpy = self.tensor2im(depth) # 0-80m
                for t_id in range(depth_numpy.shape[0]):
                    t_id_global = (i*self.batch_size)+t_id
                    # _,_,_,ground_depth = self.real_val_dataset.read_data(self.real_val_dataset.files[(i*self.batch_size)+t_id])
                    h, w = self.real_val_image.shape[2], self.real_val_image.shape[3]
                    datafiles1 = self.real_val_dataset.files[t_id_global]
                    ground_depth = self.get_depth_manually(datafiles1['depth']) 
                    height, width = ground_depth.shape

                    predicted_depth = cv2.resize(depth_numpy[t_id],(width, height),interpolation=cv2.INTER_LINEAR)
                    predicted_depth[predicted_depth < 1.0] = 1.0
                    # predicted_depth[predicted_depth < 1.0] = 1.0
                    predicted_depth[predicted_depth > 50.0] = 50.0

                    mask = np.logical_and(ground_depth > 1.0, ground_depth < 50.0)
                    
                    # crop used by Garg ECCV16
                    if self.garg_crop:
                        self.crop = np.array([0.40810811 * height,  0.99189189 * height,
                                            0.03594771 * width,   0.96405229 * width]).astype(np.int32)

                    # crop we found by trail and error to reproduce Eigen NIPS14 results
                    elif self.eigen_crop:
                        self.crop = np.array([0.3324324 * height,  0.91351351 * height,
                                            0.0359477 * width,   0.96405229 * width]).astype(np.int32)

                    crop_mask = np.zeros(mask.shape)
                    crop_mask[self.crop[0]:self.crop[1],self.crop[2]:self.crop[3]] = 1
                    mask = np.logical_and(mask, crop_mask)

                    
                    abs_rel[t_id_global], sq_rel[t_id_global], rmse[t_id_global], rmse_log[t_id_global], a1[t_id_global], a2[t_id_global], a3[t_id_global] = self.compute_errors(ground_depth[mask],predicted_depth[mask])

                    # print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                    #     .format(t_id, abs_rel[t_id], sq_rel[t_id], rmse[t_id], rmse_log[t_id], a1[t_id], a2[t_id], a3[t_id]))

            print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
            print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))

            self.writer.add_scalars('Kitti_Validatoin_metrics/Abs_Rel', {self.exp: abs_rel.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/Sq_Rel', {self.exp: sq_rel.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/RMSE', {self.exp: rmse.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/RMSE_log', {self.exp: rmse_log.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/del<1.25', {self.exp: a1.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/del<1.25^2', {self.exp: a2.mean()}, self.iteration)
            self.writer.add_scalars('Kitti_Validatoin_metrics/del<1.25^3', {self.exp: a3.mean()}, self.iteration)

        self.writer.close()

        # self.writer.close()

if __name__=='__main__':
    solver = Solver(exp='PTNet_Baseline_NEW')
    solver.Validate()