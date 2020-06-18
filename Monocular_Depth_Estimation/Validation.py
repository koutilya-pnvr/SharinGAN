import os
import os.path as osp
import glob
import numpy as np
import random
from tqdm import tqdm
import argparse
import matplotlib
import matplotlib.cm
from PIL import Image
import cv2
import imageio
import time

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

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset
from Dataloaders.Kitti_dataloader import DepthToTensor, KittiDataset as real_dataset
import Dataloaders.transform as transf


class Solver():
    def __init__(self, opt):
        self.root_dir = '.'
        self.opt = opt
        self.val_string = 'test'
        if self.opt.val:
            self.val_string = 'val'
        
        # Seed
        self.seed = 1729
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Initialize networks
        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                  'PReLU', 'ResNet', 'kaiming', 0,
                                                  False, [0])

        self.netG.cuda()
        
        self.netT = all_networks.define_G(3, 1, 64, 4, 'batch',
                                            'PReLU', 'UNet', 'kaiming', 0,
                                            False, [0], 0.1)
        self.netG.cuda()
        self.netT.cuda()
        
        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        self.netT_loss_fn = nn.L1Loss()
        self.netG_loss_fn = self.netG_loss_fn.cuda()
        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Training Configuration details
        self.batch_size = 16
        self.iteration = None
        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=True, no_rotation=True, no_augment=True, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])
        
        self.writer = SummaryWriter(os.path.join(self.root_dir,'tensorboard_logs/Vkitti-kitti',self.val_string))
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        self.get_validation_data()

        self.garg_crop = True
        self.eigen_crop = False
        self.flag = True

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

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
        if self.opt.val:
            self.real_val_dataset = real_dataset(data_file='val.txt',phase='val',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        else:
            self.real_val_dataset = real_dataset(data_file='test.txt',phase='test',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        self.real_val_sample_dataloader = DataLoader(self.real_val_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.real_val_sample_images, self.real_val_sample_filenames = next(iter(self.real_val_sample_dataloader))
        self.real_val_sample_images = self.real_val_sample_images['left_img']
        self.real_val_sample_images = Variable(self.real_val_sample_images.cuda())

    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da-'+str(self.iteration)+'.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[2].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            return True
        return False

    def tensor2im(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy*80.0

    def save_image(self,depth, filename):
        imageio.imwrite(filename, depth)

    def colorize(self,value, vmin=None, vmax=None, cmap=None):
        """
        A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
        colormap for use with TensorBoard image summaries.
        By default it will normalize the input value to the range 0..1 before mapping
        to a grayscale colormap.
        Arguments:
        - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
            [height, width, 1].
        - vmin: the minimum value of the range used for normalization.
            (Default: value minimum)
        - vmax: the maximum value of the range used for normalization.
            (Default: value maximum)
        - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
            (Default: Matplotlib default colormap)
        
        Returns a 4D uint8 tensor of shape [height, width, 4].
        """

        # # normalize
        # vmin = value.min() if vmin is None else vmin
        # vmax = value.max() if vmax is None else vmax
        # if vmin!=vmax:
        #     value = (value - vmin) / (vmax - vmin) # vmin..vmax
        # else:
        #     # Avoid 0-division
        #     value = value*0.
        # # squeeze last dim if it exists
        # value = value.squeeze()

        cmapper = matplotlib.cm.get_cmap(cmap)
        value = cmapper(value,bytes=True) # (nxmx4)
        return value
    
    def Validate(self):
        self.netG.eval()
        self.netT.eval()
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da*.pth.tar' ))
        START_ITER = max(self.opt.start_iter,999)
        for self.iteration in range(START_ITER,1000*len(saved_models)+999, 1000):
            self.load_prev_model()
            self.Validation()
        self.writer.close()

    def get_depth_manually(self, depth_file):
        root_dir = '/vulcan/scratch/koutilya/kitti/Depth_from_velodyne_npy/' # path to velodyne data converted to numpy
        # root_dir = '/vulcan/scratch/koutilya/kitti/Depth_from_velodyne/'
        depth_split = depth_file.split('/')
        # main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.png')
        if self.opt.val:
            main_file = osp.join(root_dir, 'val', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.npy')
        else:
            main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.npy')

        depth = np.load(main_file)
        # depth = Image.open(main_file)
        # depth = np.array(depth, dtype=np.float32) / 255.0
        return depth

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

                _, self.real_translated_image = self.netG(self.real_val_image)
                
                depth = self.netT(self.real_translated_image)
                depth = depth[-1]
                depth_numpy = self.tensor2im(depth) # 0-80m
                
                if i==0:
                    if self.flag:
                        self.writer.add_image('Real Images',torchvision.utils.make_grid((1.0+self.real_val_sample_images)/2.0,nrow=4), self.iteration)
                        self.flag=False

                    _, self.real_val_sample_translated_images = self.netG(self.real_val_sample_images)
                    sample_depth = self.netT(self.real_val_sample_translated_images)
                    sample_depth = sample_depth[-1]
                    sample_depth = sample_depth.data
                    sample_depth = (1.0+sample_depth)/2.0
                    sample_depth_colorized = self.colorize(sample_depth,cmap=matplotlib.cm.get_cmap('plasma'))
                    sample_depth_colorized = torch.from_numpy(sample_depth_colorized.squeeze()).permute(0,3,1,2)[:,:3,:,:]
                    self.writer.add_image('Real Translated Images',torchvision.utils.make_grid((1.0+self.real_val_sample_translated_images)/2.0,nrow=4), self.iteration)
                    self.writer.add_image('Predicted Depth',torchvision.utils.make_grid(sample_depth_colorized,nrow=4), self.iteration)

                for t_id in range(depth_numpy.shape[0]):
                    t_id_global = (i*self.batch_size)+t_id
                    # _,_,_,ground_depth = self.real_val_dataset.read_data(self.real_val_dataset.files[(i*self.batch_size)+t_id])
                    h, w = self.real_val_image.shape[2], self.real_val_image.shape[3]
                    datafiles1 = self.real_val_dataset.files[t_id_global]
                    # ground_depth = self.kitti.get_depth(osp.join(self.real_val_dataset.root, datafiles1['cam_intrin']),
                    #             osp.join(self.real_val_dataset.root, datafiles1['depth']), [h, w])
                    # ground_depth = ground_depth.astype(np.float32)
                    ground_depth = self.get_depth_manually(datafiles1['depth']) 
                    height, width = ground_depth.shape

                    predicted_depth = cv2.resize(depth_numpy[t_id],(width, height),interpolation=cv2.INTER_LINEAR)
                    predicted_depth[predicted_depth < 1.0] = 1.0
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

            ##################################################
            ### Tensorboard Logging
            ##################################################            
            # self.writer.add_scalars('Kitti_Validation_metrics/'+self.exp,  {'Abs Rel':abs_rel.mean(),
            # 'Sq Rel': sq_rel.mean(),
            # 'RMSE': rmse.mean(),
            # 'RMSE_log':rmse_log.mean(),
            # 'del<1.25':a1.mean(),
            # 'del<1.25^2':a2.mean(),
            # 'del<1.25^3':a3.mean(),}, self.iteration)

            self.writer.add_scalar('Kitti_Validatoin_metrics/Abs_Rel', abs_rel.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/Sq_Rel', sq_rel.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE', rmse.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE_log', rmse_log.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25', a1.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^2', a2.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^3', a3.mean(), self.iteration)


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--val', action='store_true', help='Indicate if you want to test the model on the validation data')
    parser.add_argument('--start_iter', default=999, type=int, help="Indicate what iteration of the saved model to be started with for Validation")
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.Validate()