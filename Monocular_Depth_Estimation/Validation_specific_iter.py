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
from scipy.misc import imsave
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
        joint_transform_list = [transf.RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])
        
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        self.get_validation_data()

        self.garg_crop = True
        self.eigen_crop = False
        self.flag = True

        self.opt.max_depth = float(self.opt.max_depth)

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

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
        self.syn_val_dataset = syn_dataset(train=False)
        self.syn_val_dataloader = DataLoader(self.syn_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        self.syn_val_sample_dataloader = DataLoader(self.syn_val_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.syn_val_sample_images, self.syn_label = next(iter(self.syn_val_sample_dataloader))
        self.syn_val_sample_images = Variable(self.syn_val_sample_images.cuda())
        self.syn_label = (1.0 + self.syn_label) / 2.0

        self.real_val_dataset = real_dataset(data_file='test.txt',phase='test',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        self.real_val_sample_dataloader = DataLoader(self.real_val_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.real_val_sample_images, self.real_val_sample_filenames = next(iter(self.real_val_sample_dataloader))
        self.real_val_sample_images = self.real_val_sample_images['left_img']
        self.real_val_sample_images = Variable(self.real_val_sample_images.cuda())

    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_WI_geom_bicubic_da-'+str(self.iteration)+'.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[-1].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            print(saved_model)
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            return True
        return False

    def tensor2im(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy*80.0
    
    def tensor2im_only(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy

    def Validate_all(self):
        for self.iteration in range(self.iteration,1000+self.iteration, 1000):
            self.load_prev_model()
            self.Validation()
            

    def get_depth_manually(self, depth_file):
        root_dir = '/vulcanscratch/koutilya/kitti/Depth_from_velodyne_npy/'
        # root_dir = '/vulcan/scratch/koutilya/kitti/Depth_from_velodyne/'
        depth_split = depth_file.split('/')
        # main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.png')
        main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.npy')
        
        depth = np.load(main_file)
        # depth = Image.open(main_file)
        # depth = np.array(depth, dtype=np.float32) / 255.0
        return depth

    def Validation_extract_images(self):
        self.load_prev_model()
        
        with torch.no_grad():
            _, self.real_val_sample_translated_images = self.netG(self.real_val_sample_images)
            _, self.syn_val_sample_translated_images = self.netG(self.syn_val_sample_images)
            sample_depth = self.netT(self.real_val_sample_translated_images)
            syn_sample_depth = self.netT(self.syn_val_sample_translated_images)

            sample_depth = sample_depth[-1]
            ###### code to save predicted depth maps - Validation_specific_test_examples
            real_pred_depth_all = self.tensor2im_only(sample_depth) # 0-1
            real_pred_depth_all *= 65535
            real_pred_depth_all[real_pred_depth_all<1e-3] = 1e-3
            ######
            sample_depth = sample_depth.data
            sample_depth = (1.0+sample_depth)/2.0

            syn_sample_depth = syn_sample_depth[-1]
            ###### code to save predicted depth maps - Validation_specific_test_examples
            syn_pred_depth_all = self.tensor2im_only(syn_sample_depth) # 0-1
            syn_pred_depth_all *= 65535
            syn_pred_depth_all[syn_pred_depth_all<1e-3] = 1e-3
            ######
            syn_sample_depth = syn_sample_depth.data
            syn_sample_depth = (1.0+syn_sample_depth)/2.0

            sample_depth_colorized = self.colorize(sample_depth,cmap=matplotlib.cm.get_cmap('plasma'))
            sample_depth_colorized = torch.from_numpy(sample_depth_colorized.squeeze()).permute(0,3,1,2)[:,:3,:,:]
            syn_sample_depth_colorized = self.colorize(syn_sample_depth,cmap=matplotlib.cm.get_cmap('plasma'))
            syn_sample_depth_colorized = torch.from_numpy(syn_sample_depth_colorized.squeeze()).permute(0,3,1,2)[:,:3,:,:]
            syn_label_colorized = self.colorize(self.syn_label,cmap=matplotlib.cm.get_cmap('plasma'))
            syn_label_colorized = torch.from_numpy(syn_label_colorized.squeeze()).permute(0,3,1,2)[:,:3,:,:]
        
        # if not self.opt.save_images:
        #     self.writer.add_image('Real Images',torchvision.utils.make_grid((1.0+self.real_val_sample_images)/2.0,nrow=4), self.iteration)
        #     self.writer.add_image('Syn Images',torchvision.utils.make_grid((1.0+self.syn_val_sample_images)/2.0,nrow=4), self.iteration)
        #     self.writer.add_image('Real Translated Images',torchvision.utils.make_grid((1.0+self.real_val_sample_translated_images)/2.0,nrow=4), self.iteration)
        #     self.writer.add_image('Syn Translated Images',torchvision.utils.make_grid((1.0+self.syn_val_sample_translated_images)/2.0,nrow=4), self.iteration)
        #     self.writer.add_image('Predicted Depth',torchvision.utils.make_grid(sample_depth_colorized,nrow=4), self.iteration)
        #     self.writer.add_image('Syn Predicted Depth',torchvision.utils.make_grid(syn_sample_depth_colorized,nrow=4), self.iteration)
        #     self.writer.add_image('Syn GT Depth',torchvision.utils.make_grid(syn_label_colorized,nrow=4), self.iteration)

        #     self.writer.close()

        if self.opt.save_images:
            if not osp.exists(osp.join(self.root_dir,'Results',str(self.iteration))):
                os.system('mkdir -p '+osp.join(self.root_dir,'Results',str(self.iteration)))
            
            syn_val_images = (1.0+self.syn_val_sample_images)/2.0
            syn_val_translated_images = (1.0+self.syn_val_sample_translated_images)/2.0
            syn_val_images = syn_val_images.cpu().data.numpy().transpose(0,2,3,1)
            syn_val_translated_images = syn_val_translated_images.cpu().data.numpy().transpose(0,2,3,1)
            
            real_val_images = (1.0+self.real_val_sample_images)/2.0
            real_val_translated_images = (1.0+self.real_val_sample_translated_images)/2.0
            real_val_images = real_val_images.cpu().data.numpy().transpose(0,2,3,1)
            real_val_translated_images = real_val_translated_images.cpu().data.numpy().transpose(0,2,3,1)
            for k in range(self.syn_val_sample_images.shape[0]):
                imsave(osp.join(self.root_dir,'Results', str(self.iteration), 'Syn_'+str(k)+'.jpg'), syn_val_images[k])
                imsave(osp.join(self.root_dir,'Results', str(self.iteration), 'Syn_translated_'+str(k)+'.jpg'), syn_val_translated_images[k])
                imsave(osp.join(self.root_dir,'Results', str(self.iteration), 'Real_'+str(k)+'.jpg'), real_val_images[k])
                imsave(osp.join(self.root_dir,'Results', str(self.iteration), 'Real_translated_'+str(k)+'.jpg'), real_val_translated_images[k])

                pred_depth = real_pred_depth_all[k]
                pred_img = Image.fromarray(pred_depth.squeeze().astype(np.int32), 'I')
                pred_img.save(osp.join(self.root_dir,'Results', str(self.iteration), 'Real_predicted_depth_'+str(k)+'.png'))

                pred_depth = syn_pred_depth_all[k]
                pred_img = Image.fromarray(pred_depth.squeeze().astype(np.int32), 'I')
                pred_img.save(osp.join(self.root_dir,'Results', str(self.iteration), 'Syn_predicted_depth_'+str(k)+'.png'))



    def validate(self):
        self.netG.eval()
        self.netT.eval()
        self.iteration = self.opt.iter
        if self.opt.gen_images:
            self.Validation_extract_images()
        else:
            self.Validate_all()

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
                # depth = self.netT(self.real_val_image)
                
                depth = depth[-1]
                depth_numpy = self.tensor2im(depth) # 0-80m
                
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
                    # save_pred_depth = predicted_depth * 65535/80.0
                    # save_pred_depth[save_pred_depth<1e-3] = 1e-3
       
                    # save_pred_img = Image.fromarray(save_pred_depth.astype(np.int32), 'I')
                    # save_pred_img.save('%s/%05d_pred.png'%('results', t_id_global))

                    predicted_depth[predicted_depth < self.opt.min_depth] = self.opt.min_depth
                    predicted_depth[predicted_depth > self.opt.max_depth] = self.opt.max_depth

                    mask = np.logical_and(ground_depth > self.opt.min_depth, ground_depth < self.opt.max_depth)
                    
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

            # self.writer.add_scalar('Kitti_Validatoin_metrics/Abs_Rel', abs_rel.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/Sq_Rel', sq_rel.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE', rmse.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE_log', rmse_log.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25', a1.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^2', a2.mean(), self.iteration)
            # self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^3', a3.mean(), self.iteration)

        # self.writer.close()

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', default=999, type=int, help="Indicate what iteration of the saved model for Validation")
    parser.add_argument('--max_depth', default=50, type=int, help="Indicate what max depth for Validation")
    parser.add_argument('--min_depth', default=1.0, type=float, help="Indicate what min depth for Validation")
    parser.add_argument('--gen_images', action='store_true', help="Indicate if you want to save the real and syn image results")
    parser.add_argument('--save_images', action='store_true', help="Indicate if you want to save the real and syn image results to memory")
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.validate()