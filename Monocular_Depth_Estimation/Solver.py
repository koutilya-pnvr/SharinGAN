import os
import glob
import numpy as np
import random
from tqdm import tqdm
import matplotlib
import matplotlib.cm
import cv2

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms as tr
from tensorboardX import SummaryWriter

from networks import all_networks
from networks.da_net import Discriminator
from Geometric_consistency_loss import *

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset
from Dataloaders.Kitti_dataloader_training import DepthToTensor, KittiDataset as real_dataset
from Dataloaders.Kitti_dataloader import KittiDataset as real_val_dataset
import Dataloaders.transform as transf

class Solver():
    def __init__(self, opt):
        self.root_dir = '.'
        self.opt = opt
        # Seed
        self.seed = 1729 # The famous Hardy-Ramanujan number
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Initialize networks
        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                  'PReLU', 'ResNet', 'kaiming', 0,
                                                  False, [0])
        self.netD = [Discriminator(nout=1, last_layer_activation=False)]
        
        self.netT = all_networks.define_G(3, 1, 64, 4, 'batch',
                                            'PReLU', 'UNet', 'kaiming', 0,
                                            False, [0], 0.1, uncertainty=True)
        self.netG.cuda()
        self.netT.cuda()
        for disc in self.netD:
            disc.cuda()

        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        self.netD_loss_fn = nn.KLDivLoss()
        self.netT_loss_fn = nn.L1Loss()
        self.netT_loss_fn_individual = nn.L1Loss(reduction='none')
        self.geometric_loss_fn = ReconLoss()

        self.netG_loss_fn = self.netG_loss_fn.cuda()
        self.netD_loss_fn = self.netD_loss_fn.cuda()
        self.netT_loss_fn = self.netT_loss_fn.cuda()
        self.netT_loss_fn_individual = self.netT_loss_fn_individual.cuda()

        # Initialize Optimizers
        self.netG_optimizer = Optim.Adam(self.netG.parameters(), lr=1e-5)
        self.netD_optimizer = []
        for disc in self.netD:
            self.netD_optimizer.append(Optim.Adam(disc.parameters(), lr=1e-5))
        self.netT_optimizer = Optim.Adam(self.netT.parameters(), lr=1e-5)

        # Training Configuration details
        self.batch_size = 2
        self.iteration = None
        self.total_iterations = 200000
        self.START_ITER = 0
        self.flag = True
        self.garg_crop = True
        self.eigen_crop = False
        self.best_a1 = 0.0

        self.kr = 1
        self.kd = 1 
        self.kcritic = 5
        self.gamma = 10
        
        # Transforms
        joint_transform_list = [transf.RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])
        
        self.writer = SummaryWriter(os.path.join(self.root_dir,'tensorboard_logs/Vkitti-kitti/train_uncertainty'))
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        self.get_training_data()
        self.get_training_dataloaders()
        self.get_validation_data()

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

    def get_training_data(self):
        self.syn_loader = DataLoader(syn_dataset(train=True), batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.real_loader = DataLoader(real_dataset(img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform), batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    def get_training_dataloaders(self):
        self.syn_iter = self.loop_iter(self.syn_loader)
        self.real_iter = self.loop_iter(self.real_loader)

    def load_my_state_dict(self, student, teacher_state_dict):
 
        own_state = student.state_dict()
        for name, param in teacher_state_dict.items():
            if name not in own_state:
                 continue
            if isinstance(param, nn.parameter.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            own_state[name].copy_(param)

    def load_pretrained_models(self):
        model_state = torch.load(os.path.join(self.root_dir, 'Gen_Baseline/saved_models/Gen_Resnet_Baseline.pth.tar'))
        
        self.netG.load_state_dict(model_state['netG_state_dict'])
        self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
        
        model_state = torch.load(os.path.join(self.root_dir, 'PTNet_Baseline/saved_models/PTNet_baseline-8999_bicubic.pth.tar'))
        self.load_my_state_dict(self.netT, model_state['netT_state_dict'])
        # self.load_my_state_dict(self.netT_optimizer, model_state['netT_optimizer'])
        # self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            
    def load_prev_model(self, model_status='latest'):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_WI_geom_bicubic_uncertainty_da-'+model_status+'.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[-1].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
            self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            
            for i,disc in enumerate(self.netD):
                disc.load_state_dict(model_state['netD'+str(i)+'_state_dict'])
                self.netD_optimizer[i].load_state_dict(model_state['netD'+str(i)+'_optimizer_state_dict'])
            
            self.START_ITER = model_state['iteration']+1
            return True
        return False

    def save_model(self, model_status='latest'):
        if not os.path.exists(os.path.join(self.root_dir, self.saved_models_dir)):
            os.mkdir(os.path.join(self.root_dir, self.saved_models_dir))
        
        dict1 = {
                'iteration': self.iteration,
                'netG_state_dict': self.netG.state_dict(),
                'netT_state_dict': self.netT.state_dict(),
                'netG_optimizer': self.netG_optimizer.state_dict(),
                'netT_optimizer': self.netT_optimizer.state_dict(),
                }
        dict2 = {'netD'+str(i)+'_state_dict':disc.state_dict() for i,disc in enumerate(self.netD)}
        dict3 = {'netD'+str(i)+'_optimizer_state_dict':self.netD_optimizer[i].state_dict() for i,disc in enumerate(self.netD)}
        final_dict = dict(dict1.items()+dict2.items()+dict3.items())
        torch.save(final_dict, os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_uncertainty-da_tmp.pth.tar'))
        os.system('mv '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_uncertainty-da_tmp.pth.tar')+' '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_WI_geom_bicubic_uncertainty_da-'+model_status+'.pth.tar'))
        
    def get_syn_data(self):
        self.syn_image, self.syn_label = next(self.syn_iter)
        self.syn_image, self.syn_label = Variable(self.syn_image.cuda()), Variable(self.syn_label.cuda())
        self.syn_label_scales = self.scale_pyramid(self.syn_label, 5-1)
        
    def get_real_data(self):
        self.real_image, self.real_right_image, self.fb = next(self.real_iter)
        self.real_image = Variable(self.real_image.cuda())
        self.real_right_image = Variable(self.real_right_image.cuda())
        self.real_image_scales = self.scale_pyramid(self.real_image, 5-1)
        self.real_right_image_scales = self.scale_pyramid(self.real_right_image, 5-1)
        
    def gradient_penalty(self, model, h_s, h_t):
        # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
        batch_size =min(h_s.size(0), h_t.size(0))
        h_s = h_s[:batch_size]
        h_t = h_t[:batch_size]
        size = len(h_s.shape)
        alpha = torch.rand(batch_size)#, 1, 1, 1)
        for ki in range(1,size):
            alpha = alpha.unsqueeze(ki)
        alpha = alpha.expand_as(h_s)
        alpha = alpha.cuda()
        differences = h_t - h_s
        interpolates = h_s + (alpha * differences)
        # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
        interpolates = Variable(interpolates.cuda(), requires_grad=True)
        preds = model(interpolates)
        gradients = torch.autograd.grad(preds, interpolates,
                            grad_outputs=torch.ones_like(preds).cuda(),
                            retain_graph=True, create_graph=True)[0]
        gradients = gradients.view(batch_size,-1) 
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1)**2).mean()
        return gradient_penalty
    
    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]

        s = img.size()

        h = s[2]
        w = s[3]

        for i in range(1, num_scales):
            ratio = 2**i
            nh = h // ratio
            nw = w // ratio
            scaled_img = torch.nn.functional.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=True)
            scaled_imgs.append(scaled_img)

        scaled_imgs.reverse()
        return scaled_imgs
    
    def gradient_x(self,img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx


    def gradient_y(self,img):
        gy = img[:, :, :, :-1] - img[:, :, :, 1:]
        return gy


    # calculate the gradient loss
    def get_smooth_weight(self, depths, Images, num_scales):

        depth_gradient_x = [self.gradient_x(d) for d in depths]
        depth_gradient_y = [self.gradient_y(d) for d in depths]

        Image_gradient_x = [self.gradient_x(img) for img in Images]
        Image_gradient_y = [self.gradient_y(img) for img in Images]

        weight_x = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_x]
        weight_y = [torch.exp(-torch.mean(torch.abs(g), 1, keepdim=True)) for g in Image_gradient_y]

        smoothness_x = [depth_gradient_x[i] * weight_x[i] for i in range(num_scales)]
        smoothness_y = [depth_gradient_y[i] * weight_y[i] for i in range(num_scales)]

        loss_x = [torch.mean(torch.abs(smoothness_x[i]))/2**i for i in range(num_scales)]
        loss_y = [torch.mean(torch.abs(smoothness_y[i]))/2**i for i in range(num_scales)]

        return sum(loss_x+loss_y)
    
    def reset_netD_grad(self, i=None):
        if i==None:
            for disc_op in self.netD_optimizer:
                disc_op.zero_grad()
        else:
            for idx, disc_op in enumerate(self.netD):
                if idx==i:
                    continue
                else:
                    disc_op.zero_grad()

    def reset_grad(self, exclude=None):
        if(exclude==None):
            self.netG_optimizer.zero_grad()
            self.reset_netD_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netG'):
            self.reset_netD_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netD'):
            self.netG_optimizer.zero_grad()
            self.netT_optimizer.zero_grad()
        elif(exclude=='netT'):
            self.netG_optimizer.zero_grad()
            self.reset_netD_grad()

    def forward_netD(self, mode='gen'):
        self.D_real = [self.netD[0](self.real_recon_image)]
        self.D_syn = [self.netD[0](self.syn_recon_image)]
    
    def loss_from_disc(self, mode='gen'):
        self.just_adv_loss = self.D_syn[0].mean() - self.D_real[0].mean()
        if mode == 'disc':
            self.just_adv_loss = -1* self.just_adv_loss
        
    def set_requires_grad(self, model, mode=False):
        for param in model.parameters():
            param.requires_grad = mode 

    def update_netG(self):
        
        self.set_requires_grad(self.netT, False)
        for disc in self.netD:
            self.set_requires_grad(disc, False)

        self.real_features, self.real_recon_image = self.netG(self.real_image)
        self.syn_features, self.syn_recon_image = self.netG(self.syn_image)
        
        _, real_uncertainty, real_depth = self.netT(self.real_recon_image)
        _, syn_uncertainty, syn_depth = self.netT(self.syn_recon_image)
        self.real_predicted_depth, self.syn_predicted_depth = real_depth[-1], syn_depth[-1]

        self.forward_netD()
        self.loss_from_disc()

        real_reconstruction_loss, syn_reconstruction_loss = self.netG_loss_fn(self.real_recon_image, self.real_image), self.netG_loss_fn(self.syn_recon_image, self.syn_image)
        real_size = len(real_depth)
        gradient_smooth_loss = self.get_smooth_weight(real_depth[1:], self.real_image_scales, real_size-1)
        
        task_loss = 0.0
        syn_probability_loss = torch.zeros(2).cuda()
        real_probability_loss = torch.zeros(2).cuda()
        for (lab_fake_i, lab_real_i, uncertainty_output) in zip(syn_depth[1:], self.syn_label_scales, syn_uncertainty[1:]):
            task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)
            syn_probability_loss_temp = self.netT_loss_fn_individual(lab_fake_i, lab_real_i)
            syn_probability_loss_temp /= uncertainty_output
            syn_probability_loss_temp += torch.log(uncertainty_output)
            syn_probability_loss += (syn_probability_loss_temp.view(syn_probability_loss_temp.size(0),-1).mean(1))
        
        geo_id = 0
        for (l_img, r_img, gen_depth, uncertainty_output) in zip(self.real_image_scales, self.real_right_image_scales, real_depth[1:], real_uncertainty[1:]):
            loss, _ = self.geometric_loss_fn(l_img, r_img, gen_depth, self.fb / 2**(3-geo_id), reduction='none')
            real_probability_loss_temp = loss/uncertainty_output
            real_probability_loss_temp += torch.log(uncertainty_output)
            real_probability_loss += (real_probability_loss_temp.view(real_probability_loss_temp.size(0),-1).mean(1))
            task_loss += loss.mean()
            geo_id += 1
        
        uncertainty_loss = syn_probability_loss + real_probability_loss
        uncertainty_loss = uncertainty_loss.mean()
        self.netG_loss = self.just_adv_loss + (0.01*gradient_smooth_loss) + (100*task_loss) + (100*uncertainty_loss)
        self.netG_loss += 10*(real_reconstruction_loss + syn_reconstruction_loss)
        
        self.reset_grad()
        self.netG_loss.backward()
        self.reset_grad(exclude='netG')
        self.netG_optimizer.step()

        self.set_requires_grad(self.netT, True)
        for disc in self.netD:
            self.set_requires_grad(disc, True)

    def update_netT(self):

        self.set_requires_grad(self.netG, False)
        for disc in self.netD:
            self.set_requires_grad(disc, False)

        _, syn_refined_image = self.netG(self.syn_image)
        _, real_refined_image = self.netG(self.real_image)
        _, syn_uncertainty, syn_depth = self.netT(syn_refined_image)
        _, real_uncertainty, real_depth = self.netT(real_refined_image)

        task_loss = 0.0
        self.syn_probability_loss = torch.zeros(2).cuda()
        self.real_probability_loss = torch.zeros(2).cuda()
        for (lab_fake_i, lab_real_i, uncertainty_output) in zip(syn_depth[1:], self.syn_label_scales, syn_uncertainty[1:]):
            task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)
            syn_probability_loss_temp = self.netT_loss_fn_individual(lab_fake_i, lab_real_i)
            syn_probability_loss_temp /= uncertainty_output
            syn_probability_loss_temp += torch.log(uncertainty_output)
            self.syn_probability_loss += (syn_probability_loss_temp.view(syn_probability_loss_temp.size(0),-1).mean(1))
        
        geo_id = 0
        for (l_img, r_img, gen_depth, uncertainty_output) in zip(self.real_image_scales, self.real_right_image_scales, real_depth[1:], real_uncertainty[1:]):
            loss, _ = self.geometric_loss_fn(l_img, r_img, gen_depth, self.fb / 2**(3-geo_id), reduction='none')
            real_probability_loss_temp = loss/uncertainty_output
            real_probability_loss_temp += torch.log(uncertainty_output)
            self.real_probability_loss += (real_probability_loss_temp.view(real_probability_loss_temp.size(0),-1).mean(1))
            task_loss += loss.mean()
            geo_id += 1
        self.syn_probability_loss = self.syn_probability_loss.mean()
        self.real_probability_loss = self.real_probability_loss.mean()
        self.uncertainty_loss = self.syn_probability_loss + self.real_probability_loss

        _, real_refined_image = self.netG(self.real_image)
        _, real_uncertainty, real_depth = self.netT(real_refined_image)
        real_size = len(real_depth)
        gradient_smooth_loss = self.get_smooth_weight(real_depth[1:], self.real_image_scales, real_size-1)
        
        self.netT_loss = (100*task_loss) + (0.01*gradient_smooth_loss) + (100*self.uncertainty_loss)
        
        self.reset_grad()
        self.netT_loss.backward()
        self.reset_grad(exclude='netT')
        self.netT_optimizer.step()

        self.set_requires_grad(self.netG, True)
        for disc in self.netD:
            self.set_requires_grad(disc, True)

    def update_netD(self):
        
        self.set_requires_grad(self.netG, False)

        with torch.no_grad():
            self.syn_features, self.syn_recon_image  = self.netG(self.syn_image)
            self.real_features, self.real_recon_image = self.netG(self.real_image)

        for _ in range(self.kcritic):
            self.forward_netD(mode='disc')
            self.loss_from_disc(mode='disc')
            
            gp = self.gradient_penalty(self.netD[0], self.syn_recon_image, self.real_recon_image)
            self.netD_loss = self.just_adv_loss + self.gamma*gp
            self.netD_step()

        self.set_requires_grad(self.netG, True)

    def netD_step(self):
        self.reset_grad()
        self.netD_loss.backward()
        self.reset_grad(exclude='netD')
        for disc_op in self.netD_optimizer:
            disc_op.step()

    def train(self):
        self.load_pretrained_models()
        if self.opt.resume:
            self.load_prev_model()
        for self.iteration in tqdm(range(self.START_ITER, self.total_iterations)): 
            
            self.get_syn_data()
            self.get_real_data()
            ###################################################
            #### Update netD
            ###################################################
            
            self.update_netD()
        
            ###################################################
            #### Update netG
            ###################################################            
            
            for i in range(self.kr):
                self.update_netG()

            ###################################################
            #### Update netT
            ###################################################
            
            self.update_netT()

            ###################################################
            #### Tensorboard Logging
            ###################################################            
            self.writer.add_scalar('1) Total Generator loss', self.netG_loss, self.iteration)
            self.writer.add_scalar('2) Total Discriminator loss', self.netD_loss, self.iteration)
            self.writer.add_scalar('3) Total Depth Regressor loss', self.netT_loss, self.iteration)

            if self.iteration % 1000 == 999:
                # Validation and saving models
                self.save_model(model_status='latest')
                self.Validate()
                
        self.writer.close()

    def get_validation_data(self):
        self.real_val_dataset = real_val_dataset(data_file='val.txt',phase='val',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        self.real_val_sample_dataloader = DataLoader(self.real_val_dataset, shuffle=True, batch_size=self.batch_size, num_workers=4)
        self.real_image, self.real_right_image, self.fb = next(self.real_iter)
        self.real_val_sample_images, self.real_val_sample_filenames = next(iter(self.real_val_sample_dataloader))
        self.real_val_sample_images = self.real_val_sample_images['left_img']
        self.real_val_sample_images = Variable(self.real_val_sample_images.cuda())

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

    def tensor2im(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy*80.0

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
        value = cmapper(value.cpu().numpy(),bytes=True) # (nxmx4)
        return value
    
    def get_depth_manually(self, depth_file):
        root_dir = '/vulcanscratch/koutilya/kitti/Depth_from_velodyne_npy/' # path to velodyne data converted to numpy
        depth_split = depth_file.split('/')
        main_file = os.path.join(root_dir, 'val', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.npy')
        depth = np.load(main_file)
        # root_dir = '/vulcanscratch/koutilya/kitti/Depth_from_velodyne/'
        # main_file = osp.join(root_dir, 'test', depth_split[0], depth_split[1], depth_split[-1].split('.')[0]+'.png')
        # depth = Image.open(main_file)
        # depth = np.array(depth, dtype=np.float32) / 255.0
        return depth

    def Validate(self):
        self.netG.eval()
        self.netT.eval()

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
                _, real_recon_image = self.netG(self.real_val_image)
                
                _, _, depth = self.netT(real_recon_image)
                depth = depth[-1]
                depth_numpy = self.tensor2im(depth) # 0-80m
                
                if i==0:
                    if self.flag:
                        self.writer.add_image('Real Images',torchvision.utils.make_grid((1.0+self.real_val_sample_images)/2.0,nrow=4), self.iteration)
                        self.flag=False

                    _, self.real_val_sample_translated_images = self.netG(self.real_val_sample_images)
                    
                    _, _, sample_depth = self.netT(self.real_val_sample_translated_images)
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

            print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
            print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean(),a1.mean(),a2.mean(),a3.mean()))

            ##################################################
            ### Tensorboard Logging
            ##################################################            
            self.writer.add_scalar('Kitti_Validatoin_metrics/Abs_Rel', abs_rel.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/Sq_Rel', sq_rel.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE', rmse.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/RMSE_log', rmse_log.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25', a1.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^2', a2.mean(), self.iteration)
            self.writer.add_scalar('Kitti_Validatoin_metrics/del<1.25^3', a3.mean(), self.iteration)

        if self.best_a1 < a1.mean():
            # Found a new best model
            self.save_model(model_status='best')
            self.best_a1 = a1.mean()

        self.netG.train()
        self.netT.train()