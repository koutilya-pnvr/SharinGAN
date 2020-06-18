import os
import os.path as osp
import glob
import numpy as np
import random
from tqdm import tqdm

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
from networks.da_net import Discriminator
from Geometric_consistency_loss import *

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset
from Dataloaders.Kitti_dataloader_training import DepthToTensor, KittiDataset as real_dataset
import Dataloaders.transform as transf

class Solver():
    def __init__(self, opt):
        self.root_dir = '.'

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
                                            False, [0], 0.1)
        self.netG.cuda()
        self.netT.cuda()
        for disc in self.netD:
            disc.cuda()

        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        self.netD_loss_fn = nn.KLDivLoss()
        self.netT_loss_fn = nn.L1Loss()
        self.geometric_loss_fn = ReconLoss()

        self.netG_loss_fn = self.netG_loss_fn.cuda()
        self.netD_loss_fn = self.netD_loss_fn.cuda()
        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Initialize Optimizers
        self.netG_optimizer = Optim.Adam(self.netG.parameters(), lr=1e-5)
        self.netD_optimizer = []
        for disc in self.netD:
            self.netD_optimizer.append(Optim.Adam(disc.parameters(), lr=1e-5))
        self.netT_optimizer = Optim.Adam(self.netT.parameters(), lr=1e-5)

        # Training Configuration details
        self.batch_size = 2
        self.iteration = None
        self.total_iterations = 2000000
        self.START_ITER = 0

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
        
        self.writer = SummaryWriter(os.path.join(self.root_dir,'tensorboard_logs/Vkitti-kitti/train'))
        self.saved_models_dir = 'saved_models'

        # Initialize Data
        self.get_training_data()
        self.get_training_dataloaders()

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

    def load_pretrained_models(self):
        model_state = torch.load(os.path.join(self.root_dir, 'Gen_Baseline/saved_models_new/AE_Resnet_Baseline.pth.tar'))
        
        self.netG.load_state_dict(model_state['netG_state_dict'])
        self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
        
        model_state = torch.load(os.path.join(self.root_dir, 'UNet_Baseline/saved_models_all_iters/UNet_baseline-8999_bicubic.pth.tar'))
        self.netT.load_state_dict(model_state['netT_state_dict'])
        self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            
    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da*.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[2].split('.')[0]) for s in saved_models]
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

    def save_model(self):
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
        torch.save(final_dict, os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator-da_tmp.pth.tar'))
        os.system('mv '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator-da_tmp.pth.tar')+' '+os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_da-'+str(self.iteration)+'.pth.tar'))
        
    def get_syn_data(self):
        self.syn_image, self.syn_label = next(self.syn_iter)
        self.syn_image, self.syn_label = Variable(self.syn_image.cuda()), Variable(self.syn_label.cuda())
        self.syn_label_scales = self.scale_pyramid(self.syn_label, 5-1)
        
    def get_real_data(self):
        self.real_image, self.real_right_image, self.fb, self.real_label = next(self.real_iter)
        self.real_image, self.real_label = Variable(self.real_image.cuda()), Variable(self.real_label.cuda())
        self.real_right_image = Variable(self.real_right_image.cuda())
        self.real_label_scales = self.scale_pyramid(self.real_label, 5-1)
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
            for idx, disc_op in enumerate(netD):
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
        
        real_depth = self.netT(self.real_recon_image)
        syn_depth = self.netT(self.syn_recon_image)
        self.real_predicted_depth, self.syn_predicted_depth = real_depth[-1], syn_depth[-1]

        self.forward_netD()
        self.loss_from_disc()

        real_reconstruction_loss, syn_reconstruction_loss = self.netG_loss_fn(self.real_recon_image, self.real_image), self.netG_loss_fn(self.syn_recon_image, self.syn_image)
        real_size = len(real_depth)
        gradient_smooth_loss = self.get_smooth_weight(real_depth[1:], self.real_image_scales, real_size-1)
        
        task_loss = 0.0
        for (lab_fake_i, lab_real_i) in zip(syn_depth[1:], self.syn_label_scales):
            task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)
        
        geo_id = 0
        for (l_img, r_img, gen_depth) in zip(self.real_image_scales, self.real_right_image_scales, real_depth[1:]):
            loss, _ = self.geometric_loss_fn(l_img, r_img, gen_depth, self.fb / 2**(3-geo_id))
            task_loss += loss
            geo_id += 1
                
        self.netG_loss = self.just_adv_loss + (0.01*gradient_smooth_loss) + (100*task_loss)
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
        syn_depth = self.netT(syn_refined_image)
        real_depth = self.netT(real_refined_image)

        task_loss = 0.0
        for (lab_fake_i, lab_real_i) in zip(syn_depth[1:], self.syn_label_scales):
            task_loss += self.netT_loss_fn(lab_fake_i, lab_real_i)
        
        geo_id = 0
        for (l_img, r_img, gen_depth) in zip(self.real_image_scales, self.real_right_image_scales, real_depth[1:]):
            loss, _ = self.geometric_loss_fn(l_img, r_img, gen_depth, self.fb / 2**(3-geo_id))
            task_loss += loss
            geo_id += 1

        _, real_refined_image = self.netG(self.real_image)
        real_depth = self.netT(real_refined_image)
        real_size = len(real_depth)
        gradient_smooth_loss = self.get_smooth_weight(real_depth[1:], self.real_image_scales, real_size-1)
        
        self.netT_loss = (100*task_loss) + (0.01*gradient_smooth_loss) + adv_loss
        
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
                self.save_model()
                
        self.writer.close()
                