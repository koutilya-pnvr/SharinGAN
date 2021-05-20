import os
import glob
import numpy as np
import random
from tqdm import tqdm

import torch
from torch import nn
import torch.optim as Optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms as tr
from tensorboardX import SummaryWriter

from networks import all_networks

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset
from Dataloaders.Kitti_dataloader import KittiDataset as real_dataset
from Dataloaders.transform import *

class Solver():
    def __init__(self):
        self.root_dir = '.'
        
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
        
        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        
        self.netG_loss_fn = self.netG_loss_fn.cuda()
        
        # Initialize Optimizers
        self.netG_optimizer = Optim.Adam(self.netG.parameters(), lr=5e-5, betas=(0.5,0.9))
        
        # Training Configuration details
        self.batch_size = 6
        self.iteration = None
        self.total_iterations = 200000
        self.START_ITER = 0
        self.kr = 1
        self.kd = 1 
        self.writer = SummaryWriter(os.path.join(self.root_dir,'../tensorboard_logs/Vkitti-kitti/AE_Baseline/Resnet_NEW'))

        
        # Flags
        self.tb_flag = 0
        self.best_acc = 0

        # Transforms
        joint_transform_list = [RandomImgAugment(no_flip=False, no_rotation=False, no_augment=False, size=(192,640))]
        img_transform_list = [tr.ToTensor(), tr.Normalize([.5, .5, .5], [.5, .5, .5])]
        self.joint_transform = tr.Compose(joint_transform_list)
        self.img_transform = tr.Compose(img_transform_list)
        self.depth_transform = tr.Compose([DepthToTensor()])

        # Initialize Data
        self.syn_image, self.syn_label, self.real_image = None, None, None
        self.get_training_data()
        self.get_training_dataloaders()
        # self.get_validation_data()


    # def get_validation_data(self):
    #     val_dataset = real_dataset(data_file='test.txt',phase='test',img_transform=self.img_transform, joint_transform=self.joint_transform, depth_transform=self.depth_transform)
    #     self.val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
    #     syn_val_dataset = syn_dataset(train=False)
    #     self.syn_val_dataloader = DataLoader(syn_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        
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

    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, 'saved_models_new', 'AE_Resnet_Baseline.pth.tar' ))
        if len(saved_models)>0:
            model_state = torch.load(saved_models[0])
            self.netG.load_state_dict(model_state['netG_state_dict'])
            
            self.netG_optimizer.load_state_dict(model_state['netG_optimizer'])
            self.START_ITER = model_state['iteration']+1
            return True
        return False

    def save_model(self):
        if not os.path.exists(os.path.join(self.root_dir, 'saved_models_new')):
            os.mkdir(os.path.join(self.root_dir, 'saved_models_new'))
        
        torch.save({
                'iteration': self.iteration,
                'netG_state_dict': self.netG.state_dict(),
                'netG_optimizer': self.netG_optimizer.state_dict(),
                }, os.path.join(self.root_dir,'saved_models_new', 'AE_Resnet_Baseline.pth.tar'))
        
    def get_syn_data(self):
        self.syn_image, self.syn_label = next(self.syn_iter)
        self.syn_image, self.syn_label = Variable(self.syn_image.cuda()), Variable(self.syn_label.cuda())
        
    def get_real_data(self):
        self.real_image = next(self.real_iter)
        self.real_image = Variable(self.real_image.cuda())

    def update_netG(self):
        real_features, self.real_recon_image = self.netG(self.real_image)
        syn_features, self.syn_recon_image = self.netG(self.syn_image)
        real_reconstruction_loss, syn_reconstruction_loss = self.netG_loss_fn(self.real_recon_image, self.real_image), self.netG_loss_fn(self.syn_recon_image, self.syn_image)
        self.netG_loss = real_reconstruction_loss + syn_reconstruction_loss
        
        self.netG_optimizer.zero_grad()
        self.netG_loss.backward()
        self.netG_optimizer.step()

    def train(self):
        
        self.load_prev_model()
        for self.iteration in tqdm(range(self.START_ITER, self.total_iterations)): 
            
            self.get_syn_data()
            self.get_real_data()
            
            self.update_netG()

            ###################################################
            #### Tensorboard Logging
            ###################################################            
            self.writer.add_scalar('Total Generator loss', self.netG_loss, self.iteration)
            
            if self.iteration % 1000 == 999:
                # Validation and saving models
                self.save_model()
                
        self.writer.close()