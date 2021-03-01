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
import torchvision
from torchvision import transforms as tr
import torch.nn.functional as F
from tensorboardX import SummaryWriter

from networks import all_networks

from Dataloaders.VKitti_dataloader import VKitti as syn_dataset

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
        self.netT = all_networks.define_G(3, 1, 64, 4, 'batch',
                                            'PReLU', 'UNet', 'kaiming', 0,
                                            False, [0], 0.1)
        self.netT.cuda()

        # Initialize Loss
        self.netT_loss_fn = nn.L1Loss()

        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Initialize Optimizers
        self.netT_optimizer = Optim.Adam(self.netT.parameters(), lr=1e-4, betas=(0.95,0.999))

        # Training Configuration details
        self.batch_size = 16
        self.iteration = None
        self.total_iterations = 200000
        self.START_ITER = 0
        self.kr = 1
        self.kd = 1 
        self.writer = SummaryWriter(os.path.join(self.root_dir,'../tensorboard_logs/Vkitti-kitti/PTNet_Baseline_bicubic'))

        # Initialize Data
        self.syn_image, self.syn_label = None, None
        self.get_training_data()
        self.get_training_dataloaders()
        self.get_validation_data()

    def get_validation_data(self):
        syn_val_dataset = syn_dataset(train=False)
        self.syn_val_dataloader = DataLoader(syn_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        
    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

    def get_training_data(self):
        self.syn_loader = DataLoader(syn_dataset(train=True, resize_mode='bicubic'), batch_size=self.batch_size, shuffle=True, num_workers=4)
        
    def get_training_dataloaders(self):
        self.syn_iter = self.loop_iter(self.syn_loader)
        
    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, 'saved_models', 'PTNet_baseline-*_bicubic.pth.tar' ))
        if len(saved_models)>0:
            model_state = torch.load(saved_models[0])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            self.netT_optimizer.load_state_dict(model_state['netT_optimizer'])
            self.START_ITER = model_state['iteration']+1
            return True
        return False

    def save_model(self):
        if not os.path.exists(os.path.join(self.root_dir, 'saved_models')):
            os.mkdir(os.path.join(self.root_dir, 'saved_models'))
        
        torch.save({
                'iteration': self.iteration,
                'netT_state_dict': self.netT.state_dict(),
                'netT_optimizer': self.netT_optimizer.state_dict(),
                }, os.path.join(self.root_dir,'saved_models', 'PTNet_baseline_tmp_bicubic.pth.tar'))
        os.system('mv '+os.path.join(self.root_dir,'saved_models', 'PTNet_baseline_tmp_bicubic.pth.tar')+' '+os.path.join(self.root_dir,'saved_models', 'PTNet_baseline-'+str(self.iteration)+'_bicubic.pth.tar'))
        
    def get_syn_data(self):
        self.syn_image, self.syn_label = next(self.syn_iter)
        self.syn_image, self.syn_label = Variable(self.syn_image.cuda()), Variable(self.syn_label.cuda())
        
    def update_netT(self):

        depth = self.netT(self.syn_image)
        self.netT_loss = self.netT_loss_fn(depth[-1], self.syn_label) 
        
        self.netT_optimizer.zero_grad()
        self.netT_loss.backward()
        self.netT_optimizer.step()


    def train(self):
        self.load_prev_model()
        for self.iteration in tqdm(range(self.START_ITER, self.total_iterations)): 
            
            self.get_syn_data()
            self.update_netT()

            ###################################################
            #### Tensorboard Logging
            ###################################################            
            self.writer.add_scalar('Depth Estimation', self.netT_loss, self.iteration)

            if self.iteration % 1000 == 999:
                # Validation and saving models
                self.save_model()
                self.Validation()
                
        self.writer.close()

    def Validation(self):
        self.netT.eval()
        
        with torch.no_grad():
            l1_loss = 0.0
            for i,(syn_image, syn_label) in tqdm(enumerate(self.syn_val_dataloader)):
                syn_image, syn_label = Variable(syn_image.cuda()), Variable(syn_label.cuda())
                syn_depth = self.netT(syn_image)
                loss = self.netT_loss_fn(syn_depth[-1], syn_label)
                l1_loss += loss

        self.writer.add_scalar('Validation/L1loss', l1_loss / (1+i) , self.iteration)

        self.netT.train()
        