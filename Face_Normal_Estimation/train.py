from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as tr
from torch.autograd import Variable

import os
import glob
import sys
import time
import argparse
import datetime
import subprocess
import gc
from tqdm import tqdm

from data_loader import SfSDataset, ToTensor
from sfsnet import SfSNet as primary_network, conv_init
import sfsnet
from sfs_loss import SfSLoss, Recon_Loss
from disc import Discriminator
from RBDN_original import RBDN_network as autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--resume', type='store_true', required=True, help='Indicate if you want to resume the training from the latest saved model. If not, the end-to-end training will start from a pretrained generator and primary network on synthetic data.')
opt = parser.parse_args()

# DATA LOADING

print('\n[Phase 1] : Data Preparation')
Batch_size = 16

transform_train = tr.Compose([ToTensor()])
syndata = SfSDataset(csv_file='./Syn_train.csv', transform=transform_train) 
syn_dataloader = torch.utils.data.DataLoader(syndata, batch_size=Batch_size, shuffle=True, num_workers=4)

realdata = SfSDataset(csv_file='./CelebA_train.csv', transform=transform_train)
real_dataloader = torch.utils.data.DataLoader(realdata, batch_size=Batch_size, shuffle=True, num_workers=4)

real_valdata = SfSDataset(csv_file='./CelebA_test.csv', transform=transform_train)
real_val_dataloader = torch.utils.data.DataLoader(real_valdata, batch_size=Batch_size, shuffle=True, num_workers=4)

syn_iter = iter(syn_dataloader)
real_iter = iter(real_dataloader)
real_val_iter = iter(real_dataloader)
syn_per_epoch = len(syn_iter)
real_per_epoch = len(real_iter)
real_val_per_epoch = len(real_val_iter)
        
autoenc = autoencoder()
primary_net = primary_network()
disc = Discriminator()
primary_net.apply(conv_init)

autoenc.cuda()
primary_net.cuda()
disc.cuda()
print('Networks Initialized')

#Loss function
criterion = nn.MSELoss()
disc_criterion = nn.BCELoss()
primary_criterion = SfSLoss()
primary_criterion.cuda()
criterion.cuda()
disc_criterion.cuda()
print('Losses Initialized')

autoenc_optimizer = optim.Adam(autoenc.parameters(), lr=1e-4)
primary_net_optimizer = optim.Adam(primary_net.parameters(), lr=1e-4)
disc_optimizer = optim.Adam(disc.parameters(), lr=1e-4)

print('Starting Training')

def reset_grad(exclude=None):
    if(exclude==None):
        autoenc_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        primary_net_optimizer.zero_grad()
    elif(exclude=='autoenc'):
        disc_optimizer.zero_grad()
        primary_net_optimizer.zero_grad()
    elif(exclude=='disc'):
        autoenc_optimizer.zero_grad()
        primary_net_optimizer.zero_grad()
    elif(exclude=='primary_net'):
        autoenc_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        
if (not opt.resume):
    rbdn_saved_model = torch.load('pretrained_models/preg_RBDN-19999.pth.tar')
    autoenc.load_state_dict(rbdn_saved_model['autoenc_state_dict'])
    autoenc_optimizer.load_state_dict(rbdn_saved_model['autoenc_optimizer'])

    primary_net_checkpoint = torch.load('pretrained_models/net_epoch_r5_5.pth')
    primary_net.load_state_dict(primary_net_checkpoint)


START_ITER = 0
saved_models = glob.glob(os.path.join('saved_models/Face_Normal_Estimator_da-*.pth.tar' ))
if len(saved_models)>0 and opt.resume:
    saved_iters = [int(s.split('-')[1].split('.')[0]) for s in saved_models]
    recent_id = saved_iters.index(max(saved_iters))
    saved_model = saved_models[recent_id]
    model_state = torch.load(saved_model)
    autoenc.load_state_dict(model_state['autoenc_state_dict'])
    disc.load_state_dict(model_state['disc_state_dict'])
    primary_net.load_state_dict(model_state['primary_net_state_dict'])
    
    autoenc_optimizer.load_state_dict(model_state['autoenc_optimizer'])
    disc_optimizer.load_state_dict(model_state['disc_optimizer'])
    primary_net_optimizer.load_state_dict(model_state['primary_net_optimizer'])
    START_ITER = model_state['iteration']+1

def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    batch_size =min(h_s.size(0), h_t.size(0))
    h_s = h_s[:batch_size,:,:,:]
    h_t = h_t[:batch_size,:,:,:]
    alpha = torch.rand(batch_size, 1, 1, 1)
    alpha = alpha.expand_as(h_s)
    alpha = alpha.cuda()
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    # interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()
    interpolates = Variable(interpolates.cuda(), requires_grad=True)
    preds = critic(interpolates)
    gradients = torch.autograd.grad(preds, interpolates,
                        grad_outputs=torch.ones_like(preds).cuda(),
                        retain_graph=True, create_graph=True)[0]
    gradients = gradients.view(batch_size,-1) 
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty

# set_iter(START_ITER)
BATCH_NUM = 0

ITERATIONS = 1000000
k_critic = 5
gamma = 10

for i in range(START_ITER, ITERATIONS):
    if (BATCH_NUM) % (syn_per_epoch) == 0:
        syn_iter = iter(syn_dataloader)
    if (BATCH_NUM) % (real_per_epoch) == 0:
        real_iter = iter(real_dataloader)     

    syn_data = syn_iter.next()
    real_data = real_iter.next()
    BATCH_NUM += 1

    syn_image, syn_mask, syn_normal, syn_albedo, syn_light = syn_data['image'], syn_data['mask'], syn_data['normal'], syn_data['albedo'], syn_data['light']
    real_image, real_mask, real_normal, real_albedo, real_light = real_data['image'], real_data['mask'], real_data['normal'], real_data['albedo'], real_data['light']

    syn_image, syn_mask, syn_normal, syn_albedo, syn_light = Variable(syn_image.cuda()), Variable(syn_mask.cuda()), Variable(syn_normal.cuda()), Variable(syn_albedo.cuda()), Variable(syn_light.cuda())
    real_image, real_mask, real_normal, real_albedo, real_light = Variable(real_image.cuda()), Variable(real_mask.cuda()), Variable(real_normal.cuda()), Variable(real_albedo.cuda()), Variable(real_light.cuda())
   
    with torch.no_grad():
        syn_features, syn_recon_image  = autoenc(syn_image)
        real_features, real_recon_image = autoenc(real_image)

    for _ in range(k_critic):
    #########################################################
    ######## Train Discriminator
    #########################################################

        gp = gradient_penalty(disc, syn_recon_image, real_recon_image)
        D_real = disc(real_recon_image)
        D_syn = disc(syn_recon_image)
        D_real, D_syn = D_real.mean(), D_syn.mean()
        Wasserstein_distance = D_syn - D_real
        Total_disc_loss = -Wasserstein_distance + (gamma * gp)
        
        reset_grad()
        (Total_disc_loss).backward()
        reset_grad(exclude='disc')
        disc_optimizer.step()

    for k in range(3):
#########################################################
######## Train AutoEncoder
#########################################################
        if (BATCH_NUM) % (syn_per_epoch) == 0:
            syn_iter = iter(syn_dataloader)
        if (BATCH_NUM) % (real_per_epoch) == 0:
            real_iter = iter(real_dataloader)     

        syn_data = syn_iter.next()
        real_data = real_iter.next()
        BATCH_NUM += 1

        syn_image, syn_mask, syn_normal, syn_albedo, syn_light = syn_data['image'], syn_data['mask'], syn_data['normal'], syn_data['albedo'], syn_data['light']
        real_image, real_mask, real_normal, real_albedo, real_light = real_data['image'], real_data['mask'], real_data['normal'], real_data['albedo'], real_data['light']

        syn_image, syn_mask, syn_normal, syn_albedo, syn_light = Variable(syn_image.cuda()), Variable(syn_mask.cuda()), Variable(syn_normal.cuda()), Variable(syn_albedo.cuda()), Variable(syn_light.cuda())
        real_image, real_mask, real_normal, real_albedo, real_light = Variable(real_image.cuda()), Variable(real_mask.cuda()), Variable(real_normal.cuda()), Variable(real_albedo.cuda()), Variable(real_light.cuda())
       
        syn_features, syn_recon_image = autoenc(syn_image)
        real_features, real_recon_image =  autoenc(real_image)

        real_normal_out, real_albedo_out, real_light_out = primary_net(real_recon_image)
        syn_normal_out, syn_albedo_out, syn_light_out = primary_net(syn_recon_image)
        
        D_real, D_syn = disc(real_recon_image), disc(syn_recon_image)
        D_real, D_syn = D_real.mean(), D_syn.mean()
        Wasserstein_distance = D_syn - D_real
        Total_gen_loss = Wasserstein_distance

        real_reconstruction_loss, syn_reconstruction_loss = criterion(real_recon_image, real_image), criterion(syn_recon_image, syn_image)
        syn_sfs_loss, syn_normal_loss, syn_albedo_loss, syn_light_loss, syn_recon_sfs_loss = primary_criterion(syn_image, syn_mask, syn_normal, syn_albedo, syn_light, syn_normal_out, syn_albedo_out, syn_light_out)
        real_sfs_loss, real_normal_loss, real_albedo_loss, real_light_loss, real_recon_sfs_loss = primary_criterion(real_image, real_mask, real_normal, real_albedo, real_light, real_normal_out, real_albedo_out, real_light_out)
        
        total_sfs_loss = real_sfs_loss + syn_sfs_loss

        total_ae_loss = 10*(real_reconstruction_loss + syn_reconstruction_loss) + 0.1*(total_sfs_loss) + Total_gen_loss
        
        reset_grad()
        total_ae_loss.backward()
        reset_grad(exclude='autoenc')
        autoenc_optimizer.step()

#########################################################
######## Train Primary network
#########################################################
    if (BATCH_NUM) % (syn_per_epoch) == 0:
        syn_iter = iter(syn_dataloader)
    if (BATCH_NUM) % (real_per_epoch) == 0:
        real_iter = iter(real_dataloader)     

    syn_data = syn_iter.next()
    real_data = real_iter.next()
    BATCH_NUM += 1

    syn_image, syn_mask, syn_normal, syn_albedo, syn_light = syn_data['image'], syn_data['mask'], syn_data['normal'], syn_data['albedo'], syn_data['light']
    real_image, real_mask, real_normal, real_albedo, real_light = real_data['image'], real_data['mask'], real_data['normal'], real_data['albedo'], real_data['light']
    syn_image, syn_mask, syn_normal, syn_albedo, syn_light = Variable(syn_image.cuda()), Variable(syn_mask.cuda()), Variable(syn_normal.cuda()), Variable(syn_albedo.cuda()), Variable(syn_light.cuda())
    real_image, real_mask, real_normal, real_albedo, real_light = Variable(real_image.cuda()), Variable(real_mask.cuda()), Variable(real_normal.cuda()), Variable(real_albedo.cuda()), Variable(real_light.cuda())
   
    syn_features, syn_recon_image = autoenc(syn_image)
    real_features, real_recon_image =  autoenc(real_image)

    real_normal_out, real_albedo_out, real_light_out = primary_net(real_recon_image)
    syn_normal_out, syn_albedo_out, syn_light_out = primary_net(syn_recon_image)
    
    syn_sfs_loss, syn_normal_loss, syn_albedo_loss, syn_light_loss, syn_recon_sfs_loss = primary_criterion(syn_image, syn_mask, syn_normal, syn_albedo, syn_light, syn_normal_out, syn_albedo_out, syn_light_out)
    real_sfs_loss, real_normal_loss, real_albedo_loss, real_light_loss, real_recon_sfs_loss = primary_criterion(real_image, real_mask, real_normal, real_albedo, real_light, real_normal_out, real_albedo_out, real_light_out)
    
    total_sfs_loss = real_sfs_loss + syn_sfs_loss

    reset_grad()
    total_sfs_loss.backward()
    reset_grad(exclude='primary_net')
    primary_net_optimizer.step()
    
    if(i%1000==999):
        p_path = os.path.join('saved_models/Face_Normal_Estimator_da-%d.pth.tar' %(i))
                    
        torch.save({
                'iteration': i,
                'autoenc_state_dict': autoenc.state_dict(),
                'primary_net_state_dict': primary_net.state_dict(),
                'disc_state_dict': disc.state_dict(),
                'autoenc_optimizer': autoenc_optimizer.state_dict(),
                'disc_optimizer': disc_optimizer.state_dict(),
                'primary_net_optimizer': primary_net_optimizer.state_dict(),
                }, p_path)   
        
