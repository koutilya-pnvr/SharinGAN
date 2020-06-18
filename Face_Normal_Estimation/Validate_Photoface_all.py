from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as tr
from torch.autograd import Variable

import os
import glob
import sys
import time
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import cv2

from Photoface_dataloader import Photoface
from sfsnet import SfSNet
from sfs_loss import SfSLoss
from tqdm import tqdm
from RBDN_original import RBDN_network as autoencoder
from tensorboardX import SummaryWriter

def Recon_loss(im,n,a,light,mask, iteration, normal_gt, recon_image):
	n=normalize(n)
	# normal_gt=normalize(normal_gt)

	att = np.pi*np.array([1, 2.0/3, 0.25])

	c1=att[0]*(1.0/np.sqrt(4*np.pi))
	c2=att[1]*(np.sqrt(3.0/(4*np.pi)))
	c3=att[2]*0.5*(np.sqrt(5.0/(4*np.pi)))
	c4=att[2]*(3.0*(np.sqrt(5.0/(12*np.pi))))
	c5=att[2]*(3.0*(np.sqrt(5.0/(48*np.pi))))

	c=torch.from_numpy(np.array([c1,c2,c3,c4,c5])).float().cuda()
	shd=Variable((torch.zeros(a.size())).cuda())

	L1_err=0
	L2_err=0
	for i in range(0,n.size(0)):
		nx=n[i,0,...]
		ny=n[i,1,...]
		nz=n[i,2,...]

		
		H1=c[0]*Variable((torch.ones((n.size(2),n.size(3)))).cuda())
		H2=c[1]*nz
		H3=c[1]*nx
		H4=c[1]*ny
		H5=c[2]*(2*nz*nz - nx*nx -ny*ny)
		H6=c[3]*nx*nz
		H7=c[3]*ny*nz
		H8=c[4]*(nx*nx - ny*ny)
		H9=c[3]*nx*ny

		for j in range(0,3):
			Lo=light[i,j*9:(j+1)*9]

			shd[i,j,...]=Lo[0]*H1+Lo[1]*H2+Lo[2]*H3+Lo[3]*H4+Lo[4]*H5+Lo[5]*H6+Lo[6]*H7+Lo[7]*H8+Lo[8]*H9

		

		err=255*(torch.sum((shd[i,...]*a[i,...]*mask[i,...] - im[i,...]*mask[i,...])**2,dim=0)**0.5)
		nele=torch.sum(mask[i,0,...])

		L1_err+=torch.sum(err)/nele
		L2_err+=(torch.norm(err)**2/nele)**0.5

	# if(opt.test):
	# 	save_images_tensorboard(im,mask,light,n,a,shd, iteration, normal_gt, recon_image)

	return L1_err/(i+1), L2_err/(i+1)


def save_images_tensorboard(im,mask,l,n,a,shd, iteration, normal_gt, recon_image):
	rec=(shd*a)*mask+(1-mask)*im
	im, mask, a, shd, rec = postprocess(im), postprocess(mask), postprocess(a), postprocess(shd), postprocess(rec)
	# n = postprocess(n)
	# normal_gt = postprocess(normal_gt)
	n=(1+n)/2
	# normal_gt=(1+normal_gt)/2
	writer.add_image('1 Image',torchvision.utils.make_grid(im,nrow=10), iteration)
	writer.add_image('2 AE Reconstructed Image',torchvision.utils.make_grid(recon_image,nrow=10), iteration)
	writer.add_image('3 Normal',torchvision.utils.make_grid(n*mask,nrow=10), iteration)
	writer.add_image('4 GT Normal',torchvision.utils.make_grid(normal_gt*mask,nrow=10), iteration)
	writer.add_image('6 Albedo',torchvision.utils.make_grid(a,nrow=10), iteration)
	writer.add_image('7 Shading',torchvision.utils.make_grid(shd*mask,nrow=10), iteration)
	writer.add_image('5 Reconstruction',torchvision.utils.make_grid(rec,nrow=10), iteration)
	

def postprocess(rec0):
	rec0 = torch.clamp(rec0,0,1)
	return rec0

def normalize_to_im(rec0): #Takes a GPU variable and converts it back to image format
	rec0=((rec0.data).cpu()).numpy()
	rec0=rec0.transpose((1,2,0))
	rec0[rec0>1]=1
	rec0[rec0<0]=0
	return rec0

def normalize(n):
	n=2*n-1
	norm=torch.norm(n,2,1,keepdim=True)
	norm=norm.repeat(1,3,1,1)
	return (n/norm)

def get_loss(preds, truths):
	# Calculate loss : average cosine value between predicted/actual normals at each pixel
	# theta = arccos((P dot Q) / (|P|*|Q|)) -> cos(theta) = (P dot Q) / (|P|*|Q|)
	# Both the predicted and ground truth normals normalized to be between -1 and 1
	preds_norm = normalize(preds)
	truths_norm = normalize(truths)
	# make negative so function decreases (cos -> 1 if angles same)
	loss = -torch.sum(preds_norm * truths_norm, dim = 1)
	angular_error = torch.acos(torch.clamp(-loss,-1,1))
	return loss, angular_error*180.0/np.pi

#Pass data through network
def validate(autoenc, net, writer, epoch):
	L1_err=0.0
	L2_err=0.0
	normal_loss=0.0
	angular_loss=0.0
	L20 = 0.0
	L25 = 0.0
	L30 = 0.0
	total_valid_pixels = 0.0
	
	with torch.no_grad():
		t=tqdm(test_loader, desc='Validation Iteration: %d'%epoch)
		for i,(image, normal, mask) in enumerate(t):
			image=Variable(image.cuda())
			mask=Variable(mask.cuda())
			normal=Variable(normal.cuda())
			_,recon_image = autoenc(image)
			nout, aout, lout = net(recon_image)
			l1err, l2err = Recon_loss(image, nout, aout, lout, mask, epoch, normal, recon_image)

			L1_err+=l1err.item()
			L2_err+=l2err.item()

			zero_tensor = Variable(torch.zeros((image.size(0), 3, 128, 128)).cuda().float())
			one_tensor = Variable(torch.ones((image.size(0), 3, 128, 128)).cuda().float())
			
			mask = torch.where(mask<0.9, zero_tensor, one_tensor)
			normal_error, angular_error = get_loss(nout*mask, normal*mask)
			normal_error*=mask[:,0,:,:]
			angular_error*=mask[:,0,:,:]
			normal_loss += torch.sum(normal_error)/torch.sum(mask[:,0,:,:])
			angular_loss += torch.sum(angular_error)/torch.sum(mask[:,0,:,:])
			# L20 += torch.sum((angular_error<20)*(angular_error>0)).item()
			# L25 += torch.sum((angular_error<25)*(angular_error>0)).item()
			# L30 += torch.sum((angular_error<30)*(angular_error>0)).item()
			L20 += torch.sum((angular_error<20)*(mask[:,0,:,:].byte())).item()
			L25 += torch.sum((angular_error<25)*(mask[:,0,:,:].byte())).item()
			L30 += torch.sum((angular_error<30)*(mask[:,0,:,:].byte())).item()
			total_valid_pixels += torch.sum(mask[:,0,:,:]).item()
			t.set_postfix(L1_Loss=L1_err/(i+1), L2_Loss=L2_err/(i+1), Normal_Loss=normal_loss.item()/(i+1), Angular_Loss=angular_loss.item()/(i+1), L20_Acc=100*L20/total_valid_pixels, L25_Acc=100*L25/total_valid_pixels, L30_Acc=100*L30/total_valid_pixels)

	l1loss = L1_err/(i+1)
	l2loss = L2_err/(i+1)
	mean_normal_loss = normal_loss.item()/(i+1)
	mean_angular_loss = angular_loss.item()/(i+1)

	# writer.add_scalar('L1 Loss',l1loss,epoch)
	# writer.add_scalar('L2 Loss',l2loss,epoch)
	# writer.add_scalar('Mean Normal Loss',mean_normal_loss,epoch)
	# writer.add_scalar('Mean Angular Loss',mean_angular_loss,epoch)
	# writer.add_scalar('Acc <20',100*L20/total_valid_pixels,epoch)
	# writer.add_scalar('Acc <25',100*L25/total_valid_pixels,epoch)
	# writer.add_scalar('Acc <30',100*L30/total_valid_pixels,epoch)
	return l1loss, l2loss, mean_normal_loss, mean_angular_loss, 100*L20/total_valid_pixels, 100*L25/total_valid_pixels, 100*L30/total_valid_pixels

global opt 
parser = argparse.ArgumentParser()
parser.add_argument('--a', dest='all_iters', action='store_true', help='Indicate whether to validate on all saved models')
parser.add_argument('--test', dest='test', action='store_true', help='Evaluation on Test set (val set by default)')
parser.add_argument('--iteration', dest='iteration', type=int, required=True, help='Iteration of the saved model for validation/testing')
opt = parser.parse_args()

root_dir = '.'
print('\n[Phase 1] : Data Preparation')

saved_models = glob.glob(os.path.join(root_dir,'saved_models','Face_Normal_Estimator_da-*.pth.tar'))
saved_iters = [int(s.split('-')[1].split('.')[0]) for s in saved_models]
saved_iters.sort()
	
if(opt.test):
	testdata =  Photoface(train=False)
	writer_file = './tensorboard_logs/Face_Normal_Estimation/test/'+'iteration='+str(opt.iteration)
	saved_iters = [opt.iteration]
else:
	testdata =  Photoface(train=True, val=True)
	writer_file = '/vulcan/scratch/koutilya/projects/Domain_Adaptation/SFS_net/tensorboard_logs/Face_Normal_Estimator/val/
	if(not opt.all_iters):
		saved_iters = [saved_iters[-1]]


test_loader = torch.utils.data.DataLoader(testdata, batch_size=20, shuffle=False, num_workers=1)

#Load the model
net=SfSNet()
autoenc = autoencoder()
print('Network Initialized')

#Use GPU
net.cuda()
autoenc.cuda()
cudnn.benchmark = True

writer = SummaryWriter(writer_file)
for i in saved_iters:
	model = torch.load(os.path.join(root_dir, 'saved_models/Face_Normal_Estimator_da-'+str(i)+'.pth.tar'))
	net.load_state_dict(model['primary_net_state_dict'])
	autoenc.load_state_dict(model['autoenc_state_dict'])
	net.eval()
	autoenc.eval()
	validate(autoenc, net, writer, i)
writer.close()
