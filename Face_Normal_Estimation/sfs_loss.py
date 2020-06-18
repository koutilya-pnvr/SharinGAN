import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torch.nn.modules.loss import _Loss
from torch.autograd import Function, Variable

class SfSLoss(_Loss):
	def __init__(self):
		super(SfSLoss,self).__init__()

	def forward(self, image,mask,normal,albedo,light, nout, aout,lout):

		aloss=L1loss(albedo*mask,aout*mask)
		nloss=L1loss(normal*mask,nout*mask)
		lloss=L2loss(light.float(),lout)

		recloss=ReconLoss(nout,aout,lout,image,mask)

		#recloss=ReconLoss(normal,albedo,light.float(),image,mask)

		loss= 0.5*aloss + 0.5*nloss + 0.1*lloss+ 0.5*recloss

		return loss, nloss, aloss, lloss, recloss

def L1loss(a,b):
	err=torch.sum(torch.abs(a-b))
	return err/a.size(0)

def L2loss(a,b):
	return torch.sum((a-b)**2)/(2*a.size(0))

def ReconLoss(n,a,light,im,mask):
	n=normalize(n)

	att = np.pi*np.array([1, 2.0/3, 0.25])

	c1=att[0]*(1.0/np.sqrt(4*np.pi))
	c2=att[1]*(np.sqrt(3.0/(4*np.pi)))
	c3=att[2]*0.5*(np.sqrt(5.0/(4*np.pi)))
	c4=att[2]*(3.0*(np.sqrt(5.0/(12*np.pi))))
	c5=att[2]*(3.0*(np.sqrt(5.0/(48*np.pi))))

	c=torch.from_numpy(np.array([c1,c2,c3,c4,c5])).float().cuda()

	loss=0.0
	
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

		shd=Variable((torch.zeros(a.size())).cuda())

		for j in range(0,3):
			Lo=light[i,j*9:(j+1)*9]

			shd[i,j,...]=Lo[0]*H1+Lo[1]*H2+Lo[2]*H3+Lo[3]*H4+Lo[4]*H5+Lo[5]*H6+Lo[6]*H7+Lo[7]*H8+Lo[8]*H9

	
		#visualize(torch.abs(shd[i,...]*a[i,...]*mask[i,...] - im[i,...]*mask[i,...]))
		loss+=torch.sum(torch.abs(shd[i,...]*a[i,...]*mask[i,...] - im[i,...]*mask[i,...]))
		#visualize(shd[i,...]*a[i,...]*mask[i,...])

	return loss/n.size(0)

def normalize(n):
	n=2*n-1
	norm=torch.norm(n,2,1,keepdim=True)
	norm=norm.repeat(1,3,1,1)
	return (n/norm)


def visualize(rec0):
	rec0=rec0.data.numpy()
	rec0=rec0.transpose((1,2,0))
	print(rec0.shape)
	plt.imshow(rec0)
	plt.show()



class Recon_Loss(_Loss):
	def __init__(self):
		super(Recon_Loss,self).__init__()

	def forward(self, nout, aout, lout, image, mask):
		recloss=ReconLoss(nout,aout,lout,image,mask)
		return recloss