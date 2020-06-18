import torch
import torch.nn as nn
from torch.nn import init
import torchvision
import functools
from torch.optim import lr_scheduler
import torch.nn.functional as F
from copy import deepcopy
from torch.autograd import Variable

class RBDN_network(nn.Module):

	def __init__(self, channels=3):
		super(RBDN_network,self).__init__()
		self.num_features = 128
		self.bn0 = nn.BatchNorm2d(channels)
		self.conv1 = nn.Conv2d(channels,self.num_features,3,1,padding=1)
		self.bn1 = nn.BatchNorm2d(self.num_features)
		
		# Branch 1
		num_features_b1 = self.num_features
		self.conv_b11 = nn.Conv2d(num_features_b1,num_features_b1,3,1,padding=1)
		self.bn_b11 = nn.BatchNorm2d(num_features_b1)
		self.conv_b11_strided = nn.Conv2d(num_features_b1,num_features_b1,3,2,padding=1)
		self.bn_b11_strided = nn.BatchNorm2d(num_features_b1)

		self.conv_b12 = nn.Conv2d(2*num_features_b1, num_features_b1,3,1,padding=1)
		self.bn_b12 = nn.BatchNorm2d(num_features_b1)
		self.unpool_b1 = nn.ConvTranspose2d(num_features_b1, num_features_b1,3,2,padding=1,output_padding=1)
		self.bn_unpool_b1 = nn.BatchNorm2d(num_features_b1)
		self.conv_de_b11 = nn.Conv2d(num_features_b1, num_features_b1,3,1,padding=1)
		self.bn_de_b11 = nn.BatchNorm2d(num_features_b1)

		# Branch 2
		num_features_b2 = self.num_features
		self.conv_b21 = nn.Conv2d(num_features_b2,num_features_b2,3,1,padding=1)
		self.bn_b21 = nn.BatchNorm2d(num_features_b2)
		self.conv_b21_strided = nn.Conv2d(num_features_b2,num_features_b2,3,2,padding=1)
		self.bn_b21_strided = nn.BatchNorm2d(num_features_b2)

		self.conv_b22 = nn.Conv2d(2*num_features_b2,num_features_b2,3,1,padding=1)
		self.bn_b22 = nn.BatchNorm2d(num_features_b2)
		self.unpool_b2 = nn.ConvTranspose2d(num_features_b2, num_features_b2, 3,2,padding=1,output_padding=1)
		self.bn_unpool_b2 = nn.BatchNorm2d(num_features_b2)
		self.conv_de_b21 = nn.Conv2d(num_features_b2, num_features_b2, 3,1,padding=1)
		self.bn_de_b21 = nn.BatchNorm2d(num_features_b2)

		# # Branch 3
		num_features_b3 = self.num_features
		self.conv_b31 = nn.Conv2d(num_features_b3,num_features_b3,3,1,padding=1)
		self.bn_b31 = nn.BatchNorm2d(num_features_b3)
		self.conv_b31_strided = nn.Conv2d(num_features_b3,num_features_b3,3,2,padding=1)
		self.bn_b31_strided = nn.BatchNorm2d(num_features_b3)

		self.conv_b32 = nn.Conv2d(num_features_b3,num_features_b3,3,1,padding=1)
		self.bn_b32 = nn.BatchNorm2d(num_features_b3)
		self.unpool_b3 = nn.ConvTranspose2d(num_features_b3, num_features_b3, 3,2,padding=1,output_padding=1)
		self.bn_unpool_b3 = nn.BatchNorm2d(num_features_b3)
		self.conv_de_b31 = nn.Conv2d(num_features_b3, num_features_b3, 3,1,padding=1)
		self.bn_de_b31 = nn.BatchNorm2d(num_features_b3)


		self.conv21 = nn.Conv2d(2*self.num_features,self.num_features,3,1,padding=1)
		self.bn21 = nn.BatchNorm2d(self.num_features)
		self.conv31 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		self.bn31 = nn.BatchNorm2d(self.num_features)
		self.conv41 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		self.bn41 = nn.BatchNorm2d(self.num_features)
		self.conv51 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		self.bn51 = nn.BatchNorm2d(self.num_features)
		self.conv61 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		self.bn61 = nn.BatchNorm2d(self.num_features)
		# self.conv71 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		# self.bn71 = nn.BatchNorm2d(self.num_features)
		# self.conv81 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		# self.bn81 = nn.BatchNorm2d(self.num_features)
		# self.conv91 = nn.Conv2d(self.num_features,self.num_features,3,1,padding=1)
		# self.bn91 = nn.BatchNorm2d(self.num_features)
		
		self.conv_de = nn.Conv2d(self.num_features,channels,3,1,padding=1)

		self.relu = nn.ReLU()
		
		self.main_network = nn.Sequential(
			self.conv21, self.bn21, self.relu,
			self.conv31, self.bn31, self.relu,
			self.conv41, self.bn41, self.relu,
			self.conv51, self.bn51, self.relu,
			self.conv61, self.bn61, self.relu,
			# self.conv71, self.bn71, self.relu,
			# self.conv81, self.bn81, self.relu,
			# self.conv91, self.bn91, self.relu,
			self.conv_de
			)


	def forward(self,input):
		x = self.bn0(input)
		x_scale1 = self.relu(self.bn1(self.conv1(x)))

		x = self.relu(self.bn_b11(self.conv_b11(x_scale1)))
		x_scale2 = self.relu(self.bn_b11_strided(self.conv_b11_strided(x)))
		
		x = self.relu(self.bn_b21(self.conv_b21(x_scale2)))
		x_scale3 = self.relu(self.bn_b21_strided(self.conv_b21_strided(x)))
		
		x = self.relu(self.bn_b31(self.conv_b31(x_scale3)))
		x_scale4 = self.relu(self.bn_b31_strided(self.conv_b31_strided(x)))
		x = self.relu(self.bn_b32(self.conv_b32(x_scale4)))
		x = self.relu(self.bn_unpool_b3(self.unpool_b3(x)))
		# x = self.relu(self.unpool_b3(x))
		x_scale3_cat = self.relu(self.bn_de_b31(self.conv_de_b31(x)))
		
		x3_concatenated = torch.cat((x_scale3, x_scale3_cat),1)
		x = self.relu(self.bn_b22(self.conv_b22(x3_concatenated)))
		x = self.relu(self.bn_unpool_b2(self.unpool_b2(x)))
		# x = self.relu(self.unpool_b2(x))
		x_scale2_cat = self.relu(self.bn_de_b21(self.conv_de_b21(x)))
		
		x2_concatenated = torch.cat((x_scale2, x_scale2_cat),1)
		x = self.relu(self.bn_b12(self.conv_b12(x2_concatenated)))
		x = self.relu(self.bn_unpool_b1(self.unpool_b1(x)))
		# x = self.relu(self.unpool_b1(x))
		x_scale1_cat = self.relu(self.bn_de_b11(self.conv_de_b11(x)))
		
		x1_concatenated = torch.cat((x_scale1, x_scale1_cat),1)
		output = self.main_network(x1_concatenated)
		return x1_concatenated, output
