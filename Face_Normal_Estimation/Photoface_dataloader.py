import torch
from torchvision import transforms as tr 
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import glob
import lycon

class Photoface(Dataset):
	def __init__(self, train=True, val=False, transform=tr.Compose([tr.ToTensor()])):
		super(Photoface, self).__init__()
		self.train = train
		self.val = val
		self.transform = transform
		if(self.train):
			self.data_file = 'photoface-data/Photoface_train_list.txt'
		else:
			self.data_file = 'photoface-data/Photoface_test_list.txt'

		with open(self.data_file,'r') as file:
			self.data_list = file.read()
			self.data_list = self.data_list.split('\n')[:-1]

		self.data_list = [f.replace('scratch2','.') for f in self.data_list]
		self.data_list = [f.split(' ')[0] for f in self.data_list]

		if(self.train and self.val):
			self.data_list_final = self.data_list[int(0.85*len(self.data_list)):]
		elif(self.train and not self.val):
			self.data_list_final = self.data_list[:int(0.85*len(self.data_list))]

		else:
			self.data_list_final = self.data_list

		self.normals = [f.split('_align.png')[0]+'_normal.png' for f in self.data_list_final]
		self.masks = [f.split('_align.png')[0]+'_mask.png' for f in self.data_list_final]

	def __len__(self):
		return len(self.data_list_final)

	def __getitem__(self,idx):
		image, normal, mask = lycon.load(self.data_list_final[idx]), lycon.load(self.normals[idx]), lycon.load(self.masks[idx])
		normal = np.ascontiguousarray(normal)
		if(self.transform):
			input, normal, mask = self.transform(image), self.transform(normal), self.transform(mask)
		return input, normal, mask