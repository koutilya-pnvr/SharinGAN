import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
import torchvision.models as models

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Discriminator(nn.Module):
	def __init__(self, in_channels=3):
		super(Discriminator, self).__init__()
		self.main = nn.Sequential(
			nn.Conv2d(in_channels,32,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(32,32,3,2, padding=1),
			nn.ReLU(),
			nn.Conv2d(32,64,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(64,64,3,2, padding=1),
			nn.ReLU(),
			nn.Conv2d(64,128,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(128,128,3,2, padding=1),
			nn.ReLU(),
			nn.Conv2d(128,256,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(256,256,3,2, padding=1),
			nn.ReLU(),
			nn.Conv2d(256,512,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,3,1, padding=1),
			nn.ReLU(),
			nn.Conv2d(512,512,3,2, padding=1),
			nn.ReLU(),
			Flatten(),
			nn.Linear(2048,1024),
			nn.ReLU(),
			nn.Linear(1024,512),
			nn.ReLU(),
			nn.Linear(512,1)
			)
	
	def forward(self, image):
		prob = self.main(image)
		return prob

