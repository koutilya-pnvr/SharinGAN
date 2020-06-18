from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tr, utils
import glob
import cv2
import lycon

class SfSDataset(Dataset):
	def __init__(self,csv_file,transform=None):
		self.frames = pd.read_csv(csv_file, header=None)
		self.transform = transform

	def __len__(self):
		return len(self.frames)

	def __getitem__(self,idx):
		image = lycon.load(self.frames.iloc[idx, 0].replace('/vulcan/scratch/koutilya', '.'))
		mask = lycon.load(self.frames.iloc[idx, 1].replace('/vulcan/scratch/koutilya', '.'))
		normal = lycon.load(self.frames.iloc[idx, 2].replace('/vulcan/scratch/koutilya', '.'))
		albedo = lycon.load(self.frames.iloc[idx, 3].replace('/vulcan/scratch/koutilya', '.'))

		light = self.frames.iloc[idx, 4:].values #a 1x27 array


		sample = {'image': image, 'mask': mask, 'normal': normal, 'albedo': albedo, 'light': light}

		if self.transform:
			sample = self.transform(sample)
		return sample

class ToTensor(object):
	def __call__(self, sample):
		image, mask, normal, albedo, light = sample['image'], sample['mask'], sample['normal'], sample['albedo'], sample['light']
		light=light.astype(float)

		return {'image': to_tensor(image),
                'mask': to_tensor(mask),
                'normal': to_tensor(normal),
                'albedo': to_tensor(albedo),
                'light': torch.from_numpy(light)}



def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.
    See ``ToTensor`` for more details.
    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
    Returns:
        Tensor: Converted image.
    """
    #if not(_is_pil_image(pic) or _is_numpy_image(pic)):
    #    raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img