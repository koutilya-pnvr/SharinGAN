
import os.path as osp
import torch
from PIL import Image
from torch.utils import data
from Kitti_dataset_util import KITTI

class DepthToTensor(object):
    def __call__(self, arr_input):
        # tensors = [], [0, 1] -> [-1, 1]
        # arr_input = np.array(input)
        tensors = torch.from_numpy(arr_input.reshape((1, arr_input.shape[0], arr_input.shape[1]))).float()
        return tensors

class KittiDataset(data.Dataset):
    def __init__(self, root='/vulcanscratch/koutilya/kitti', data_file='train.txt', phase='train',
                 img_transform=None, joint_transform=None, depth_transform=None, depth_resize='bilinear'):
      
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.depth_transform = depth_transform
        self.depth_resize = depth_resize
        depth_path = ''
        if self.depth_resize == 'bilinear':
            depth_path = 'Bilinear_model_pseudo_labels'
        elif self.depth_resize == 'bicubic':
            depth_path = 'Bicubic_model_pseudo_labels'

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                self.files.append({
                        "l_rgb": data_info[0],
                        "r_rgb": data_info[1],
                        "cam_intrin": data_info[2],
                        "depth": osp.join('Depth_baseline_all_syn', depth_path, data_info[0])
                        })
                                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
        
        assert osp.exists(osp.join(self.root, datafiles['l_rgb'])), "Image does not exist"
        l_rgb = Image.open(osp.join(self.root, datafiles['l_rgb'])).convert('RGB')
        w = l_rgb.size[0]
        h = l_rgb.size[1]
        assert osp.exists(osp.join(self.root, datafiles['r_rgb'])), "Image does not exist"
        r_rgb = Image.open(osp.join(self.root, datafiles['r_rgb'])).convert('RGB')

        kitti = KITTI()
        assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        fb = kitti.get_fb(osp.join(self.root, datafiles['cam_intrin']))
        return l_rgb, r_rgb, fb
            
    def __getitem__(self, index):
        # if self.phase == 'train':
        #     index = random.randint(0, len(self)-1)
        # if index > len(self)-1:
        #     index = index % len(self)
        datafiles = self.files[index]
        l_img, r_img, fb = self.read_data(datafiles)

        if self.joint_transform is not None:
            if self.phase == 'train':
                l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'train', fb))
            else:
                l_img, r_img, _, fb = self.joint_transform((l_img, r_img, None, 'test', fb))
            
        if self.img_transform is not None:
            l_img = self.img_transform(l_img)
            if r_img is not None:
                r_img = self.img_transform(r_img)
        
        return l_img, r_img, fb