
import os.path as osp

from PIL import Image
from torch.utils import data
from Kitti_dataset_util import KITTI
from transform import *

class KittiDataset(data.Dataset):
    def __init__(self, root='/vulcanscratch/koutilya/kitti', data_file='train.txt', phase='train',
                 img_transform=None, joint_transform=None, depth_transform=None, complete_data=False):
      
        self.root = root
        self.data_file = data_file
        self.files = []
        self.phase = phase
        self.img_transform = img_transform
        self.joint_transform = joint_transform
        self.depth_transform = depth_transform
        self.complete_data = complete_data

        with open(osp.join(self.root, self.data_file), 'r') as f:
            data_list = f.read().split('\n')
            for data in data_list:
                if len(data) == 0:
                    continue
                
                data_info = data.split(' ')

                if not self.complete_data:
                    self.files.append({
                        "l_rgb": data_info[0],
                        "r_rgb": data_info[1],
                        "cam_intrin": data_info[2],
                        "depth": data_info[3]
                        })
                else:
                    self.files.append({
                        "l_rgb": data_info[0],
                        "r_rgb": None,
                        "cam_intrin": data_info[2],
                        "depth": data_info[3]
                        })
                    self.files.append({
                        "l_rgb": data_info[1],
                        "r_rgb": None,
                        "cam_intrin": data_info[2],
                        "depth": data_info[3]
                        })

                                    
    def __len__(self):
        return len(self.files)

    def read_data(self, datafiles):
        
        assert osp.exists(osp.join(self.root, datafiles['l_rgb'])), "Image does not exist"
        l_rgb = Image.open(osp.join(self.root, datafiles['l_rgb'])).convert('RGB')
        w = l_rgb.size[0]
        h = l_rgb.size[1]
        if not self.complete_data:
            assert osp.exists(osp.join(self.root, datafiles['r_rgb'])), "Image does not exist"
            r_rgb = Image.open(osp.join(self.root, datafiles['r_rgb'])).convert('RGB')

        kitti = KITTI()
        assert osp.exists(osp.join(self.root, datafiles['cam_intrin'])), "Camera info does not exist"
        fb = kitti.get_fb(osp.join(self.root, datafiles['cam_intrin']))
        # assert osp.exists(osp.join(self.root, datafiles['depth'])), "Depth does not exist"
        # depth = kitti.get_depth(osp.join(self.root, datafiles['cam_intrin']),
        #                         osp.join(self.root, datafiles['depth']), [h, w])

        if not self.complete_data:
            return l_rgb, r_rgb, fb
        else:
            return l_rgb, None, fb
    
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
        
        # if self.depth_transform is not None:
        #     depth = self.depth_transform(depth)

        if self.phase =='test' or self.phase=='val':
            data = {}
            data['left_img'] = l_img
            data['right_img'] = r_img
            # data['depth'] = depth
            data['fb'] = fb
            return data, datafiles['depth']

        data = {}
        if l_img is not None:
            data['left_img'] = l_img
        if r_img is not None:
            data['right_img'] = r_img
        if fb is not None:
            data['fb'] = fb

        return l_img