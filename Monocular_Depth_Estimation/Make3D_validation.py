import os
import os.path as osp
import glob
import numpy as np
import random
from tqdm import tqdm
import argparse
from PIL import Image
import cv2
import scipy.io

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from networks import all_networks

from Dataloaders.Make3d_dataloader import Make3D_dataset as real_dataset


class Solver():
    def __init__(self, opt):
        self.root_dir = '.'
        self.opt = opt

        # Seed
        self.seed = 1729
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        # Initialize networks
        self.netG = all_networks.define_G(3, 3, 64, 9, 'batch',
                                                'PReLU', 'ResNet', 'kaiming', 0,
                                                False, [0])

        self.netG.cuda()
        
        self.netT = all_networks.define_G(3, 1, 64, 4, 'batch',
                                            'PReLU', 'UNet', 'kaiming', 0,
                                            False, [0], 0.1)
        self.netG.cuda()
        self.netT.cuda()
        
        # Initialize Loss
        self.netG_loss_fn = nn.MSELoss()
        self.netT_loss_fn = nn.L1Loss()
        self.netG_loss_fn = self.netG_loss_fn.cuda()
        self.netT_loss_fn = self.netT_loss_fn.cuda()

        # Training Configuration details
        self.batch_size = 16
        self.iteration = None
        
        self.saved_models_dir = 'saved_models'
        self.output_images_dir = os.path.join(self.root_dir, 'make3d_results')
        if not os.path.exists(self.output_images_dir):
            os.mkdir(self.output_images_dir)

        # Initialize Data
        self.get_validation_data()

        self.garg_crop = True
        self.eigen_crop = False
        self.flag = True

    def loop_iter(self, loader):
        while True:
            for data in iter(loader):
                yield data

    def compute_errors_make3d(self, gt, pred):
        rmse = (gt - pred) ** 2
        rmse = np.sqrt(rmse.mean())

        rmse_log = (np.log10(gt) - np.log10(pred)) ** 2
        rmse_log = np.sqrt(rmse_log.mean())

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        sq_rel = np.mean(((gt - pred)**2) / gt)

        return abs_rel, sq_rel, rmse, rmse_log

    def get_validation_data(self):
        self.real_val_dataset = real_dataset()
        self.real_val_dataloader = DataLoader(self.real_val_dataset, shuffle=False, batch_size=self.batch_size, num_workers=4)
        
    def load_prev_model(self):
        saved_models = glob.glob(os.path.join(self.root_dir, self.saved_models_dir, 'Depth_Estimator_'+self.opt.disc+'_'+self.opt.supervision+'_'+self.opt.depth_resize+'_da-'+str(self.iteration)+'.pth.tar' ))
        if len(saved_models)>0:
            saved_iters = [int(s.split('-')[-1].split('.')[0]) for s in saved_models]
            recent_id = saved_iters.index(max(saved_iters))
            saved_model = saved_models[recent_id]
            print(saved_model)
            model_state = torch.load(saved_model)
            self.netG.load_state_dict(model_state['netG_state_dict'])
            self.netT.load_state_dict(model_state['netT_state_dict'])

            return True
        return False

    def tensor2im(self,depth):
        depth_numpy = depth.cpu().data.float().numpy().transpose(0,2,3,1)
        depth_numpy = (depth_numpy + 1.0) / 2.0 # Unnormalize between 0 and 1
        return depth_numpy*80.0

    def Validate(self):
        self.netG.eval()
        self.netT.eval()
        START_ITER = self.opt.iter
        for self.iteration in range(START_ITER,1000+START_ITER, 1000):
            self.load_prev_model()
            self.Validation()
    
    def crop_center(self, img,cropx,cropy):
        y,x = img.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)    
        return img[starty:starty+cropy,startx:startx+cropx]

    def get_Make3D_depth_manually(self, depth_file):
        depth = scipy.io.loadmat(depth_file, verify_compressed_data_integrity=False)
        depth = depth['Position3DGrid']
        depth = depth[:,:,3]

        depth = cv2.resize(depth, (1800, 2000), interpolation=cv2.INTER_NEAREST)
        depth = self.crop_center(depth, 1600, 1024)
        return depth

    def Validation(self):
        num_samples = len(self.real_val_dataset)
        abs_rel = np.zeros(num_samples, np.float32)
        sq_rel = np.zeros(num_samples,np.float32)
        rmse = np.zeros(num_samples,np.float32)
        rmse_log = np.zeros(num_samples,np.float32)

        with torch.no_grad():
            for i,(self.real_val_image, image_filenames, depth_filenames) in tqdm(enumerate(self.real_val_dataloader)): 
                
                self.real_val_image = Variable(self.real_val_image.cuda())

                _, self.real_translated_image = self.netG(self.real_val_image)
                depth = self.netT(self.real_translated_image)
                depth = depth[-1]
                depth_numpy = self.tensor2im(depth) # 0-80m
                
                for t_id in range(depth_numpy.shape[0]):
                    t_id_global = (i*self.batch_size)+t_id
                    ground_depth = self.get_Make3D_depth_manually(depth_filenames[t_id]) 
                    height, width = ground_depth.shape

                    predicted_depth = cv2.resize(depth_numpy[t_id],(width, height),interpolation=cv2.INTER_LINEAR)

                    save_pred_depth = predicted_depth * 65535/80.0
                    save_pred_depth[save_pred_depth<1e-3] = 1e-3
       
                    save_pred_img = Image.fromarray(save_pred_depth.astype(np.int32), 'I')
                    save_pred_img.save('%s/%05d_pred.png'%(self.output_images_dir, t_id_global))
                
                    mask = np.logical_and(ground_depth > 0.0, ground_depth < 70.0)
                    
                    depth_gt = ground_depth[mask]
                    depth_pred = predicted_depth[mask]
                    depth_pred[depth_pred > 70.0] = 70.0
                    depth_pred[depth_pred < 0.0] = 0.0

                    abs_rel[t_id_global], sq_rel[t_id_global], rmse[t_id_global], rmse_log[t_id_global] = self.compute_errors_make3d(depth_gt, depth_pred)
                    
            print ('{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log'))
            print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                .format(abs_rel.mean(),sq_rel.mean(),rmse.mean(),rmse_log.mean()))

def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--supervision', default='geom', help='None | pseudo | pseudo_scratch | geom | pseudo+geom')
    parser.add_argument('--gen', default='Resnet', help='Resnet | RBDN')
    parser.add_argument('--disc', default='WI', help='3Node | WFWI')
    parser.add_argument('--iter', default=999, type=int, help="Indicate what iteration of the saved model for Validation")
    parser.add_argument('--depth_resize', default='bicubic', help='bicubic | bilinear')
    parser.add_argument('--val', action='store_true', help='Indicate if you want to test the model on the validation data')
    opt = parser.parse_args()
    return opt

if __name__=='__main__':
    opt = get_params()
    solver = Solver(opt)
    solver.Validate()