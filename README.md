# SharinGAN
Official repo for the work titled "SharinGAN: Combining Synthetic and Real Data for Unsupervised GeometryEstimation"

The official project website for this work can be found <a href="https://koutilya-pnvr.github.io/SharinGAN/">here</a>.

# Requirements
Python 2.7
Pytorch 0.4.1

The trained model files for both the tasks are made available here <a href="https://drive.google.com/drive/folders/1SRznz7AlezF655doEZSAxk_YFSdGGhd4?usp=sharing">here</a>. The pretrained_models folder contains the pretrained model for the generator and the primary task networks before end-to-end training the SharinGAN network as a whole. The final trained models are present in Face_Normal_Estimation/ and Monocular_Depth_Estimation/ directories of the google drive.

The environment.yml file is also provided for one to replicate the environment.

We added the training and validation codes for both the tasks of Monocular Depth Estimation and Face Normal Estimation. We hope to improve the repository with time. We appreciate your inputs and feedback

## Monocular Depth Estimation
Place the saved model file (Depth_Estimator_WI_geom_bicubic_da-144999.pth.tar) inside a newly created folder *Monocular_Depth_Estimation/saved_models/* of the current repo.

The dataset files required for the dataloaders Kitti_dataloader.py and VKitti_dataloader.py are made available at *Monocular_Depth_Estimation/dataset_files/*.

Place the *Monocular_Depth_Estimation/dataset_files/Kitti/.txt* files in the original downloaded kitti/ dataset folder. Similarly place the *Monocular_Depth_Estimation/dataset_files/VKitti/.txt* files in the original downloaded Virtual_Kitti/ dataset folder.
