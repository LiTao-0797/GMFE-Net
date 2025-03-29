# -*- coding: utf-8 -*-
"""
This code is used to implement the preprocessing of the EVO dataset.

Acknowledgments:
This project wouldn't have been possible without the support and contributions of several individuals and resources.
*in no particular order*
Thanks to:
    https://github.com/charlesq34/pointnet
    https://github.com/charlesq34/pointnet2
    https://github.com/HuguesTHOMAS/KPConv
    https://github.com/hkust-vgd/shellnet
    https://github.com/QingyongHu/RandLA-Net
    https://github.com/leofansq/SCF-Net
    https://keras.io/examples/vision/pointnet/
"""

import numpy as np
import glob
import os
import sys
from plyfile import PlyData, PlyElement
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

data_path='D:\\my_test\\evo2016_mls_labelled'    # Here you want to modify the path to your dataset folder.

ply_files = glob.glob(os.path.join(data_path, "*.ply"))

def read_ply_file(file_path):
    plydata=[]
    plydata = PlyData.read(file_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float32)
    property_names = data[0].dtype.names
    for j, name in enumerate(property_names):
        data_np[:, j] = data_pd[name]
    xyz = data_np[:,0:3]
    mirror_angle=data_np[:,4].reshape(-1,1)    # There is only one value for mirror angle
    labels = data_np[:,14].astype(np.uint8)
    data_split=data_np[:,15].astype(np.uint8)
    feature=np.concatenate((mirror_angle,mirror_angle,mirror_angle),axis=1)
    return xyz,feature,labels,data_split

def feature_norm(feature_in):
    angle=feature_in[:,0]
    angle=angle/360.0    # The value range of mirror angle is [0,360]
    angle=angle.reshape(-1,1)
    feature_out=np.concatenate((angle,angle,angle),axis=1)    # Tile the mirror angle to the same shape as RGB.
    return feature_out

out_folder = os.path.join(BASE_DIR,'npy_data')
os.makedirs(out_folder) if not os.path.exists(out_folder) else None

train_folder=os.path.join(out_folder,'train')
os.makedirs(train_folder) if not os.path.exists(train_folder) else None

val_folder=os.path.join(out_folder,'validation')
os.makedirs(val_folder) if not os.path.exists(val_folder) else None

test_folder=os.path.join(out_folder,'test')
os.makedirs(test_folder) if not os.path.exists(test_folder) else None

for i in range(len(ply_files)):        
    cloud_name = ply_files[i].split('\\')[-1][:-4]
    print(cloud_name,'is processing……')
    xyz,feature,labels,data_split=read_ply_file(ply_files[i])
    del_index=np.argwhere((labels==0) | (labels==6) | (labels==18))
    xyz=np.delete(xyz,del_index,axis=0)
    feature=np.delete(feature,del_index,axis=0)
    data_split=np.delete(data_split,del_index,axis=0)
    labels=np.delete(labels,del_index,axis=0)
    labels=labels-2
    # The above categories follow as:
    # Kaijaluoto, R., Kukko, A., El Issaoui, A., Hyyppä, J., & Kaartinen, H. (2022).
    # Semantic segmentation of point cloud data using raw laser scanner measurements and deep neural networks.
    # ISPRS Open Journal of Photogrammetry and Remote Sensing, 3, 100011.
    # https://doi.org/10.1016/j.ophoto.2021.100011

    if cloud_name=='plot_a':
        train_index=np.argwhere(data_split==0)
        train_xyz=xyz[train_index.reshape(-1,)]
        train_feature=feature[train_index.reshape(-1,)]
        train_feature=feature_norm(train_feature)
        train_labels=labels[train_index.reshape(-1,)]
        train_labels=train_labels.reshape(-1,1)
        train_data=np.concatenate((train_xyz,train_feature,train_labels),axis=1)
        train_npy_file=os.path.join(train_folder, 'train_'+cloud_name + '.npy')
        np.save(train_npy_file,train_data)
        
        val_index=np.argwhere(data_split==1)
        val_xyz=xyz[val_index.reshape(-1,)]
        val_feature=feature[val_index.reshape(-1,)]
        val_feature=feature_norm(val_feature)
        val_labels=labels[val_index.reshape(-1,)]
        val_labels=val_labels.reshape(-1,1)
        val_data=np.concatenate((val_xyz,val_feature,val_labels),axis=1)
        val_npy_file=os.path.join(val_folder, 'val_'+cloud_name + '.npy')
        np.save(val_npy_file,val_data)
        
        test_index=np.argwhere(data_split==2)
        test_xyz=xyz[test_index.reshape(-1,)]
        test_feature=feature[test_index.reshape(-1,)]
        test_feature=feature_norm(test_feature)
        test_labels=labels[test_index.reshape(-1,)]
        test_labels=test_labels.reshape(-1,1)
        test_data=np.concatenate((test_xyz,test_feature,test_labels),axis=1)
        test_npy_file=os.path.join(test_folder, 'test_'+cloud_name + '.npy')
        np.save(test_npy_file,test_data)






