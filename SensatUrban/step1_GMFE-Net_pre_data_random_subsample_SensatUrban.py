# -*- coding: utf-8 -*-
"""
This code is used to implement the preprocessing of the SensatUrban dataset.

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
from plyfile import PlyData
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import time

t1=time.time()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

'''
Divide the downloaded dataset files into train, test, and validation folders.

val_file_name = ['birmingham_block_1',
                      'birmingham_block_5',
                      'cambridge_block_10',
                      'cambridge_block_7']

test_file_name = ['birmingham_block_2', 'birmingham_block_8',
                       'cambridge_block_15', 'cambridge_block_22',
                       'cambridge_block_16', 'cambridge_block_27']

train_file_name = ['birmingham_block_0', 'birmingham_block_3',
                       'birmingham_block_4','birmingham_block_6',
                       'birmingham_block_7','birmingham_block_9',
                       'birmingham_block_10','birmingham_block_11',
                       'birmingham_block_12','birmingham_block_13',
                       'cambridge_block_2','cambridge_block_3',
                       'cambridge_block_4','cambridge_block_6',
                       'cambridge_block_8','cambridge_block_9',
                       'cambridge_block_12','cambridge_block_13',
                       'cambridge_block_14','cambridge_block_17',
                       'cambridge_block_18','cambridge_block_19',
                       'cambridge_block_20','cambridge_block_21',
                       'cambridge_block_23','cambridge_block_25',
                       'cambridge_block_26','cambridge_block_28',
                       'cambridge_block_32','cambridge_block_33']

The above division follows: https://github.com/QingyongHu/SensatUrban
'''

DATA_DIR = 'D:\\my_test\\SensatUrban_DataSet'    # Here you want to modify the path to your dataset folder.
train_dir=os.path.join(DATA_DIR, 'train')
test_dir=os.path.join(DATA_DIR, 'test')
validation_dir=os.path.join(DATA_DIR, 'validation')

train_files = glob.glob(os.path.join(train_dir, "*.ply"))
test_files = glob.glob(os.path.join(test_dir, "*.ply"))
validation_files=glob.glob(os.path.join(validation_dir, "*.ply"))
files = np.sort(np.hstack((train_files, test_files,validation_files)))

preparation_types = ['random']
random_sample_ratio = 20

def random_sub_sampling(points, features=None, labels=None, sub_ratio=10, verbose=0):
    num_input = np.shape(points)[0]
    num_output = num_input // sub_ratio
    idx = np.random.choice(num_input, num_output)

    if (features is None) and (labels is None):
        return points[idx]
    elif labels is None:
        return points[idx], features[idx]
    elif features is None:
        return points[idx], labels[idx]
    else:
        return points[idx], features[idx], labels[idx]

for sample_type in preparation_types:
    for pc_path in files:
        cloud_name = pc_path.split('\\')[-1][:-4]
        print('Processing:', cloud_name)

        out_folder = os.path.join(BASE_DIR, sample_type + '_{:.3f}'.format(random_sample_ratio))
        os.makedirs(out_folder) if not os.path.exists(out_folder) else None

        if os.path.exists(os.path.join(out_folder, cloud_name + '_KDTree.pkl')):
            print(cloud_name, 'has been processed')
            continue

        plydata = PlyData.read(pc_path)
        data = plydata.elements[0].data
        data_pd = pd.DataFrame(data)
        data_np = np.zeros(data_pd.shape, dtype=np.float32)
        property_names = data[0].dtype.names
        for j, name in enumerate(property_names): 
            data_np[:, j] = data_pd[name]

        xyz = data_np[:,0:3]
        rgb=data_np[:,3:6]
        labels = data_np[:,-1].astype(np.uint8)

        sub_npy_file = os.path.join(out_folder, cloud_name + '.npy')
        sub_xyz, sub_rgb, sub_labels = random_sub_sampling(xyz, rgb, labels, random_sample_ratio)
        sub_rgb = sub_rgb / 255.0

        sub_labels = np.reshape(sub_labels,(-1,1))
        sub_npy=np.concatenate((sub_xyz,sub_rgb,sub_labels),axis=1)
        np.save(sub_npy_file,sub_npy) 

        search_tree = KDTree(sub_xyz, leaf_size=50)

        kd_tree_file = os.path.join(out_folder, cloud_name + '_KDTree.pkl')
        with open(kd_tree_file, 'wb') as f:
            pickle.dump(search_tree, f)

t2=time.time()
print('It took a total of ',(t2-t1)/60,' minutes to generate the downsampled npy and KDTree files.')
