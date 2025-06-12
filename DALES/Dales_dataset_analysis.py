# -*- coding: utf-8 -*-
"""

This code is used to analyze the DALES dataset distribution

"""
import numpy as np
import glob
import os
import sys
from plyfile import PlyData, PlyElement
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

DATA_DIR = 'D:\\my_test\\DALESObjects'    # Here you want to modify the path to your dataset folder.
train_dir=os.path.join(DATA_DIR, 'train')
test_dir=os.path.join(DATA_DIR, 'test')
validation_dir=test_dir

train_files = glob.glob(os.path.join(train_dir, "*.ply"))
test_files = glob.glob(os.path.join(test_dir, "*.ply"))
validation_files=test_files
files = np.sort(np.hstack((train_files, test_files)))


def read_ply_file(file_path):
    plydata=[]    
    plydata = PlyData.read(file_path)
    data = plydata.elements[0].data
    data_pd = pd.DataFrame(data)
    data_np = np.zeros(data_pd.shape, dtype=np.float32)
    property_names = data[0].dtype.names
    #property_names：('x', 'y', 'z', 'intensity', 'sem_class', 'ins_class')
    for j, name in enumerate(property_names):
        data_np[:, j] = data_pd[name]
    xyz = data_np[:,0:3]
    intensity=data_np[:,3]
    labels = data_np[:,-2].astype(np.uint8)
    labels=labels-1
    return xyz,intensity,labels


xyz=[]
intensity=[]
labels=[]

for pc_path in train_files:
    cloud_name = pc_path.split('\\')[-1][:-4]
    print('Reading:', cloud_name)
    one_xyz,one_intensity,one_labels=read_ply_file(pc_path)
    xyz.append(one_xyz)
    intensity.append(one_intensity)
    labels.append(one_labels)

xyz=np.concatenate(xyz, 0)
intensity=np.concatenate(intensity,0)
labels=np.concatenate(labels,0)

min_label=np.min(labels)
max_label=np.max(labels)

min_intensity=np.min(intensity)
max_intensity=np.max(intensity)

label_names=['Ground','Vegetation','Cars','Trucks','Power lines','Fences','Poles','Buildings']

pd.set_option('display.float_format',lambda x:'%.2f' % x)


for i in range(max_label+1):
    idx=np.argwhere(labels==i)
    selected_intensity=intensity[idx]
    data_dp=pd.DataFrame(selected_intensity)
    print('Label',i,'\'s intensity distribution：')
    print(data_dp.describe())
    fig,ax=plt.subplots(figsize=(5.8,3.6))
    title=label_names[i]
    data_dp.columns=['intensity']
    data_dp.plot(kind='hist',bins=20,color='deepskyblue', edgecolor='black',ax=ax,title=title)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.title.set_size(20)
    plt.ylabel('Number of points',fontsize='15')
    plt.legend(fontsize='15')
    plt.xticks(fontsize='12')
    plt.yticks(fontsize='12')







