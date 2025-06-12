# -*- coding: utf-8 -*-
"""

This code is used to analyze the EVO dataset distribution

"""
import numpy as np
import glob
import os
import sys
from plyfile import PlyData, PlyElement
import pandas as pd
import matplotlib.pyplot as plt


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
    mirror_angle=data_np[:,4].reshape(-1,1)    # Mirror angle
    las_range=data_np[:,5].reshape(-1,1)    # Range
    reflectance=data_np[:,6].reshape(-1,1)    # Reflectance
    echo_deviation=data_np[:,7].reshape(-1,1)    # Echo devidation
    echo_number=data_np[:,11].reshape(-1,1)    # Echo count
    labels = data_np[:,14].astype(np.uint8)
    data_split=data_np[:,15].astype(np.uint8)
    feature=np.concatenate((mirror_angle,las_range,reflectance,echo_deviation,echo_number),axis=1)    #组合特征
    return xyz,feature,labels,data_split

for i in range(len(ply_files)):
    cloud_name = ply_files[i].split('\\')[-1][:-4]
    if cloud_name != 'plot_a':
        continue
    print('Reading:', cloud_name)
    xyz,feature,labels,data_split=read_ply_file(ply_files[i])
    del_index=np.argwhere((labels==0) | (labels==6) | (labels==18))
    #Uncategorized（0）和Noise（18）和Unnatural（6）
    xyz=np.delete(xyz,del_index,axis=0)
    feature=np.delete(feature,del_index,axis=0)
    data_split=np.delete(data_split,del_index,axis=0)
    labels=np.delete(labels,del_index,axis=0)
    labels=labels-2

feature_list=['Mirror angle','Range','Reflectance','Echo deviation','Echo count']
label_names=['Ground','Understorey','Tree trunk','Foliage']
color_list=['deepskyblue','orange','yellowgreen','tomato','blue']

min_label=np.min(labels)
max_label=np.max(labels)

pd.set_option('display.float_format',lambda x:'%.2f' % x)

for i in range(len(feature_list)):
    feature_name=feature_list[i]
    current_feature=feature[:,i]
    print('Analyzing：',feature_name)
    for j in range(max_label+1):
        idx=np.argwhere(labels==j)
        selected_feature=current_feature[idx]
        data_dp=pd.DataFrame(selected_feature)
        print('Label',j,'\'',feature_name,'distribution：')
        print(data_dp.describe())
        fig,ax=plt.subplots(figsize=(5.8,3.6))
        title=label_names[j]
        data_dp.columns=[feature_name]
        data_dp.plot(kind='hist',bins=20,color=color_list[i], edgecolor='black',ax=ax,title=title)
        ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
        ax.title.set_size(20)
        plt.ylabel('Number of points',fontsize='15')
        plt.legend(fontsize='15')
        plt.xticks(fontsize='12')
        plt.yticks(fontsize='12')


############### Redraw the diagram of the echo deviation ################
i=3
feature_name=feature_list[i]
current_feature=feature[:,i]
left_idx=np.argwhere(current_feature<400)
current_feature=current_feature[left_idx].reshape(-1,)
new_labels=labels[left_idx].reshape(-1,)
print('Analyzing：',feature_name)
for j in range(max_label+1):
    idx=np.argwhere(new_labels==j)
    selected_feature=current_feature[idx]
    data_dp=pd.DataFrame(selected_feature)
    print('Label',j,'\'',feature_name,'distributio：')
    print(data_dp.describe())
    fig,ax=plt.subplots(figsize=(5.8,3.6))
    title=label_names[j]
    data_dp.columns=[feature_name]
    data_dp.plot(kind='hist',bins=20,color=color_list[i], edgecolor='black',ax=ax,title=title)
    ax.ticklabel_format(style='sci', scilimits=(-1,2), axis='y')
    ax.title.set_size(20)
    plt.ylabel('Number of points',fontsize='15')
    plt.legend(fontsize='15')
    plt.xticks(fontsize='12')
    plt.yticks(fontsize='12')

