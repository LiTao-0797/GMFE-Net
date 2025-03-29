# -*- coding: utf-8 -*-
"""
This code implements the training, validation, and testing of GMFE-Net on the EVO dataset.

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
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as bk
from keras.models import Model
from keras.utils.np_utils import to_categorical
import time
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

label_to_names = {0: 'Ground', 1: 'Understorey', 2: 'Tree trunk', 3: 'Foliage'}
num_classes = len(label_to_names)

DATA_DIR = os.path.join(BASE_DIR, 'npy_data')
train_dir=os.path.join(DATA_DIR, 'train')
test_dir=os.path.join(DATA_DIR, 'test')
validation_dir=os.path.join(DATA_DIR, 'validation')
train_files = glob.glob(os.path.join(train_dir, "*"))
test_files = glob.glob(os.path.join(test_dir, "*"))
validation_files=glob.glob(os.path.join(validation_dir, "*"))
files = np.sort(np.hstack((train_files, test_files,validation_files)))

def get_file_name(file_path):
    file_name=[]
    for i in file_path:
        cloud_name = i.split('\\')[-1][:-4]
        file_name.append(cloud_name)
    return file_name

train_file_name=get_file_name(train_files)
test_file_name=get_file_name(test_files)
val_file_name=get_file_name(validation_files)

# Part of the hyperparameters
train_steps = 1000
batch_size = 8
val_steps = 160
val_batch_size = 8
test_steps= 34
test_batch_size=8
noise_init = 3.5
num_points = 16384    #2**14
# Note: The steps* batch_size must be less than the number of grid files generated.

########################GMFE-Net model################################
ss = 16    # The K neighborhood value is set to 16
sconv_param_name = ('K', 'D', 'P', 'C', 'links')
sconv_params = [dict(zip(sconv_param_name, sconv_param)) for sconv_param in
                            [(ss*1, 1, num_points//(4**1), 64, []),
                             (ss*1, 1, num_points//(4**2), 128, []),
                             (ss*1, 1, num_points//(4**3), 256, []),
                             (ss*1, 1, num_points//(4**4), 512, [])]]

sdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
sdconv_params = [dict(zip(sdconv_param_name, sdconv_param)) for sdconv_param in
                            [(ss*1, 1, 3, 2),
                             (ss*1, 1, 2, 1),
                             (ss*1, 1, 1, 0),
                             (ss*1, 1, 0, -1)]]

x = 2
fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                        [(64 * x, 0),
                        (32 * x, 0.5)]]

def dense(inputs, outputs, bn_decay=None, with_bn=True, activation='relu'):
    if with_bn:
        inputs =layers.BatchNormalization(momentum=0.98)(inputs)
    dense=layers.Dense(outputs,activation=activation)(inputs)
    return dense

def conv1d(x, filters):
    x = layers.Conv1D(filters, kernel_size=1, padding="valid")(x)
    x = layers.BatchNormalization(momentum=0.98)(x)
    return layers.Activation("relu")(x)

def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated =np.ones((N, 1, P),dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated

def prepare_for_unique_top_k(BD, A):
    indices_duplicated = tf.py_function(find_duplicate_columns, [A], tf.int32)
    BD += tf.reduce_max(BD)*tf.cast(indices_duplicated, tf.float32)

def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    BD = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return BD

def knn_indices_general(queries, points, k, sort=True, unique=True):
    queries_shape = tf.shape(queries)
    batch_size = queries_shape[0]
    point_num = queries_shape[1]
    tmp_k = 0
    BD = batch_distance_matrix_general(queries, points)
    if unique:
        prepare_for_unique_top_k(BD, points)
    _, point_indices = tf.nn.top_k(-BD, k=k+tmp_k, sorted=sort)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, k, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices[:,:,tmp_k:], axis=3)], axis=3)
    return indices

def GMFE_Unit(pts, fts_prev, qrs,rgb,rgb_qrs, K, D, P, C, with_local, bn_decay=None):
    ######## Local Multi-Feature Fusion and Extraction (LocMFE) Block ########
    indices = knn_indices_general(qrs, pts, K, True)
    nn_pts = tf.gather_nd(pts, indices)
    nn_pts_center = tf.expand_dims(qrs, axis=2)
    nn_pts_center=tf.tile(nn_pts_center,[1,1,K,1])
    nn_pts_subtract = tf.subtract(nn_pts, nn_pts_center)

    relative_alpha = tf.expand_dims(tf.math.atan2(nn_pts_subtract[:,:,:,1], nn_pts_subtract[:,:,:,0]), axis=-1)
    relative_xydis = tf.sqrt(tf.reduce_sum(tf.square(nn_pts_subtract[:,:,:,:2]), axis=-1))
    relative_beta = tf.expand_dims(tf.math.atan2(nn_pts_subtract[:,:,:,2], relative_xydis), axis=-1)
    relative_dis = tf.sqrt(tf.reduce_sum(tf.square(nn_pts_subtract), axis=-1, keepdims=True))
    relative_info = tf.concat([relative_dis, nn_pts_subtract,nn_pts_center, nn_pts], axis=-1)

    neighbor_mean = tf.reduce_mean(nn_pts, axis=-2)
    direction = qrs - neighbor_mean
    direction_tile = tf.tile(tf.expand_dims(direction, axis=2), [1, 1,K, 1])
    direction_alpha = tf.expand_dims(tf.math.atan2(direction_tile[:,:,:,1], direction_tile[:,:,:,0]), axis=-1)
    direction_xydis = tf.sqrt(tf.reduce_sum(tf.square(direction_tile[:,:,:,:2]), axis=-1))
    direction_beta = tf.expand_dims(tf.math.atan2(direction_tile[:,:,:,2], direction_xydis), axis=-1)

    angle_alpha = relative_alpha - direction_alpha
    angle_beta = relative_beta - direction_beta
    angle_updated = tf.concat([angle_alpha, angle_beta], axis=-1)

    nn_pts_local=tf.concat([relative_info,angle_updated], axis=-1)

    nn_rgb=tf.gather_nd(rgb, indices)
    nn_rgb_center = tf.tile(tf.expand_dims(rgb_qrs, axis=2),[1,1,K,1])
    nn_rgb_subtract = tf.subtract(nn_rgb, nn_rgb_center)
    nn_rgb_dis =tf.sqrt(tf.reduce_sum(tf.square(nn_rgb_subtract), axis=-1, keepdims=True))

    nn_rgb_local=tf.concat([nn_rgb,nn_rgb_subtract,nn_rgb_dis],axis=-1)

    nn_pts_concat=tf.concat([nn_pts_local,nn_rgb_local],axis=-1)

    [N,P,K,dim] = nn_pts_concat.shape
    nn_fts_local = None

    C_pts_fts = 32
    if with_local:
        nn_fts_local = dense(nn_pts_concat, C_pts_fts ,bn_decay=bn_decay)
        nn_fts_local = dense(nn_fts_local, C_pts_fts*2 ,bn_decay=bn_decay)
    else:
        nn_fts_local = nn_pts_concat


    if fts_prev is not None:
        fts_prev = tf.gather_nd(fts_prev, indices)
        pts_X_0 = tf.concat([nn_fts_local,fts_prev], axis=-1)
    else:
        pts_X_0 = nn_fts_local

    ######## Attention-pooling block ########
    batch_size=pts_X_0.shape[0]
    num_points=pts_X_0.shape[1]
    num_neigh=pts_X_0.shape[2]
    d_in=pts_X_0.shape[3]
    f_reshaped = tf.reshape(pts_X_0, shape=[-1, num_neigh, d_in])
    att_activation=layers.Dense(d_in,use_bias=False)(f_reshaped)
    att_scores=layers.Softmax(axis=1)(att_activation)
    f_agg = f_reshaped * att_scores
    f_agg = tf.reduce_sum(f_agg, axis=1)
    f_agg=tf.reshape(f_agg, [batch_size, num_points, d_in])

    ######## Global multi-feature fusion and extraction (GloMFE) blockk ########
    local_volume = tf.math.pow(tf.reduce_max(tf.reduce_max(relative_dis, -1), -1), 3)
    global_dis = tf.sqrt(tf.reduce_sum(tf.square(qrs), axis=-1, keepdims=True))
    global_volume = tf.pow(tf.reduce_max(global_dis, axis=-1), 3)
    lg_volume_ratio = tf.expand_dims(local_volume / global_volume, -1)

    mean_nn_pts=tf.reduce_mean(nn_pts,axis=-2)
    delta_nn_pts=tf.subtract(mean_nn_pts,qrs)
    dis_nn_pts=tf.sqrt(tf.reduce_sum(tf.square(delta_nn_pts), axis=-1, keepdims=True))
    f_gc=dense(tf.concat([qrs, lg_volume_ratio,dis_nn_pts], axis=-1),d_in//4,bn_decay=bn_decay)

    # Integrates the local and global feature maps
    fts_X=tf.concat([f_agg,f_gc],axis=-1)

    # Produces outputs in GMFE-Net pre-defined shapes for each layer
    fts_X=conv1d(fts_X,C)

    return fts_X

with_local=True

def get_model(layer_pts,layer_rgb, sconv_params, sdconv_params, fc_params, weight_decay=0.0, bn_decay=None, num_classes=num_classes):
    layer_fts_list = [None]
    layer_pts_list = [layer_pts]
    layer_rgb_list=[layer_rgb]

    for layer_idx, layer_param in enumerate(sconv_params):
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        if P == -1:
            qrs = layer_pts
            rgb_qrs=layer_rgb
        else:
            qrs = tf.slice(layer_pts, (0, 0, 0), (-1, P, -1))
            rgb_qrs=tf.slice(layer_rgb, (0, 0, 0), (-1, P, -1))

        layer_fts= GMFE_Unit(layer_pts_list[-1], layer_fts_list[-1], qrs,layer_rgb_list[-1],rgb_qrs, K, D, P, C, with_local=True, bn_decay=None)

        layer_pts = qrs
        layer_rgb=rgb_qrs
        layer_pts_list.append(qrs)
        layer_fts_list.append(layer_fts)
        layer_rgb_list.append(layer_rgb)

    if sdconv_params is not None:
        fts = layer_fts_list[-1]
        for layer_idx, layer_param in enumerate(sdconv_params):
            K = layer_param['K'] 
            D = layer_param['D'] 
            pts_layer_idx = layer_param['pts_layer_idx']
            qrs_layer_idx = layer_param['qrs_layer_idx']

            pts = layer_pts_list[pts_layer_idx + 1]
            qrs = layer_pts_list[qrs_layer_idx + 1]
            fts_qrs = layer_fts_list[qrs_layer_idx + 1]

            rgb=layer_rgb_list[pts_layer_idx + 1]
            rgb_qrs=layer_rgb_list[qrs_layer_idx + 1]

            C = fts_qrs.get_shape()[-1] if fts_qrs is not None else C//2
            P = qrs.get_shape()[1]

            layer_fts= GMFE_Unit(pts, fts, qrs,rgb,rgb_qrs,K, D, P, C, with_local=True, bn_decay=None)

            if fts_qrs is not None:
                fts_concat = tf.concat([layer_fts, fts_qrs], axis=-1)
                fts = dense(fts_concat, C, bn_decay=None)

    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        layer_fts = dense(layer_fts, C, bn_decay=None)
        layer_fts=layers.Dropout(rate=dropout_rate)(layer_fts)

    outputs=dense(layer_fts, num_classes,activation='softmax')
    return outputs

xyz_inputs= keras.Input(batch_shape=(batch_size,num_points, 3))
rgb_inputs= keras.Input(batch_shape=(batch_size,num_points, 3))
outputs=get_model(xyz_inputs,rgb_inputs, sconv_params, sdconv_params, fc_params)

GMFE_Net=keras.Model([xyz_inputs,rgb_inputs],outputs)

keras.backend.clear_session()
# GMFE_Net.summary()    # Review the model structure and parameters

# Another part of the hyperparameters
INITIAL_LR = 0.001
end_lr = 0.00001
epochs=100

training_step_size = train_steps
total_training_steps = train_steps * epochs
print(f"总的训练步数: {total_training_steps}.") 

lr_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(
    INITIAL_LR,
    total_training_steps,
    end_lr,
    power=0.5)

def minibatches(inputs=None, inputs2=None,targets=None, batch_size=None):
    while 1:
        assert len(inputs) == len(targets)
        for start_idx in range(len(inputs)//batch_size):
            excerpt = slice(start_idx*batch_size, start_idx*batch_size + batch_size)
            yield [inputs[excerpt],inputs2[excerpt]], targets[excerpt]

timestamp=time.time()
local_time=time.localtime(timestamp)
current_year=local_time.tm_year
current_month=local_time.tm_mon
current_day=local_time.tm_mday
year_mon_day=str(current_year)+str(current_month)+str(current_day)

save_weights_dir=os.path.join(BASE_DIR,'weights')
if not os.path.exists(save_weights_dir):
    os.mkdir(save_weights_dir)

# Visualize Numpy data in Open3D
import open3d as o3d
def vis_rgb_data(xyz,rgb):
    # xyz:(N,3)
    # rgb:(N,3)
    visual_xyz=xyz
    visual_rgb=rgb
    visual_npy=np.concatenate((visual_xyz,visual_rgb),axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(visual_npy[:, 0:3])
    if visual_npy.shape[1] == 3:
        o3d.visualization.draw_geometries([pc])
    if np.max(visual_npy[:, 3:6]) > 20:
        pc.colors = o3d.utility.Vector3dVector(visual_npy[:, 3:6] / 255.)
    else:
        pc.colors = o3d.utility.Vector3dVector(visual_npy[:, 3:6])
    o3d.visualization.draw_geometries([pc])

def vis_class_data(xyz,label):
    # xyz:(N,3)
    # label:(N,)
    visual_xyz=xyz
    visual_label=label
    class2color = [[205, 173, 0],    # ground -> Earthy yellow
                    [0, 0, 255],  # understory -> Blue
                    [211, 44, 31],  # trunk -> Red
                    [97, 222, 42]]  # foliage -> Green
    class2color=np.asarray(class2color)
    all_true_color=np.zeros([visual_label.shape[0],3])
    for k in range(all_true_color.shape[0]):
        max_idx=visual_label[k]
        all_true_color[k]=class2color[max_idx]

    visual_npy=np.concatenate((visual_xyz,all_true_color),axis=1)
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(visual_npy[:, 0:3])
    if visual_npy.shape[1] == 3:
        o3d.visualization.draw_geometries([pc])
    if np.max(visual_npy[:, 3:6]) > 20:
        pc.colors = o3d.utility.Vector3dVector(visual_npy[:, 3:6] / 255.)
    else:
        pc.colors = o3d.utility.Vector3dVector(visual_npy[:, 3:6])
    o3d.visualization.draw_geometries([pc])
    

# The following functions are used to generate grid files
def sample_data(data, num_sample):
    N = data.shape[0]
    if (N == num_sample):
        return data, range(N)
    elif (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], sample
    else:
        sample = np.random.choice(N, num_sample-N)
        dup_data = data[sample, ...]
        return np.concatenate([data, dup_data], 0), list(range(N))+list(sample)

def sample_data_label(data, label, num_sample):
    new_data, sample_indices = sample_data(data, num_sample)
    new_label = label[sample_indices]
    return new_data, new_label

def room2blocks(data, label, num_point, block_size=1.0, stride=1.0,
                random_sample=False, sample_num=None, sample_aug=1):
    assert(stride<=block_size)
    limit = np.amax(data, 0)[0:3]
    xbeg_list = []
    ybeg_list = []
    if not random_sample:
        num_block_x = int(np.ceil((limit[0] - block_size) / stride)) + 1
        num_block_y = int(np.ceil((limit[1] - block_size) / stride)) + 1
        for i in range(num_block_x):
            for j in range(num_block_y):
                xbeg_list.append(i*stride)
                ybeg_list.append(j*stride)
    else:
        num_block_x = int(np.ceil(limit[0] / block_size))
        num_block_y = int(np.ceil(limit[1] / block_size))
        if sample_num is None:
            sample_num = num_block_x * num_block_y * sample_aug
        for _ in range(sample_num):
            xbeg = np.random.uniform(-block_size, limit[0]) 
            ybeg = np.random.uniform(-block_size, limit[1]) 
            xbeg_list.append(xbeg)
            ybeg_list.append(ybeg)

    block_data_list = []
    block_label_list = []
    idx = 0
    for idx in range(len(xbeg_list)):
       xbeg = xbeg_list[idx]
       ybeg = ybeg_list[idx]
       xcond = (data[:,0]<=xbeg+block_size) & (data[:,0]>=xbeg)
       ycond = (data[:,1]<=ybeg+block_size) & (data[:,1]>=ybeg)
       cond = xcond & ycond    
       if np.sum(cond) < num_points//2:    # Discards point clouds with less than a certain number
           continue
       block_data = data[cond, :]
       block_label = label[cond]
       block_data_sampled, block_label_sampled = \
           sample_data_label(block_data, block_label, num_point)
       block_data_list.append(np.expand_dims(block_data_sampled, 0))
       block_label_list.append(np.expand_dims(block_label_sampled, 0))

    return np.concatenate(block_data_list, 0), \
           np.concatenate(block_label_list, 0)


######################## Main #########################
print('\nIf you want to generate grid training data and save it, press: 1')
print('If you have grid training data and want to start training, press: 2')
print('If you want to generate the grid test (for test and evaluate), press: 3')
print('If you already have grid test data and want to evaluate it, press: 4')
print('If you want to generate grid test data (for visualization of results), press: 5')
print('If you already have grid test data and want to visualize it, press: 6')
print('If you want to exit, press any other key.')
press_key=input()

if press_key=='1':
    save_grid_data_dir=os.path.join(BASE_DIR,'saved_grid_data')
    if not os.path.exists(save_grid_data_dir):
        os.mkdir(save_grid_data_dir)
    
    save_train_data_dir=os.path.join(save_grid_data_dir,'train')
    if not os.path.exists(save_train_data_dir):
        os.mkdir(save_train_data_dir)
    
    save_val_data_dir=os.path.join(save_grid_data_dir,'val')
    if not os.path.exists(save_val_data_dir):
        os.mkdir(save_val_data_dir)
    
    npy_data_dir=os.path.join(BASE_DIR,'npy_data')
    npy_train_data_dir=os.path.join(npy_data_dir,'train')
    npy_val_data_dir=os.path.join(npy_data_dir,'validation')

    block_size=10.0
    stride=1.0
    random_sample=False
    sample_num=None
    sample_aug=1

    t1=time.time()

    print('Training data is generating……')
    for i in range(len(train_file_name)):
        npy_data=[]
        cloud_name=train_file_name[i]
        print(cloud_name,'is processing……')
        current_file_dir = os.path.join(npy_train_data_dir, '{:s}.npy'.format(cloud_name))
        npy_data=np.load(current_file_dir)
        data = npy_data[:,0:6]
        xyz_min=np.amin(data, axis=0)[0:3]
        data[:,0:3]-=xyz_min
        label = npy_data[:,-1].astype(np.uint8)
        data_batch=[]
        label_batch=[]
        data_batch, label_batch = room2blocks(data, label, num_points, block_size, stride,
                                              random_sample, sample_num, sample_aug)
        for b in range(data_batch.shape[0]):
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx+block_size/2)
            data_batch[b, :, 1] -= (miny+block_size/2)
        save_data_name=cloud_name+'_data'+'.npy'
        save_label_name=cloud_name+'_label'+'.npy'
        np.save(os.path.join(save_train_data_dir,save_data_name),data_batch)
        np.save(os.path.join(save_train_data_dir,save_label_name),label_batch)
    
    print('Validation data is generating……')

    stride=2.0

    for i in range(len(val_file_name)):
        npy_data=[]
        cloud_name=val_file_name[i]
        print(cloud_name,'is processing……')
        current_file_dir = os.path.join(npy_val_data_dir, '{:s}.npy'.format(cloud_name))
        npy_data=np.load(current_file_dir)
        data = npy_data[:,0:6]
        xyz_min=np.amin(data, axis=0)[0:3]
        data[:,0:3]-=xyz_min
        label = npy_data[:,-1].astype(np.uint8)
        data_batch=[]
        label_batch=[]
        data_batch, label_batch = room2blocks(data, label, num_points, block_size, stride,
                                              random_sample, sample_num, sample_aug)

        for b in range(data_batch.shape[0]):
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx+block_size/2)
            data_batch[b, :, 1] -= (miny+block_size/2)
        save_data_name=cloud_name+'_data'+'.npy'
        save_label_name=cloud_name+'_label'+'.npy'
        np.save(os.path.join(save_val_data_dir,save_data_name),data_batch)
        np.save(os.path.join(save_val_data_dir,save_label_name),label_batch)

    t2=time.time()
    print('It took a total of ',(t2-t1)/60/60,' hours to generate the training and validation grid files.')


elif press_key=='2':
    train_data_dir=os.path.join(BASE_DIR,'saved_grid_data\\train\\')
    val_data_dir=os.path.join(BASE_DIR,'saved_grid_data\\val')

    train_data_files = glob.glob(os.path.join(train_data_dir, "*"))
    val_data_files=glob.glob(os.path.join(val_data_dir, "*"))

    train_data_list=[]
    train_label_list=[]
    val_data_list=[]
    val_label_list=[]

    for i in range(len(train_data_files)):
        temp_train_data=[]
        current_file_name=train_data_files[i].split('\\')[-1][:-4]
        split_file_name=current_file_name.split('_')[-1]
        if split_file_name=='data':
            temp_train_data=np.load(train_data_files[i])
            train_data_list.append(temp_train_data)
        elif split_file_name=='label':
            temp_train_data=np.load(train_data_files[i])
            train_label_list.append(temp_train_data)
    train_data_batch=np.concatenate(train_data_list, 0)
    train_label_batch=np.concatenate(train_label_list, 0)
    print('The number of training files is:',train_data_batch.shape[0])

    for i in range(len(val_data_files)):
        temp_val_data=[]
        current_file_name=val_data_files[i].split('\\')[-1][:-4]
        split_file_name=current_file_name.split('_')[-1]
        if split_file_name=='data':
            temp_val_data=np.load(val_data_files[i])
            val_data_list.append(temp_val_data)
        elif split_file_name=='label':
            temp_val_data=np.load(val_data_files[i])
            val_label_list.append(temp_val_data)
    val_data_batch=np.concatenate(val_data_list, 0)
    val_label_batch=np.concatenate(val_label_list, 0)
    print('The number of validation files is:',val_data_batch.shape[0])

    def shuffle_data(data, labels):
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx, ...], idx

    train_data_batch, train_label_batch, _ = shuffle_data(train_data_batch, train_label_batch)
    val_data_batch, val_label_batch, _ = shuffle_data(val_data_batch, val_label_batch)

    train_xyz=train_data_batch[0:train_steps*batch_size,:,0:3]
    train_rgb=train_data_batch[0:train_steps*batch_size,:,3:6]
    train_label=to_categorical(train_label_batch[0:train_steps*batch_size, ...],num_classes)

    val_xyz=val_data_batch[0:val_batch_size*val_steps,:,0:3]
    val_rgb=val_data_batch[0:val_batch_size*val_steps,:,3:6]
    val_label=to_categorical(val_label_batch[0:val_batch_size*val_steps, ...],num_classes)

    t1=time.time()

    GMFE_Net.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_scheduler),
                      loss=keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])

    checkpoint_filepath = os.path.join(save_weights_dir,'check_point'+'_'+year_mon_day)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
                                                        checkpoint_filepath,
                                                        monitor="val_accuracy",
                                                        save_best_only=True,
                                                        save_weights_only=True)

    history=GMFE_Net.fit(minibatches(train_xyz,train_rgb,train_label,batch_size),
                          steps_per_epoch=len(train_xyz)//batch_size,
                          validation_data=minibatches(val_xyz,val_rgb,val_label,batch_size),
                          validation_steps=len(val_xyz)//batch_size,
                          epochs=epochs,
                          callbacks=[checkpoint_callback,
                                      keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=20)]) 
    
    t2=time.time()
    print('Total training time:',(t2-t1)/60/60,'hours.')


elif press_key=='3':
    save_grid_data_dir=os.path.join(BASE_DIR,'saved_grid_data')

    save_test_data_dir=os.path.join(save_grid_data_dir,'test')
    if not os.path.exists(save_test_data_dir):
        os.mkdir(save_test_data_dir)

    npy_data_dir=os.path.join(BASE_DIR,'npy_data')
    npy_test_data_dir=os.path.join(npy_data_dir,'test')

    block_size=10.0
    stride=5.0
    random_sample=False
    sample_num=None
    sample_aug=1

    t1=time.time()

    print('Test data is generating……')

    for i in range(len(test_file_name)):
        npy_data=[]
        cloud_name=test_file_name[i]
        print(cloud_name,'is processing……')
        current_file_dir = os.path.join(npy_test_data_dir, '{:s}.npy'.format(cloud_name))
        npy_data=np.load(current_file_dir)
        data = npy_data[:,0:6]

        xyz_min=np.amin(data, axis=0)[0:3]
        data[:,0:3]-=xyz_min

        label = npy_data[:,-1].astype(np.uint8)
        data_batch=[]
        label_batch=[]
        data_batch, label_batch = room2blocks(data, label, num_points, block_size, stride,
                                              random_sample, sample_num, sample_aug)

        for b in range(data_batch.shape[0]):
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx+block_size/2)
            data_batch[b, :, 1] -= (miny+block_size/2)

        save_data_name=cloud_name+'_data'+'.npy'
        save_label_name=cloud_name+'_label'+'.npy'
        np.save(os.path.join(save_test_data_dir,save_data_name),data_batch)
        np.save(os.path.join(save_test_data_dir,save_label_name),label_batch)

    t2=time.time()
    print('It took a total of ',(t2-t1)/60,' minutes to generate the test grid files.')


elif press_key=='4':
    checkpoint_filepath = os.path.join(BASE_DIR,'weights\\check_point_2024126')
    GMFE_Net.load_weights(checkpoint_filepath)

    test_data_dir=os.path.join(BASE_DIR,'saved_grid_data\\test\\')
    test_data_files = glob.glob(os.path.join(test_data_dir, "*"))

    test_data_list=[]
    test_label_list=[]

    for i in range(len(test_data_files)):
        temp_test_data=[]
        current_file_name=test_data_files[i].split('\\')[-1][:-4]
        split_file_name=current_file_name.split('_')[-1]
        if split_file_name=='data':
            temp_test_data=np.load(test_data_files[i])
            test_data_list.append(temp_test_data)
        elif split_file_name=='label':
            temp_test_data=np.load(test_data_files[i])
            test_label_list.append(temp_test_data)
    test_data_batch=np.concatenate(test_data_list, 0)
    test_label_batch=np.concatenate(test_label_list, 0)
    print('The number of test files is:',test_data_batch.shape[0])

    def shuffle_data(data, labels):
        idx = np.arange(len(labels))
        np.random.shuffle(idx)
        return data[idx, ...], labels[idx, ...], idx

    test_data_batch, test_label_batch, _ = shuffle_data(test_data_batch, test_label_batch)

    test_xyz=test_data_batch[0:test_steps*test_batch_size,:,0:3]
    test_rgb=test_data_batch[0:test_steps*test_batch_size,:,3:6]
    test_label=test_label_batch[0:test_steps*test_batch_size, ...]

    all_pre_labels=[]
    print('Predicting……')
    for i in range(test_steps):
        input_rgbs=[]
        input_neighbors = []
        input_pools = []
        input_up_samples = []
        excerpt = slice(i*batch_size, i*batch_size + batch_size)
        pred = GMFE_Net.predict([test_xyz[excerpt],test_rgb[excerpt]])
        pred = np.argmax(pred, axis=-1)
        all_pre_labels.append(pred)

    all_pre_labels=np.array(all_pre_labels)
    num_points_pre=all_pre_labels.shape[-1]
    all_pre_labels=all_pre_labels.reshape(-1,num_points_pre)

    if all_pre_labels.shape != test_label.shape:
        print('The number of predicted and real labels is not consistent!')

    num_batch = len(all_pre_labels)

    gt_classes = [0 for _ in range(num_classes)]
    positive_classes = [0 for _ in range(num_classes)]
    true_positive_classes = [0 for _ in range(num_classes)]

    print('Evaluating……')
    for i in range(num_batch):
        npy_pre_data=[]
        npy_true_data=[]
        pred_label=all_pre_labels[i]
        gt_label=test_label[i]
        for j in range(gt_label.shape[0]):
            gt_l = int(gt_label[j])
            pred_l = int(pred_label[j])
            gt_classes[gt_l] += 1
            positive_classes[pred_l] += 1
            true_positive_classes[gt_l] += int(gt_l==pred_l)
    
    print(gt_classes)
    print(positive_classes)
    print(true_positive_classes)

    print('Overall accuracy: {0}'.format(sum(true_positive_classes)/float(sum(positive_classes))))
    
    print('IoU:')
    iou_list = []
    for i in range(num_classes):
        if_zero=gt_classes[i]+positive_classes[i]-true_positive_classes[i]
        if if_zero==0:
            iou=0
        else:
            iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        print(iou)
        iou_list.append(iou)
        
    print('\n mIoU: \n',sum(iou_list)/num_classes)


elif press_key=='5':
    t1=time.time()
    print('Grid files are generating……')
    block_size=10
    stride=10
    random_sample=False
    sample_num=None
    sample_aug=1

    vis_test_dir=os.path.join(BASE_DIR,'visulization')
    if not os.path.exists(vis_test_dir):
        os.mkdir(vis_test_dir)

    npy_data_dir=os.path.join(BASE_DIR,'npy_data')
    npy_test_data_dir=os.path.join(npy_data_dir,'test')

    for i in range(len(test_file_name)):
        npy_data=[]
        cloud_name=test_file_name[i]
        print(cloud_name,'is processing……')
        current_file_dir = os.path.join(npy_test_data_dir, '{:s}.npy'.format(cloud_name))
        npy_data=np.load(current_file_dir)
        data = npy_data[:,0:6]

        xyz_min=np.amin(data, axis=0)[0:3]
        data[:,0:3]-=xyz_min
        label = npy_data[:,-1].astype(np.uint8)
        data_batch=[]
        label_batch=[]
        data_batch, label_batch = room2blocks(data, label, num_points, block_size, stride,
                                              random_sample, sample_num, sample_aug)
        real_data=[]
        real_data=data_batch.reshape(-1,data_batch.shape[-1])
        real_label=label_batch.reshape(-1,1)
        concat_real=np.concatenate((real_data,real_label),axis=1)
        real_name=test_file_name[i]+'_real'+'.npy'
        np.save(os.path.join(vis_test_dir,real_name),concat_real)

        for b in range(data_batch.shape[0]):
            minx = min(data_batch[b, :, 0])
            miny = min(data_batch[b, :, 1])
            data_batch[b, :, 0] -= (minx+block_size/2)
            data_batch[b, :, 1] -= (miny+block_size/2)

        save_data_name=test_file_name[i]+'_data'+'.npy'
        save_label_name=test_file_name[i]+'_label'+'.npy'
        np.save(os.path.join(vis_test_dir,save_data_name),data_batch)
        np.save(os.path.join(vis_test_dir,save_label_name),label_batch)

    t2=time.time()
    print('It takes',(t2-t1)/60,'minutes to generate grid files.')


elif press_key=='6':
    checkpoint_filepath = os.path.join(BASE_DIR,'weights\\check_point_2024126')
    GMFE_Net.load_weights(checkpoint_filepath)

    vis_test_dir=os.path.join(BASE_DIR,'visulization')

    current_vis_name=test_file_name[0]

    current_vis_data_dir=os.path.join(vis_test_dir, '{:s}_data.npy'.format(current_vis_name))
    current_vis_label_dir=os.path.join(vis_test_dir, '{:s}_label.npy'.format(current_vis_name))
    current_npy_data=np.load(current_vis_data_dir)
    current_npy_label=np.load(current_vis_label_dir)

    test_xyz=current_npy_data[:,:,0:3]
    test_rgb=current_npy_data[:,:,3:6]
    test_label=current_npy_label

    num_size=test_xyz.shape[0]
    aug_size=math.ceil(num_size/batch_size)*batch_size
    del_size=aug_size-num_size
    del_np=np.ones((del_size,test_xyz.shape[1],test_xyz.shape[2]))
    concat_xyz=np.concatenate((test_xyz,del_np),axis=0)
    concat_rgb=np.concatenate((test_rgb,del_np),axis=0)

    all_pre_labels=[]
    print('Predicting……')
    for i in range(aug_size//batch_size):
        excerpt = slice(i*batch_size, i*batch_size + batch_size)
        pred = GMFE_Net.predict([concat_xyz[excerpt],concat_rgb[excerpt]])
        pred = np.argmax(pred, axis=-1)
        all_pre_labels.append(pred)

    all_pre_labels=np.array(all_pre_labels)
    num_points_pre=all_pre_labels.shape[-1]
    all_pre_labels=all_pre_labels.reshape(-1,num_points_pre)
    all_pre_labels=all_pre_labels[0:num_size,:]

    real_xyz_data_dir=os.path.join(vis_test_dir, '{:s}_real.npy'.format(current_vis_name))
    real_xyz_data=np.load(real_xyz_data_dir)
    real_vis_xyz=real_xyz_data[:,0:3]
    real_vis_rgb=real_xyz_data[:,3:6]

    vis_real_label=current_npy_label.reshape(-1,)
    vis_pre_label=all_pre_labels.reshape(-1,)

    # # Visualize realistic point clouds
    # vis_rgb_data(real_vis_xyz,real_vis_rgb)
    
    # Visualize a correctly segmented point cloud
    vis_class_data(real_vis_xyz,vis_real_label)
    # Rotate and scale well, and then use Ctrl+C to save the current state
    # The later visualization of the point cloud can be viewed and saved in the same state with Ctrl+V
    
    # Visualize the point cloud of prediction segmentation
    vis_class_data(real_vis_xyz,vis_pre_label)
    
    # Visualize correctly segmented point clouds (canopy removed)
    def del_one_class(input_xyz,input_label,del_label):
        del_index=np.argwhere(input_label==del_label)
        remain_xyz=np.delete(input_xyz,del_index,axis=0)
        remain_label=np.delete(input_label,del_index,axis=0)
        return remain_xyz,remain_label

    del_label=3
    remain_real_vis_xyz,remain_vis_real_label=del_one_class(real_vis_xyz,vis_real_label,del_label)
    vis_class_data(remain_real_vis_xyz,remain_vis_real_label)

    # Visualize the point cloud of prediction segmentation (canopy removed)
    remain_real_vis_xyz,remain_vis_pre_label=del_one_class(real_vis_xyz,vis_pre_label,del_label)
    vis_class_data(remain_real_vis_xyz,remain_vis_pre_label)


else:
    pass

