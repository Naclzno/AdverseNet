
import numpy as np
import torch
import numba as nb
from torch.utils import data

def cart2polar(input_xyz):
    rho = np.sqrt(input_xyz[:, 0] ** 2 + input_xyz[:, 1] ** 2)
    phi = np.arctan2(input_xyz[:, 1], input_xyz[:, 0])
    return np.stack((rho, phi, input_xyz[:, 2]), axis=1)

def polar2cat(input_xyz_polar):
    x = input_xyz_polar[0] * np.cos(input_xyz_polar[1])
    y = input_xyz_polar[0] * np.sin(input_xyz_polar[1])
    return np.stack((x, y, input_xyz_polar[2]), axis=0)


class DatasetWrapper_AdverseNet_train(data.Dataset):
    def __init__(self, in_dataset, grid_size, grid_size_vox=None, fill_label=0, fixed_volume_space=False,
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5],
                 rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, 
                 trans_std=[0.1, 0.1, 0.1]):
        'Initialization'
        self.point_dataset = in_dataset
        self.grid_size = np.asarray(grid_size).astype(np.int32) # [480，360，32]
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32) # [480,360,32]
        self.fill_label = fill_label # 0
        self.fixed_volume_space = fixed_volume_space # True
        self.max_volume_space = max_volume_space # [50,3.1415926,3]
        self.min_volume_space = min_volume_space # [0, -3.1415926,-5]
        self.rotate_aug = rotate_aug # True
        self.flip_aug = flip_aug # True
        self.scale_aug = scale_aug # True
        self.transform_aug = transform_aug # True
        self.trans_std = trans_std # [0.1,0.1,0.1]

    def __len__(self):
        return len(self.point_dataset)
    
    
    def __getitem__(self, index):
        
        data = self.point_dataset[index]
        points_rain, labels_rain, points_snow, labels_snow, points_fog, labels_fog = data

        data_tuple_rain = self.process_data(points_rain, labels_rain)
        data_tuple_snow = self.process_data(points_snow, labels_snow)
        data_tuple_fog = self.process_data(points_fog, labels_fog)

        data_tuple = data_tuple_rain + data_tuple_snow + data_tuple_fog

        return data_tuple
    
    def process_data(self, points, labels):
        xyz, feat = points[:, :3], points[:, 3]
        
        # random data augmentation by rotation
        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        # random data augmentation by flip x , y or x+y
        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        
        # random points augmentation by scale x & y
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        
        # random points augmentation by translate xyz
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate
        
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz) 

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)

        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space) # [50,3.1415926,3]
            min_bound = np.asarray(self.min_volume_space) # [0, -3.1415926,-5]
        
        # get grid index
        crop_range = max_bound - min_bound # [50, 2pi ,8]

        intervals = crop_range / (self.grid_size) # [50, 2pi ,8] / [480，360，32] = [5/48，pi/180，0.25] 
        intervals_vox = crop_range / (self.grid_size_vox) # [50, 2pi ,8] / [480，360，32]

        if (intervals == 0).any(): 
            print("Zero interval!")
        

        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)
        
        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32) 
         
        grid_ind_vox_float = ((xyz_pol_grid - min_bound) / intervals_vox).astype(np.float32)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound 
        return_xyz = xyz_pol - voxel_centers

        feat = np.expand_dims(feat, axis=1)  

        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1) 

        return (grid_ind, labels, return_feat, grid_ind_vox_float)


def seg_custom_collate_fn_train(data):

    grid_ind_list_rain = [d[0] for d in data]  
    point_label_list_rain = [d[1] for d in data]  
    point_feat_list_rain = [d[2] for d in data]  
    grid_ind_vox_list_rain = [d[3] for d in data]  

    grid_ind_list_snow = [d[4] for d in data]  
    point_label_list_snow = [d[5] for d in data]  
    point_feat_list_snow = [d[6] for d in data]  
    grid_ind_vox_list_snow = [d[7] for d in data] 

    grid_ind_list_fog = [d[8] for d in data]  
    point_label_list_fog = [d[9] for d in data]  
    point_feat_list_fog = [d[10] for d in data]  
    grid_ind_vox_list_fog = [d[11] for d in data]  

    data_rain = (point_feat_list_rain, grid_ind_list_rain, point_label_list_rain, grid_ind_vox_list_rain)
    data_snow = (point_feat_list_snow, grid_ind_list_snow, point_label_list_snow, grid_ind_vox_list_snow)
    data_fog = (point_feat_list_fog, grid_ind_list_fog, point_label_list_fog, grid_ind_vox_list_fog)
      
    
    return data_rain, data_snow, data_fog

class DatasetWrapper_AdverseNet_eval(data.Dataset):
    def __init__(self, in_dataset, grid_size, grid_size_vox=None, fill_label=0, fixed_volume_space=False,
                 max_volume_space=[50, np.pi, 3], min_volume_space=[0, -np.pi, -5],
                 rotate_aug=False, flip_aug=False, scale_aug=False, transform_aug=False, 
                 trans_std=[0.1, 0.1, 0.1]):
        'Initialization'
        self.point_dataset = in_dataset 
        self.grid_size = np.asarray(grid_size).astype(np.int32) # [480，360，32]
        self.grid_size_vox = np.asarray(grid_size_vox).astype(np.int32) # [480,360,32]
        self.fill_label = fill_label # 0
        self.fixed_volume_space = fixed_volume_space # True
        self.max_volume_space = max_volume_space # [50,3.1415926,3]
        self.min_volume_space = min_volume_space # [0, -3.1415926,-5]
        self.rotate_aug = rotate_aug # True
        self.flip_aug = flip_aug # True
        self.scale_aug = scale_aug # True
        self.transform_aug = transform_aug # True
        self.trans_std = trans_std # [0.1,0.1,0.1]

    def __len__(self):
        return len(self.point_dataset)
    
    
    def __getitem__(self, index):
        data = self.point_dataset[index] 
        points, labels, subcondition = data
        xyz, feat = points[:, :3], points[:, 3]

        if self.rotate_aug:
            rotate_rad = np.deg2rad(np.random.random() * 360) - np.pi
            c, s = np.cos(rotate_rad), np.sin(rotate_rad)
            j = np.matrix([[c, s], [-s, c]])
            xyz[:, :2] = np.dot(xyz[:, :2], j)

        if self.flip_aug:
            flip_type = np.random.choice(4, 1)
            if flip_type == 1:
                xyz[:, 0] = -xyz[:, 0]
            elif flip_type == 2:
                xyz[:, 1] = -xyz[:, 1]
            elif flip_type == 3:
                xyz[:, :2] = -xyz[:, :2]
        
        # random points augmentation by scale x & y
        if self.scale_aug:
            noise_scale = np.random.uniform(0.95, 1.05)
            xyz[:, 0] = noise_scale * xyz[:, 0]
            xyz[:, 1] = noise_scale * xyz[:, 1]
        
        # random points augmentation by translate xyz
        if self.transform_aug:
            noise_translate = np.array([np.random.normal(0, self.trans_std[0], 1),
                                        np.random.normal(0, self.trans_std[1], 1),
                                        np.random.normal(0, self.trans_std[2], 1)]).T
            xyz[:, 0:3] += noise_translate
        
        # convert coordinate into polar coordinates
        xyz_pol = cart2polar(xyz) 

        max_bound_r = np.percentile(xyz_pol[:, 0], 100, axis=0)
        min_bound_r = np.percentile(xyz_pol[:, 0], 0, axis=0)
        max_bound = np.max(xyz_pol[:, 1:], axis=0)
        min_bound = np.min(xyz_pol[:, 1:], axis=0)

        max_bound = np.concatenate(([max_bound_r], max_bound))
        min_bound = np.concatenate(([min_bound_r], min_bound))

        if self.fixed_volume_space:
            max_bound = np.asarray(self.max_volume_space)
            min_bound = np.asarray(self.min_volume_space)
        
        # get grid index
        crop_range = max_bound - min_bound # [50, 2pi ,8]
        intervals = crop_range / (self.grid_size) # [50, 2pi ,8] / [480，360，32] = [5/48，pi/180，0.25] 
        intervals_vox = crop_range / (self.grid_size_vox) # [50, 2pi ,8] / [480，360，32]

        if (intervals == 0).any(): 
            print("Zero interval!")

        xyz_pol_grid = np.clip(xyz_pol, min_bound, max_bound - 1e-3)

        grid_ind = (np.floor((xyz_pol_grid - min_bound) / intervals)).astype(np.int32) 
         
        grid_ind_vox_float = ((xyz_pol_grid - min_bound) / intervals_vox).astype(np.float32)

        # center data on each voxel for PTnet
        voxel_centers = (grid_ind.astype(np.float32) + 0.5) * intervals + min_bound 
        return_xyz = xyz_pol - voxel_centers

        feat = np.expand_dims(feat, axis=1)  

        return_feat = np.concatenate((return_xyz, xyz_pol, xyz[:, :2], feat), axis=1) 
        
        data_tuple = (grid_ind, labels, return_feat, grid_ind_vox_float, subcondition)

        return data_tuple

def seg_custom_collate_fn(data):

    grid_ind_list = [d[0] for d in data]  
    point_label_list = [d[1] for d in data]  
    point_feat_list = [d[2] for d in data]  
    grid_ind_vox_list = [d[3] for d in data]  
    subcondition_list = [d[4] for d in data]  
    
    return point_feat_list, grid_ind_list, point_label_list, grid_ind_vox_list, subcondition_list


def seg_custom_collate_fn_origin(data):
    grid_ind_stack = np.stack([d[0] for d in data]).astype(np.float32)
    point_label = np.stack([d[1] for d in data]).astype(np.int32)
    point_feat = np.stack([d[2] for d in data]).astype(np.float32)
    grid_ind_vox_stack = np.stack([d[3] for d in data]).astype(np.float32)
    
    return torch.from_numpy(point_feat), \
        torch.from_numpy(grid_ind_stack), \
        torch.from_numpy(point_label), \
        torch.from_numpy(grid_ind_vox_stack)



