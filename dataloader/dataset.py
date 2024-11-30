import os
import numpy as np
from torch.utils.data import Dataset
import yaml
import glob
import random

class AdverseNet_Dataset_train(Dataset):
    def __init__(self, data_path, fix_sample, label_mapping="AdverseNet.yaml"):
        with open(label_mapping, 'r') as stream:
            AdverseNetyaml = yaml.safe_load(stream)

        self.learning_map_rf = AdverseNetyaml['learning_map_rf']
        self.learning_map_s = AdverseNetyaml['learning_map_s']
        self.data_path = data_path

        self.fix_sample = fix_sample
        self.conditions = ['rain', 'snow', 'fog']
        self.subconditions = {
            'rain': ['rain15', 'rain33', 'rain55'],
            'snow': ['light', 'medium', 'heavy'],
            'fog': ['foga', 'fogb', 'fogc']
        }

        self.file_paths_rain, self.file_paths_snow, self.file_paths_fog = self.prepare_file_paths()

    def prepare_file_paths(self):
        file_paths_rain, file_paths_snow, file_paths_fog = [], [], []
        for condition in self.conditions:
            for subcondition in self.subconditions[condition]:
                folder_path = os.path.join(self.data_path, condition, subcondition)
                sequence_folders = [os.path.join(folder_path, seq_folder) 
                                    for seq_folder in os.listdir(folder_path) 
                                    if os.path.isdir(os.path.join(folder_path, seq_folder))]
                
                all_files = []
                for seq_folder in sequence_folders:
                    files_in_seq = glob.glob(f"{seq_folder}/*.txt")
                    all_files.extend(files_in_seq)
                
                selected_files = random.sample(all_files, self.fix_sample) if len(all_files) > self.fix_sample else all_files
                
                if condition == 'rain':
                    file_paths_rain.extend(selected_files)
                elif condition == 'snow':
                    file_paths_snow.extend(selected_files)
                elif condition == 'fog':
                    file_paths_fog.extend(selected_files)
        return file_paths_rain, file_paths_snow, file_paths_fog

    def __len__(self):
        return len(self.file_paths_rain)

    def __getitem__(self, index):
        file_path_rain = self.file_paths_rain[index]
        file_path_snow = self.file_paths_snow[index]
        file_path_fog = self.file_paths_fog[index]

        data_tuple_rain = self.process_file(file_path_rain, 'rain')
        data_tuple_snow = self.process_file(file_path_snow, 'snow')
        data_tuple_fog = self.process_file(file_path_fog, 'fog')

        data_tuple = data_tuple_rain + data_tuple_snow + data_tuple_fog

        return data_tuple

    def process_file(self, file_path, condition):
        points = np.loadtxt(file_path, usecols=(0, 1, 2, 3))
        points[:, -1] /= 255
        labels = np.loadtxt(file_path, usecols=4, dtype=np.uint8)
        
        if condition in ['fog', 'rain']:
            mapped_labels = np.vectorize(self.learning_map_rf.get)(labels)
        else:  # 'snow'
            mapped_labels = np.vectorize(self.learning_map_s.get)(labels)
        
        mapped_labels = mapped_labels.reshape([-1, 1]).astype(np.uint8)
        data_tuple = (points, mapped_labels)

        return data_tuple


class AdverseNet_Dataset_eval(Dataset):
    def __init__(self, data_path, fix_sample, label_mapping="AdverseNet.yaml"):
        with open(label_mapping, 'r') as stream:
            AdverseNetyaml = yaml.safe_load(stream)

        self.learning_map_rf = AdverseNetyaml['learning_map_rf']
        self.learning_map_s = AdverseNetyaml['learning_map_s']
        self.data_path = data_path # /home/yxy/datasets/AdverseNet/val
        self.fix_sample = fix_sample
        self.conditions = ['rain', 'snow', 'fog']
        self.subconditions = {
            'rain': ['rain15', 'rain33', 'rain55'],
            'snow': ['light', 'medium', 'heavy'],
            'fog': ['foga', 'fogb', 'fogc']
        }
        self.file_paths = self.prepare_file_paths()

    def prepare_file_paths(self):
        file_paths = []
        for condition in self.conditions:
            for subcondition in self.subconditions[condition]:
                folder_path = os.path.join(self.data_path, condition, subcondition)
                sequence_folders = [os.path.join(folder_path, seq_folder) 
                                    for seq_folder in os.listdir(folder_path) 
                                    if os.path.isdir(os.path.join(folder_path, seq_folder))]
                
                all_files = []
                for seq_folder in sequence_folders:
                    files_in_seq = glob.glob(f"{seq_folder}/*.txt")
                    all_files.extend(files_in_seq)

                selected_files = random.sample(all_files, self.fix_sample) if len(all_files) > self.fix_sample else all_files
                for file_path in selected_files:
                    # file_paths.append((file_path, condition))
                    file_paths.append((file_path, condition, subcondition))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        # file_path, condition = self.file_paths[index]
        file_path, condition, subcondition = self.file_paths[index]  
        points = np.loadtxt(file_path, usecols=(0, 1, 2, 3))
        points[:, -1] /= 255
        labels = np.loadtxt(file_path, usecols=4, dtype=np.uint8)
        
        if condition in ['fog', 'rain']:
            mapped_labels = np.vectorize(self.learning_map_rf.get)(labels)
        else:  # 'snow'
            mapped_labels = np.vectorize(self.learning_map_s.get)(labels)

        mapped_labels = mapped_labels.reshape([-1, 1]).astype(np.uint8)

        data_tuple = (points, mapped_labels, subcondition)

        return data_tuple

def get_label_name(label_mapping):
    with open(label_mapping, 'r') as stream:
        data = yaml.safe_load(stream)
    labels = data['labels']
    
    return labels



