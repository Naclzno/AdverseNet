import numpy as np
import os

def filter_and_process_data(file_path):
    data = np.loadtxt(file_path, dtype=np.float32)
    invalid_condition = (np.all(data[:, :4] == 0, axis=1) & ((data[:, 4] == 100) | (data[:, 4] == 0)))
    valid_data = data[~invalid_condition]
    valid_data[:, 3] = np.round(valid_data[:, 3] * 255)
    return valid_data


def process_sequence(seq_path, output_seq_path, condition=lambda file: True):
    for file in sorted(os.listdir(seq_path)):
        if condition(file):
            file_path = os.path.join(seq_path, file)
            valid_data = filter_and_process_data(file_path)
            np.savetxt(os.path.join(output_seq_path, file), valid_data, fmt='%f')

def process_point_cloud(input_dir, output_dir):
    for subfolder in ['train_01', 'train_02', 'val_01', 'test_01']:
        subfolder_path = os.path.join(input_dir, subfolder)
        for seq in os.listdir(subfolder_path):
            seq_path = os.path.join(subfolder_path, seq)
            output_seq_path = os.path.join(output_dir, subfolder, seq)
            os.makedirs(output_seq_path, exist_ok=True)
            if not seq.endswith('Clear'):
                process_sequence(seq_path, output_seq_path)
            else:
                process_sequence(seq_path, output_seq_path, lambda file: not file.endswith('_2.txt'))

input_dir = "F:/datasets/txt/cnn_denoising_raw"
output_dir = "F:/datasets/txt/cnn_denoising"
process_point_cloud(input_dir, output_dir)
