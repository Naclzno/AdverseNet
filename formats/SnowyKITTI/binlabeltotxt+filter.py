import numpy as np
import os

def convert_dataset_to_txt(snowy_kitti_dir, output_root_dir):
    dataset_dir = os.path.join(snowy_kitti_dir, 'dataset', 'sequences')

    for sequence in os.listdir(dataset_dir):
        sequence_dir = os.path.join(dataset_dir, sequence)
        snow_labels_dir = os.path.join(sequence_dir, 'snow_labels')
        snow_velodyne_dir = os.path.join(sequence_dir, 'snow_velodyne')

        bin_files = os.listdir(snow_velodyne_dir)
        label_files = os.listdir(snow_labels_dir)
        if len(bin_files) != len(label_files):
            print(f"警告: 序列{sequence}中的.bin文件与.label文件数量不匹配。")
            continue

        output_sequence_dir = os.path.join(output_root_dir, 'dataset', 'sequences', sequence)
        os.makedirs(output_sequence_dir, exist_ok=True)

        for bin_file in bin_files:
            bin_filename = os.path.join(snow_velodyne_dir, bin_file)
            label_filename = os.path.join(snow_labels_dir, bin_file.replace('.bin', '.label'))

            scan = np.fromfile(bin_filename, dtype=np.float32)
            scan = scan.reshape((-1, 4))

            label = np.fromfile(label_filename, dtype=np.int32)
            label = label.reshape((-1))

            sem_label = label & 0xFFFF
 
            mask = ~((scan == [-1, -1, -1, -1]).all(axis=1) & (sem_label == 0))
            filtered_data = scan[mask]
            filtered_labels = sem_label[mask]

            combined_data = np.hstack((filtered_data, filtered_labels[:, np.newaxis]))

            txt_filename = os.path.join(output_sequence_dir, bin_file.replace('.bin', '.txt'))
            np.savetxt(txt_filename, combined_data, fmt='%f %f %f %f %d')

    print("数据集转换完成。")

snowy_kitti_dir = 'Z:/数据集/snowyKITTI'
output_root_dir = 'Z:/datasets/filter/snowyKITTI'
convert_dataset_to_txt(snowy_kitti_dir, output_root_dir)
