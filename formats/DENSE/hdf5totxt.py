import h5py
import numpy as np
import os

def convert_hdf5_to_txt(input_dir, output_root):
    """
    遍历指定的输入目录, 读取.hdf5文件中的点云数据, 并将其转换为txt格式保存到输出目录。
    """
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".hdf5"):
                hdf5_path = os.path.join(root, file)
                output_path = root.replace(input_dir, output_root)
                os.makedirs(output_path, exist_ok=True)
                txt_filename = os.path.splitext(file)[0] + ".txt"
                txt_path = os.path.join(output_path, txt_filename)
                
                with h5py.File(hdf5_path, "r", driver='core') as hdf5:
                    # 假设sensorX, sensorY, sensorZ, intensity, label都是二维数组
                    # 使用ravel方法将它们平铺为一维数组
                    # 假设我们有一个二维数组：array_2d = np.array([[1, 2, 3], [4, 5, 6]]), 调用ravel()
                    # flattened_array = array_2d.ravel()
                    # 返回的 flattened_array 将会是：[1 2 3 4 5 6]
                    sensorX = hdf5.get('sensorX_1')[()].ravel()
                    sensorY = hdf5.get('sensorY_1')[()].ravel()
                    sensorZ = hdf5.get('sensorZ_1')[()].ravel()
                    intensity = hdf5.get('intensity_1')[()].ravel()
                    label = hdf5.get('labels_1')[()].ravel()

                    # 使用np.vstack将这些一维数组组织成n行5列的格式，并转置
                    data = np.vstack((sensorX, sensorY, sensorZ, intensity, label)).T
                    
                    # 保存为txt文件
                    np.savetxt(txt_path, data, fmt='%f')

    print("转换完成。")

# 使用示例
input_dir = "Z:/datasets/cnn_denoising_sunnynet"
output_root = "Z:/datasets/cnn_denoising_sunnynet_change"
convert_hdf5_to_txt(input_dir, output_root)
