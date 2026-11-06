import torch
import numpy as np
import h5py
# 假设你已经有训练集标签(仅train)的一维数组或列表 labels_train，取值 0..C-1
# 例如：labels_train = np.load("train_labels.npy")  # 自行替换
label_mat_path: str = r"I:\2024223020007\FCC\main\dataset\label_ding_12_6.mat"
labels_train =h5py.File(name=label_mat_path, mode='r')['label_ding'][()].squeeze()


C = int(labels_train.max() + 1)
counts = torch.bincount(torch.from_numpy(labels_train).long(), minlength=C).float()

print("class counts:", counts.tolist())
torch.save(counts, "lam_image_count.pkl")