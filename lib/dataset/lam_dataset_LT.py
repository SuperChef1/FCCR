import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import gc
from PIL import Image

class LAMDataset_LT(Dataset):
    def __init__(self, mode="train", cfg=None):
        super(LAMDataset_LT, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE

        INPUT_DIR = r'I:\2024223020007\FCC\main\dataset\input32.mat'
        OUTPUT_DIR = r'I:\2024223020007\FCC\main\dataset\label.mat'
        indices_all = np.load(r'I:\2024223020007\FCC\main\dataset\sample_Shuffle_2866.npy')

        label = h5py.File(name=OUTPUT_DIR, mode='r')['label_ding'][()]
        label = label.squeeze()
        label = label[indices_all]  # 对应打乱后的顺序

        # ===== 构造初始 index-label 对 =====
        all_idx = np.arange(len(label))
        indices_class0 = all_idx[label == 0]
        indices_class1 = all_idx[label == 1]
        indices_class2 = all_idx[label == 2]

        # ===== 模拟长尾分布：删除部分类 1 和类 2 样本 =====
        drop_1 = np.random.choice(indices_class1, size=300, replace=False)
        drop_2 = np.random.choice(indices_class2, size=500, replace=False)
        keep_mask = np.ones(len(label), dtype=bool)
        keep_mask[drop_1] = False
        keep_mask[drop_2] = False

        label = label[keep_mask]
        indices_all = indices_all[keep_mask]  # 同步打乱后的原始索引

        # ===== 重新获取每类索引（基于删减后的 label）=====
        indices_class0 = np.where(label == 0)[0]
        indices_class1 = np.where(label == 1)[0]
        indices_class2 = np.where(label == 2)[0]

        # ===== 分层划分测试集 =====
        split_ratio_test = 0.2
        n0 = int(len(indices_class0) * split_ratio_test)
        n1 = int(len(indices_class1) * split_ratio_test)
        n2 = int(len(indices_class2) * split_ratio_test)

        test_idx = np.concatenate([
            np.random.choice(indices_class0, n0, replace=False),
            np.random.choice(indices_class1, n1, replace=False),
            np.random.choice(indices_class2, n2, replace=False)
        ])
        train_idx = np.setdiff1d(np.arange(len(label)), test_idx)

        # ===== 最终选择索引 =====
        if self.mode == "train":
            final_idx = train_idx
        else:
            final_idx = test_idx

        self.label = torch.from_numpy(label[final_idx]).long()

        # ===== 读取图像数据并打乱顺序 =====
        data = scio.loadmat(INPUT_DIR)
        input_data = data['data']  # shape: (H, W, C, N)
        input_data = np.transpose(input_data, (3, 2, 1, 0))  # (N, C, H, W)
        input_data = input_data[indices_all]  # 保证与label一致

        self.input_data = torch.from_numpy(input_data[final_idx]).float() / 255.0

        self.cls_num = len(np.unique(self.label.numpy()))
        self.class_dict = self._get_class_dict()

        print(f"{self.mode} Mode: Loaded {len(self.input_data)} samples with {self.cls_num} classes.")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        img = self.input_data[index]  # (C, H, W) or (H, W, C)
        label = self.label[index].item()
        meta = {}

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img).float()

            # 确保为 [C, H, W] 格式，不再乱 permute
        if img.ndim == 3 and img.shape[0] in [1, 3, 150]:
            return img, label, meta
        else:
            raise ValueError(f"Unexpected image shape: {img.shape}")


    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        annos = [{'category_id': int(l.item())} for l in self.label]
        return annos

    def _get_class_dict(self):
        class_dict = {}
        labels_np = self.label.numpy()
        for idx, label in enumerate(labels_np):
            if label not in class_dict:
                class_dict[label] = []
            class_dict[label].append(idx)
        return class_dict

    def update(self, epoch):
        # 可用于 progressive sampler、DRS 更新
        pass

    def _get_trans_image(self, img_idx):
        img = self.input_data[img_idx]
        if isinstance(img, torch.Tensor):
            if img.ndim == 3 and img.shape[0] != 3:
                img = img.permute(2, 0, 1)
        else:
            img = torch.from_numpy(img).float()
            img = img.permute(2, 0, 1)
        return img.unsqueeze(0)  # (1, C, H, W)
