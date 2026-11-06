import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio
import gc
from PIL import Image

class LAMDataset(Dataset):
    def __init__(self, mode="train", cfg=None):
        super(LAMDataset, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE
        self.color_space = cfg.COLOR_SPACE

        INPUT_DIR = r'I:\2024223020007\FCC\main\dataset\input32.mat'
        OUTPUT_DIR = r'I:\2024223020007\FCC\main\dataset\label_ding_2866_class3_try_2059_553_254.mat'
        indices = np.load(r'I:\2024223020007\FCC\main\dataset\sample_Shuffle_2866.npy')

        label = h5py.File(name=OUTPUT_DIR, mode='r')['label_ding'][()]
        label = label.squeeze()
        label = label[indices]

        split_ratio_test = 0.2

        indices_class0 = np.where(label == 0)[0]
        indices_class1 = np.where(label == 1)[0]
        indices_class2 = np.where(label == 2)[0]

        n_class0 = int(len(indices_class0) * split_ratio_test)
        n_class1 = int(len(indices_class1) * split_ratio_test)
        n_class2 = int(len(indices_class2) * split_ratio_test)

        selected_class0 = np.random.choice(indices_class0, n_class0, replace=False)
        selected_class1 = np.random.choice(indices_class1, n_class1, replace=False)
        selected_class2 = np.random.choice(indices_class2, n_class2, replace=False)

        indices_test = np.concatenate([selected_class0, selected_class1, selected_class2])
        indices_train = np.setdiff1d(indices, indices_test)

        if self.mode == "train":
            self.indices = indices_train
        else:
            self.indices = indices_test

        self.label = label[self.indices]
        self.label = torch.from_numpy(self.label.squeeze()).long()

        gc.collect()

        data = scio.loadmat(INPUT_DIR)
        input_data = data['data']
        input_data = np.transpose(input_data, (3, 2, 1, 0))
        input_data = input_data[indices]

        self.input_data = input_data[self.indices]
        self.input_data = torch.from_numpy(self.input_data).float() / 255.0
        # mean = self.input_data.mean(dim=(0, 2, 3), keepdim=True)
        # std = self.input_data.std(dim=(0, 2, 3), keepdim=True)
        # self.input_data = (self.input_data - mean) / std
        # print(f"Normalization done: mean={mean.flatten()}, std={std.flatten()}")

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