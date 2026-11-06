import os
import numpy as np
import torch
from torch.utils.data import Dataset
import h5py
import scipy.io as scio

def _ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def _create_or_load_fixed_per_class_split(
    labels_pos,                  # 已按 indices 重排后的标签, shape [N]
    per_class: int = 191,        # 每类验证集固定个数
    seed: int = 111,
    cache_dir: str = "./splits_lam",
    cache_prefix: str = "lam_posidx_fixedpc191",   # 与旧比例划分区分开
):
    """
    返回：train_pos_idx, val_pos_idx （都是 0..N-1 的位置索引）
    每个类别从 labels_pos 中随机不放回抽取 per_class 个样本作为验证集，其余为训练集。
    """
    _ensure_dir(cache_dir)
    f_train = os.path.join(cache_dir, f"{cache_prefix}_train.npy")
    f_val   = os.path.join(cache_dir, f"{cache_prefix}_val.npy")

    if os.path.exists(f_train) and os.path.exists(f_val):
        return np.load(f_train), np.load(f_val)

    rng = np.random.RandomState(seed)
    N = labels_pos.shape[0]
    all_pos = np.arange(N)

    val_pos_list = []
    classes = np.unique(labels_pos)
    # 检查每类是否足够 191
    for cls in classes:
        pos = np.where(labels_pos == cls)[0]
        if len(pos) < per_class:
            raise ValueError(f"类别 {int(cls)} 样本数 {len(pos)} < 每类验证需要的 {per_class}。请下调 per_class 或更换数据。")
        chosen = rng.choice(pos, size=per_class, replace=False)
        val_pos_list.append(chosen)

    val_pos_idx = np.sort(np.concatenate(val_pos_list, axis=0))
    train_pos_idx = np.setdiff1d(all_pos, val_pos_idx, assume_unique=True)

    # 缓存
    np.save(f_train, train_pos_idx)
    np.save(f_val,   val_pos_idx)
    return train_pos_idx, val_pos_idx


class LAMDataset(Dataset):
    """
    与原始 LAMDataset 用法保持一致，但验证集按“每类固定数量”来划分（可复现）。
    - 先按 indices 重排，再在 0..N-1 位置域做按类定额划分（并缓存）
    """
    def __init__(self,
                 mode="train",
                 cfg=None,
                 seed: int = 111,
                 # test_ratio 不再使用；保留参数名避免外部调用报错
                 test_ratio: float = 0.2,
                 split_cache_dir: str = "./splits_lam",
                 # 每类验证集定额（你要 191）
                 val_per_class: int = 191,
                 # 路径按你的实际情况
                 input_mat_path: str = r"I:\2024223020007\FCC\main\dataset\input32.mat",
                 label_mat_path: str = r"I:\2024223020007\FCC\main\dataset\label_ding_12_6.mat",
                 shuffle_idx_path: str = r"I:\2024223020007\FCC\main\dataset\sample_Shuffle_2866.npy",
                 ):
        super().__init__()
        self.mode = mode
        self.cfg = cfg
        self.input_size = cfg.INPUT_SIZE if cfg is not None else (32, 32)
        self.color_space = cfg.COLOR_SPACE if cfg is not None else "RGB"

        # 1) 读数据 + 重排到 indices 顺序
        indices = np.load(shuffle_idx_path)
        label_all = h5py.File(name=label_mat_path, mode='r')['label_ding'][()].squeeze()
        label_pos = label_all[indices]  # 已重排

        data = scio.loadmat(input_mat_path)
        input_all = data["data"]                          # (H, W, C, N)
        input_all = np.transpose(input_all, (3, 2, 1, 0)) # -> (N, C, H, W)
        input_pos = input_all[indices]                    # 与 label_pos 对齐

        # 2) 按类定额划分（每类固定 val_per_class 个样本进验证集）
        train_pos_idx, val_pos_idx = _create_or_load_fixed_per_class_split(
            labels_pos=label_pos,
            per_class=val_per_class,
            seed=seed,
            cache_dir=split_cache_dir,
            cache_prefix=f"lam_posidx_fixedpc{val_per_class}"
        )

        final_pos = train_pos_idx if self.mode == "train" else val_pos_idx

        # 3) 取数据
        self.input_data = torch.from_numpy(input_pos[final_pos]).float() / 255.0
        self.label = torch.from_numpy(label_pos[final_pos].astype(np.int64))

        # 4) 统计
        self.cls_num = int(np.unique(self.label.numpy()).size)
        self.class_dict = self._get_class_dict()
        self.pos_indices = final_pos
        # 额外打印验证集是否真的“每类=val_per_class”
        if self.mode != "train":
            _, vc = np.unique(self.label.numpy(), return_counts=True)
            print(f"[VAL] per-class counts: {vc.tolist()} (target {val_per_class})")
        print(f"{self.mode} Mode: Loaded {len(self.input_data)} samples with {self.cls_num} classes. (seed={seed})")

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        img = self.input_data[index]     # [C,H,W]
        label = int(self.label[index])
        meta = {"image_id": int(self.pos_indices[index])}
        return img, label, meta

    def get_num_classes(self):
        return self.cls_num

    def get_annotations(self):
        return [{'category_id': int(l.item())} for l in self.label]

    def _get_class_dict(self):
        class_dict = {}
        labels_np = self.label.numpy()
        for idx, lab in enumerate(labels_np):
            class_dict.setdefault(int(lab), []).append(idx)
        return class_dict

    def update(self, epoch):
        pass

    def _get_trans_image(self, img_idx):
        img = self.input_data[img_idx]
        return img.unsqueeze(0)