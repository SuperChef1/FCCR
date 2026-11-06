import os
import random
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

def setup_repro(seed: int = 0):
    """
    全局可复现设置：必须在任何 torch/cuda 初始化 & 数据集/模型创建之前调用。
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 保证确定性
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 关掉 TF32，避免数值路径差异
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def _seed_worker(worker_id: int, base_seed: int):
    """
    DataLoader 的 worker 初始化函数。
    注意：不要用 torch.initial_seed() 去派生，因为它受 DataLoader 的内部播种影响。
    我们直接用用户给的 base_seed 做可控的偏移。
    """
    s = base_seed + worker_id
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def make_seeded_loader(
    dataset,
    batch_size: int,
    num_workers: int = 0,
    shuffle: bool = False,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int = 0,
    persistent_workers: bool = False,  # 追求最严谨复现建议 False
):
    """
    返回带固定随机性的 DataLoader。
    用法：
        train_loader = make_seeded_loader(train_set, cfg.TRAIN.BATCH_SIZE, seed=SEED, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS)
        valid_loader = make_seeded_loader(valid_set, cfg.TEST.BATCH_SIZE, seed=SEED, shuffle=False, num_workers=cfg.TEST.NUM_WORKERS)
    """
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    worker_init = partial(_seed_worker, base_seed=seed)

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        worker_init_fn=worker_init,
        generator=g,
        persistent_workers=persistent_workers,
    )