import torch
import torch.nn as nn
import torch.nn.functional as F

def get_detection_weight(n: torch.Tensor):
    n = n.clamp_min(1.0)
    a = (n.sum() - n) / n
    w = a * torch.log((1 + a) / a)
    return w[None]  # [1,C]

class EffectiveClassMarginLoss(nn.Module):
    """
    稳定版 ECM（不带背景列）：
    - 兼容 para_dict / count_file / counts 三种传参方式；
    - 用 softplus 避免 exp/log 溢出；
    - 自动做 clamp_min 防 0，检查维度一致性。
    """
    def __init__(self,
                 loss_weight: float = 1.0,
                 num_classes: int = 3,
                 count_file: str = 'I:/2024223020007/FCC/main/lam_image_count.pkl',
                 counts: torch.Tensor = None,
                 **kwargs):
        """
        也可通过 para_dict 传：
        para_dict = {
            'count_file': '...pkl',   # 或
            'counts': <1D list/ndarray/tensor of length C>
        }
        """
        super().__init__()
        self.loss_weight = loss_weight
        self.num_classes = num_classes

        # ----- 兼容 para_dict -----
        para_dict = kwargs.pop('para_dict', None)
        if para_dict is not None:
            # 若没显式给，就从 para_dict 里取
            count_file = count_file or para_dict.get('count_file', None)
            counts     = counts     if counts is not None else para_dict.get('counts', None)

        # ----- 读取类别计数 -----
        if counts is not None:
            n = torch.as_tensor(counts, dtype=torch.float32)
        elif count_file is not None:
            n = torch.tensor(torch.load(count_file)).float()
        else:
            raise ValueError(
                "[ECM] 需要提供训练集每类计数：count_file 或 counts（二者其一）。"
            )

        n = n.clamp_min(1.0)  # 防 0
        if n.numel() != num_classes:
            raise RuntimeError(
                f"[ECM] 计数长度 {n.numel()} 与 num_classes={num_classes} 不一致。"
            )

        self.register_buffer('sample_n', n)                           # [C]
        w = get_detection_weight(n)  # [1,C]
        w = w / (w.mean() + 1e-12)  # 归一化
        w = torch.clamp(w, min=0.1)  # 加下限，防止接近0
        self.register_buffer('detection_cls_weight', w)

    def forward(self, cls_score: torch.Tensor, labels: torch.Tensor, **kwargs):
        """
        cls_score: [B, C] logits（无背景列）
        labels:    [B]    in [0..C-1]
        """
        device = cls_score.device
        B, C   = cls_score.shape

        if self.sample_n.numel() != C:
            raise RuntimeError(
                f"[ECM] 计数长度({self.sample_n.numel()})与 logits 维度({C})不一致。"
            )

        target = torch.zeros(B, C, device=device, dtype=cls_score.dtype)
        target[torch.arange(B, device=device), labels] = 1.0

        pos_w, neg_w = self.compute_weight(device=device)  # [1,C]
        pos_w = pos_w.expand(B, C)
        neg_w = neg_w.expand(B, C)

        # 稳定形式
        z_pos = cls_score + pos_w          # 正项 logits
        z_neg = -cls_score + neg_w         # 负项 logits

        pos_term = -F.softplus(-z_pos) * target          # log(sigmoid(z_pos))
        neg_term = -F.softplus( z_neg)  * (1.0 - target) # log(sigmoid(-z_pos)) 的等价式

        cls_weight = self.detection_cls_weight.to(device)  # [1,C]
        loss = (-(pos_term + neg_term) * cls_weight).mean()

        return loss * self.loss_weight
    def update(self, epoch: int):
        # 目前不需要随 epoch 更新的话，留空即可
        return
    def compute_weight(self, device):
        n_pos = self.sample_n.to(device).clamp_min(1.0)  # [C]
        n_sum = n_pos.sum()
        n_neg = (n_sum - n_pos).clamp_min(1.0)

        n_pos_q = n_pos.pow(0.25)
        n_neg_q = n_neg.pow(0.25)
        denom   = (n_pos_q + n_neg_q).clamp_min(1e-12)

        pos_w = torch.log(denom / n_neg_q.clamp_min(1e-12))  # [C]
        neg_w = torch.log(denom / n_pos_q.clamp_min(1e-12))  # [C]
        return pos_w[None, :], neg_w[None, :]