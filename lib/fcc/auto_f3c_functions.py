import torch
import torch.nn as nn

class AutoF3CAligner(nn.Module):
    """
    极简版：仅学习每类的对齐强度 gamma_shift[c] ∈ [lo, hi]
    """
    def __init__(self, num_classes,
                 init_gamma_shift: float = 1.0,
                 clamp=(0.85, 1.15),
                 z_init_noise: float = 0.0):
        super().__init__()
        lo, hi = float(clamp[0]), float(clamp[1])

        # 用 z + sigmoid 映射到 [lo, hi]；初值即 init_gamma_shift
        p0 = (min(max(init_gamma_shift, lo), hi) - lo) / (hi - lo)
        p0 = float(torch.clamp(torch.tensor(p0), 1e-4, 1 - 1e-4))
        z0 = torch.log(torch.tensor(p0 / (1 - p0)))

        if z_init_noise and z_init_noise > 0:
            self.z = nn.Parameter(z0.repeat(num_classes) + z_init_noise * torch.randn(num_classes))
        else:
            self.z = nn.Parameter(z0.repeat(num_classes))

        self.lo, self.hi = lo, hi

    def forward(self):
        s = torch.sigmoid(self.z)               # (C,)
        return self.lo + (self.hi - self.lo) * s


@torch.no_grad()
def _tau_per_class(num_classes, gamma, device, dtype):
    """
    等差缩放：tau[k] = 1 + gamma - k * (gamma / C)  -> 形状 [C]
    """
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    return 1.0 + gamma - classes * (gamma / float(num_classes))


def apply_auto_f3c(feature, label, num_classes, cfg, aligner: AutoF3CAligner, **kwargs):
    """
    极简 Auto-F3C（batch 版）：
      - 从 START_EPOCH 同时开始：等差缩放 + soft 对齐（类级可学习强度）
      - delta 只用当前 batch 估计
      - 可选 NORM_DELTA / DELTA_COEF
    """
    device, dtype = feature.device, feature.dtype
    N = feature.size(0)
    raw_shape = feature.shape
    now_epoch = int(kwargs.get("now_epoch", 0))

    start_epoch = int(getattr(cfg.Auto_F3C, "START_EPOCH", 0))
    gamma       = float(getattr(cfg.Auto_F3C, "GAMMA", 0.0))
    norm_delta  = bool(getattr(cfg.Auto_F3C, "NORM_DELTA", True))
    delta_coef  = float(getattr(cfg.Auto_F3C, "DELTA_COEF", 1.0))

    # 未到 START_EPOCH：不做任何事
    if now_epoch < start_epoch:
        return feature

    # 1) 等差缩放（FCC）
    feat_flat = feature.view(N, -1)  # [N, D]
    tau_cls   = _tau_per_class(num_classes, gamma, device, dtype)              # [C]
    tau_batch = tau_cls[label].view(N, *([1] * (feature.ndim - 1)))            # [N,1,1,...]
    comp_flat = (feat_flat * tau_batch.view(N, 1))                             # [N, D]

    # 2) batch 内原始/压缩中心
    D = comp_flat.size(1)
    sums_o = torch.zeros(num_classes, D, device=device, dtype=dtype)
    sums_c = torch.zeros_like(sums_o)
    counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)

    sums_o.index_add_(0, label, feat_flat)
    sums_c.index_add_(0, label, comp_flat)
    centers_o = sums_o / counts.view(-1, 1)    # μ
    centers_c = sums_c / counts.view(-1, 1)    # μ'

    delta = (centers_o - centers_c)            # [C, D]

    # 3) 可选归一化/缩放
    if norm_delta:
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-6)
    if delta_coef != 1.0:
        delta = delta * delta_coef

    # 4) 类级可学习强度（soft 对齐）
    gamma_shift = aligner()                           # [C]
    shift_scale = gamma_shift[label].view(N, 1)       # [N, 1]
    comp_flat_aligned = comp_flat + shift_scale * delta[label]  # [N, D]

    # 5) 还原形状
    return comp_flat_aligned.view(raw_shape)