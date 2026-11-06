import torch
import torch.nn as nn

class AutoF3CAligner(nn.Module):
    def __init__(self, num_classes, init_gamma_shift=1.0, clamp=(0.85, 1.15)):
        super().__init__()
        # 用 z 参数化，初值 z=0 => gamma=中点≈1.0
        self.z = nn.Parameter(torch.zeros(num_classes))
        self.lo, self.hi = float(clamp[0]), float(clamp[1])

    def forward(self):
        # 平滑映射到 [lo, hi]，不会像 clamp_ 那样把梯度“卡死”
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.z)

@torch.no_grad()
def _tau_per_class(num_classes, gamma, device, dtype):
    # 等差缩放：tau[k] = 1 + gamma - k * (gamma/C)
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    return 1.0 + gamma - classes * (gamma / float(num_classes))

def apply_auto_f3c(feature, label, num_classes, cfg, aligner: AutoF3CAligner, **kwargs):
    """
    先按 tau 做等差缩放，再做中心对齐，但中心对齐的强度乘以可学习的 gamma_shift[label]。
    不改框架：直接在 fcc 路径里调用即可。
    """
    device, dtype = feature.device, feature.dtype
    N = feature.size(0)
    raw_shape = feature.shape

    # 1) 计算等差缩放因子 tau（来自 cfg.F3C.GAMMA，仍由你人工控制）
    tau_cls = _tau_per_class(num_classes, cfg.Auto_F3C.GAMMA, device, dtype)  # [C]
    tau_batch = tau_cls[label].view(N, *([1] * (feature.ndim - 1)))       # [N,1,1,...]

    # 2) 先做缩放（“被动压缩”）
    feat_flat = feature.view(N, -1)                  # [N, D]
    comp_flat = (feat_flat * tau_batch.view(N, 1))   # [N, D]

    # 3) 计算 batch 内各类原始/压缩中心
    D = comp_flat.size(1)
    sums_orig = torch.zeros(num_classes, D, device=device, dtype=dtype)
    sums_comp = torch.zeros_like(sums_orig)
    # bincount 最少为 1，避免 /0；没出现的类位移会被乘 0 个样本，不影响当前 batch
    counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device=device)

    sums_orig.index_add_(0, label, feat_flat)
    sums_comp.index_add_(0, label, comp_flat)

    centers_orig = sums_orig / counts.view(-1, 1)    # μ
    centers_comp = sums_comp / counts.view(-1, 1)    # μ'
    delta = (centers_orig - centers_comp).detach()               # [C, D] 需要把簇中心拉回时的位移

    # 规范化/缩放（可由 cfg 控制）
    if getattr(cfg.Auto_F3C, "NORM_DELTA", True):
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-6)
    delta = delta * getattr(cfg.Auto_F3C, "DELTA_COEF", 1.0)

    gamma_shift = aligner()  # 平滑映射后的值 ∈ [lo, hi]
    shift_scale = gamma_shift[label].view(N, 1)

    # 逐步放开强度（可选）
    now_epoch = kwargs.get("now_epoch", 0)
    start = cfg.Auto_F3C.START_EPOCH
    ramp  = getattr(cfg.Auto_F3C, "RAMP_EPOCHS", 0)
    if now_epoch <= start:
        coef = 0.0
    else:
        coef = 1.0 if ramp <= 0 else min(1.0, (now_epoch - start) / float(ramp))

    comp_flat_aligned = comp_flat + coef * shift_scale * delta[label]
    return comp_flat_aligned.view(raw_shape)