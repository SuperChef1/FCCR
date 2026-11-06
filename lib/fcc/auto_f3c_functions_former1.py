import torch
import torch.nn as nn

class AutoF3CAligner(nn.Module):
    """
    为每个类别维护一个可学习的缩放系数 gamma_shift[c]，
    用来调节“中心对齐位移”的强度（学出来的，不再和压缩 gamma 绑死）。
    """
    def __init__(self, num_classes, init_gamma_shift=1.0, clamp=None):
        super().__init__()
        self.gamma_shift = nn.Parameter(torch.full((num_classes,), float(init_gamma_shift)))
        self.clamp = clamp  # (min, max) or None

    def forward(self):
        if self.clamp is not None:
            with torch.no_grad():
                self.gamma_shift.clamp_(self.clamp[0], self.clamp[1])
        return self.gamma_shift

@torch.no_grad()
def _tau_per_class(num_classes, gamma, device, dtype):
    # 等差缩放：tau[k] = 1 + gamma - k * (gamma/C)
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    return 1.0 + gamma - classes * (gamma / float(num_classes))

def apply_auto_f3c(feature, label, num_classes, cfg, aligner: AutoF3CAligner):
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
    delta = centers_orig - centers_comp               # [C, D] 需要把簇中心拉回时的位移

    # 4) 学习到的对齐强度（每个类别一个标量）
    gamma_shift = aligner()                           # 可能会被 clamp
    shift_scale = gamma_shift[label].view(N, 1)       # [N, 1]

    # 5) 应用中心对齐（可学习强度）
    comp_flat_aligned = comp_flat + shift_scale * delta[label]  # [N, D]

    # 6) 还原形状
    return comp_flat_aligned.view(raw_shape)