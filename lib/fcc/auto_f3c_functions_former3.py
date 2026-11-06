import torch
import torch.nn as nn

class AutoF3CAligner(nn.Module):
    def __init__(self, num_classes, init_gamma_shift=1.0, clamp=(0.85, 1.15),
                 center_momentum=0.9, use_ema_center=True):
        super().__init__()
        # 让初始化就等于 cfg 里想要的值（而不是固定 0.5）
        lo, hi = float(clamp[0]), float(clamp[1])
        p0 = (min(max(init_gamma_shift, lo), hi) - lo) / (hi - lo)
        z0 = torch.log(torch.tensor(p0/(1-p0)))
        self.z  = nn.Parameter(z0.repeat(num_classes))
        self.lo, self.hi = lo, hi

        # 运行中 EMA 中心（不参与梯度）
        self.register_buffer("mu_orig", None)  # [C, D]
        self.register_buffer("mu_comp", None)  # [C, D]
        self.center_m = float(center_momentum)
        self.use_ema_center = bool(use_ema_center)

    def forward(self):
        return self.lo + (self.hi - self.lo) * torch.sigmoid(self.z)

    @torch.no_grad()
    def update_centers(self, label, feat_flat, comp_flat, num_classes):
        N, D = feat_flat.size()
        if self.mu_orig is None:
            self.mu_orig = torch.zeros(num_classes, D, device=feat_flat.device, dtype=feat_flat.dtype)
            self.mu_comp = torch.zeros_like(self.mu_orig)

        counts = torch.bincount(label, minlength=num_classes)
        present = (counts > 0).nonzero(as_tuple=False).view(-1)
        if present.numel() == 0: return

        sums_o = torch.zeros_like(self.mu_orig); sums_o.index_add_(0, label, feat_flat)
        sums_c = torch.zeros_like(self.mu_comp); sums_c.index_add_(0, label, comp_flat)
        means_o = sums_o[present] / counts[present].float().unsqueeze(1)
        means_c = sums_c[present] / counts[present].float().unsqueeze(1)

        m = self.center_m
        self.mu_orig[present] = m * self.mu_orig[present] + (1 - m) * means_o
        self.mu_comp[present] = m * self.mu_comp[present] + (1 - m) * means_c

    @torch.no_grad()
    def get_delta_ema(self):
        if self.mu_orig is None or self.mu_comp is None:
            return None
        return (self.mu_orig - self.mu_comp)

@torch.no_grad()
def _tau_per_class(num_classes, gamma, device, dtype):
    # 等差缩放：tau[k] = 1 + gamma - k * (gamma/C)
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    return 1.0 + gamma - classes * (gamma / float(num_classes))

def apply_auto_f3c(feature, label, num_classes, cfg, aligner: AutoF3CAligner, **kwargs):
    """
    Auto-F3C: 等差缩放 + 可学习强度的“软中心对齐”
    - 优先用 EMA 全局中心计算 delta（若 aligner 实现了 update_centers/get_delta_ema）
    - 否则回退到 batch 内估计
    - 支持 NORM_DELTA / DELTA_COEF / RAMP / 距离闸门（DIST_BETA）
    """
    device, dtype = feature.device, feature.dtype
    N = feature.size(0)
    raw_shape = feature.shape

    # === 1) 等差缩放（FCC 部分） ===
    gamma = getattr(cfg.Auto_F3C, "GAMMA", getattr(cfg, "F3C", None).GAMMA if hasattr(cfg, "F3C") else 0.0)
    tau_cls = _tau_per_class(num_classes, gamma, device, dtype)  # [C]
    tau_batch = tau_cls[label].view(N, *([1] * (feature.ndim - 1)))  # [N,1,1,...]

    feat_flat = feature.view(N, -1)  # [N,D]
    comp_flat = (feat_flat * tau_batch.view(N, 1))  # [N,D]

    # === 2) （可选）更新 EMA 中心，并优先取 EMA 的 delta ===
    use_ema = getattr(cfg.Auto_F3C, "USE_EMA_CENTER", True)
    delta = None
    centers_comp_batch = None  # 供“距离闸门”回退使用

    # 如果 aligner 有 update_centers/get_delta_ema，就先更新再取 EMA delta
    if use_ema and hasattr(aligner, "update_centers") and hasattr(aligner, "get_delta_ema"):
        aligner.update_centers(label, feat_flat, comp_flat, num_classes)
        delta_ema = aligner.get_delta_ema()
        if delta_ema is not None:
            delta = delta_ema.detach()  # [C,D]（不让噪声反传）

    # === 3) 若没法用 EMA，则回退到 batch 估计 ===
    if delta is None:
        D = comp_flat.size(1)
        sums_orig = torch.zeros(num_classes, D, device=device, dtype=dtype)
        sums_comp = torch.zeros_like(sums_orig)
        counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)

        sums_orig.index_add_(0, label, feat_flat)  # Σx
        sums_comp.index_add_(0, label, comp_flat)  # Σx'
        centers_orig = sums_orig / counts.view(-1, 1)  # μ
        centers_comp = sums_comp / counts.view(-1, 1)  # μ'
        delta = (centers_orig - centers_comp).detach()  # [C,D]
        centers_comp_batch = centers_comp  # 记录下来供闸门用

    # === 4) delta 归一化 / 全局系数（由 cfg 控制） ===
    if getattr(cfg.Auto_F3C, "NORM_DELTA", True):
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-6)
    delta = delta * getattr(cfg.Auto_F3C, "DELTA_COEF", 1.0)

    # === 5) 可学习强度（每类一个标量），再按样本取对应类别 ===
    gamma_shift = aligner()  # [C]，来自 aligner.forward()
    shift_scale = gamma_shift[label].view(N, 1)  # [N,1]

    # === 6) （可选）样本级“距离闸门”：抑制离群点的对齐强度 ===
    dist_beta = float(getattr(cfg.Auto_F3C, "DIST_BETA", 0.0))
    if dist_beta > 0.0:
        # 选择用于计算距离的“压缩后中心”：优先 EMA 的 comp 中心，否则用 batch comp 中心
        if use_ema and hasattr(aligner, "mu_comp") and aligner.mu_comp is not None:
            center_comp = aligner.mu_comp
        else:
            # 若未提前计算 batch centers_comp，则现算一次
            if centers_comp_batch is None:
                D = comp_flat.size(1)
                sums_comp = torch.zeros(num_classes, D, device=device, dtype=dtype)
                counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)
                sums_comp.index_add_(0, label, comp_flat)
                centers_comp_batch = sums_comp / counts.view(-1, 1)
            center_comp = centers_comp_batch

        with torch.no_grad():
            # 每个样本到其类别“压缩后中心”的距离
            dist = (comp_flat - center_comp[label]).pow(2).sum(1).sqrt()  # [N]
            # 计算“类内平均距离”，用于标准化
            sums_d = torch.zeros(num_classes, device=device)
            counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)
            tmp = torch.zeros(num_classes, device=device)
            tmp.index_add_(0, label, dist)
            mean_d = (tmp / counts.float()).clamp_min(1e-6)  # [C]
            # gate：距离越大（可能离群）→ gate 越小；距离越小 → gate 越大
            gate = torch.sigmoid(dist_beta * (mean_d[label] - dist)).view(N, 1)  # [N,1]
        shift_scale = shift_scale * gate

    # === 7) RAMP：逐步放开中心对齐强度 ===
    now_epoch = kwargs.get("now_epoch", 0)
    start = int(getattr(cfg.Auto_F3C, "START_EPOCH", 0))
    ramp = int(getattr(cfg.Auto_F3C, "RAMP_EPOCHS", 0))
    if now_epoch <= start:
        coef = 0.0
    else:
        coef = 1.0 if ramp <= 0 else min(1.0, (now_epoch - start) / float(ramp))

    # === 8) 应用“软对齐”并还原形状 ===
    comp_flat_aligned = comp_flat + coef * shift_scale * delta[label]  # [N,D]
    return comp_flat_aligned.view(raw_shape)