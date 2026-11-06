import torch
import torch.nn as nn

class AutoF3CAligner(nn.Module):
    """
    为每个类别学习一个“软对齐”强度 gamma_shift[c]（映射到 [lo, hi]），
    并维护两套用于计算 delta=μ-μ' 的中心：
      1) 上一 epoch 的全局中心（prev_mu_*）——优先使用，抑制批内噪声；
      2) EMA 中心（mu_*）——作为备选/补充；
    若两者都不可用，apply_auto_f3c 会回退到“当前 batch 中心”。
    """
    def __init__(self, num_classes,
                 init_gamma_shift: float = 1.0,
                 clamp=(0.85, 1.15),
                 center_momentum: float = 0.9,
                 use_ema_center: bool = True,
                 use_prev_epoch_center: bool = True,
                 z_temperature: float = 0.5,
                 z_init_noise: float = 0.01
    ):
        super().__init__()
        lo, hi = float(clamp[0]), float(clamp[1])

        # 用 z 参数化 + sigmoid 映射到 [lo,hi]；初值即为 init_gamma_shift
        # 注意避免 logit(0/1) 溢出
        p0 = (min(max(init_gamma_shift, lo), hi) - lo) / (hi - lo)
        p0 = float(torch.clamp(torch.tensor(p0), 1e-4, 1 - 1e-4))
        z0 = torch.log(torch.tensor(p0 / (1 - p0)))

        # ★ 加一点随机扰动，打破对称（否则三类都卡在同一个点）
        self.z = nn.Parameter(z0.repeat(num_classes) + z_init_noise * torch.randn(num_classes))

        self.lo, self.hi = lo, hi
        self.z_temperature = float(z_temperature)  # ★ 温度越小，z=0 附近的梯度越大

        # -------- EMA 中心（不参与梯度）--------
        self.register_buffer("mu_orig", None)             # [C,D]
        self.register_buffer("mu_comp", None)             # [C,D]
        self.center_m = float(center_momentum)
        self.use_ema_center = bool(use_ema_center)

        # -------- 上一 epoch 的全局中心（不参与梯度）--------
        self.use_prev_epoch_center = bool(use_prev_epoch_center)
        self.register_buffer("last_epoch", torch.tensor(-1))
        self.register_buffer("acc_sums_orig", None)       # [C,D]
        self.register_buffer("acc_sums_comp", None)       # [C,D]
        self.register_buffer("acc_counts",    None)       # [C]
        self.register_buffer("prev_mu_orig",  None)       # [C,D]
        self.register_buffer("prev_mu_comp",  None)       # [C,D]


    def forward(self):
        # ★ 用温度缩放，增加 z=0 附近的梯度：d/dz sigmoid(z/T) = (1/T)*sigmoid*(1-sigmoid)
        s = torch.sigmoid(self.z / self.z_temperature)
        return self.lo + (self.hi - self.lo) * s

    # =====================  以下均为“中心”的维护接口  =====================

    @torch.no_grad()
    def _maybe_roll_epoch(self, now_epoch: int, num_classes: int, D: int, device, dtype):
        """
        当检测到进入了新的 epoch：
          - 把当前 epoch 的累计(acc_*)结算成 prev_mu_*（供下一 epoch 使用）；
          - 重置本轮累计器。
        """
        if self.last_epoch.item() == now_epoch:
            return

        # 结算上一轮累计 → prev_mu
        if self.acc_counts is not None and int(self.acc_counts.sum().item()) > 0:
            counts = self.acc_counts.clamp(min=1).view(-1, 1)  # [C,1]
            self.prev_mu_orig = self.acc_sums_orig / counts
            self.prev_mu_comp = self.acc_sums_comp / counts

        # 重置本轮累计器
        self.acc_sums_orig = torch.zeros(num_classes, D, device=device, dtype=dtype)
        self.acc_sums_comp = torch.zeros_like(self.acc_sums_orig)
        self.acc_counts    = torch.zeros(num_classes, device=device, dtype=torch.long)
        self.last_epoch    = torch.tensor(now_epoch, device=device)

    @torch.no_grad()
    def accumulate_epoch(self, label, feat_flat, comp_flat, num_classes):
        """
        在当前 epoch 内持续累积“原始/压缩”特征和计数，用于下一 epoch 作为固定中心。
        """
        N, D = feat_flat.size()
        if self.acc_sums_orig is None:
            self.acc_sums_orig = torch.zeros(num_classes, D, device=feat_flat.device, dtype=feat_flat.dtype)
            self.acc_sums_comp = torch.zeros_like(self.acc_sums_orig)
            self.acc_counts    = torch.zeros(num_classes, device=feat_flat.device, dtype=torch.long)

        self.acc_sums_orig.index_add_(0, label, feat_flat)
        self.acc_sums_comp.index_add_(0, label, comp_flat)
        self.acc_counts.index_add_(0, label, torch.ones_like(label, dtype=torch.long))

    @torch.no_grad()
    def update_centers(self, label, feat_flat, comp_flat, num_classes):
        """
        经典 EMA 更新，作为“上一 epoch 固定中心”的补充/备选。
        """
        N, D = feat_flat.size()
        if self.mu_orig is None:
            self.mu_orig = torch.zeros(num_classes, D, device=feat_flat.device, dtype=feat_flat.dtype)
            self.mu_comp = torch.zeros_like(self.mu_orig)

        counts = torch.bincount(label, minlength=num_classes)
        present = (counts > 0).nonzero(as_tuple=False).view(-1)
        if present.numel() == 0:
            return

        sums_o = torch.zeros_like(self.mu_orig); sums_o.index_add_(0, label, feat_flat)
        sums_c = torch.zeros_like(self.mu_comp); sums_c.index_add_(0, label, comp_flat)
        means_o = sums_o[present] / counts[present].float().unsqueeze(1)
        means_c = sums_c[present] / counts[present].float().unsqueeze(1)

        m = self.center_m
        self.mu_orig[present] = m * self.mu_orig[present] + (1 - m) * means_o
        self.mu_comp[present] = m * self.mu_comp[present] + (1 - m) * means_c

    @torch.no_grad()
    def get_delta_prev_epoch(self):
        if self.prev_mu_orig is None or self.prev_mu_comp is None:
            return None
        return (self.prev_mu_orig - self.prev_mu_comp)  # [C,D]

    @torch.no_grad()
    def get_delta_ema(self):
        if self.mu_orig is None or self.mu_comp is None:
            return None
        return (self.mu_orig - self.mu_comp)            # [C,D]


@torch.no_grad()
def _tau_per_class(num_classes, gamma, device, dtype):
    """
    等差缩放：tau[k] = 1 + gamma - k * (gamma / C)
    返回形状 [C]
    """
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    return 1.0 + gamma - classes * (gamma / float(num_classes))


def apply_auto_f3c(feature, label, num_classes, cfg, aligner: AutoF3CAligner, **kwargs):
    """
    修改点：START_EPOCH 只控制“等差缩放(FCC)”开始的 epoch。
           软对齐的 ramp 由 ALIGN_START_EPOCH（可选，默认=START_EPOCH）控制。
    """
    device, dtype = feature.device, feature.dtype
    N = feature.size(0)
    raw_shape = feature.shape
    now_epoch = int(kwargs.get("now_epoch", 0))

    # -------- 读取关键超参 --------
    start_fcc   = int(getattr(cfg.Auto_F3C, "START_EPOCH", 0))                  # 仅控制等差缩放开始
    align_start = int(getattr(cfg.Auto_F3C, "ALIGN_START_EPOCH", start_fcc))    # 软对齐开始（可选）
    ramp        = int(getattr(cfg.Auto_F3C, "RAMP_EPOCHS", 0))
    gamma       = float(getattr(cfg.Auto_F3C, "GAMMA", 0.0))

    # -------- 1) 等差缩放（FCC） --------
    feat_flat = feature.view(N, -1)                                            # [N, D]
    if now_epoch < start_fcc:
        # 未到 START_EPOCH：不做缩放
        comp_flat = feat_flat.clone()                                          # [N, D]
    else:
        # 开始做等差缩放
        tau_cls   = _tau_per_class(num_classes, gamma, device, dtype)          # [C]
        tau_batch = tau_cls[label].view(N, *([1] * (feature.ndim - 1)))        # [N,1,1,...]
        comp_flat = (feat_flat * tau_batch.view(N, 1))                         # [N, D]

    D = comp_flat.size(1)

    # -------- 2) 维护“上一 epoch 累积”，优先取全局 delta --------
    delta = None
    if getattr(cfg.Auto_F3C, "USE_PREV_EPOCH_CENTER", True):
        aligner._maybe_roll_epoch(now_epoch, num_classes, D, device, dtype)
        aligner.accumulate_epoch(label, feat_flat, comp_flat, num_classes)
        delta_prev = aligner.get_delta_prev_epoch()
        if delta_prev is not None:
            delta = delta_prev.detach()                                        # [C, D]

    # 退回到 EMA
    if delta is None and getattr(cfg.Auto_F3C, "USE_EMA_CENTER", True):
        aligner.update_centers(label, feat_flat, comp_flat, num_classes)
        delta_ema = aligner.get_delta_ema()
        if delta_ema is not None:
            delta = delta_ema.detach()

    # 再退回到 batch
    centers_comp_batch = None
    if delta is None:
        sums_o = torch.zeros(num_classes, D, device=device, dtype=dtype)
        sums_c = torch.zeros_like(sums_o)
        counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)
        sums_o.index_add_(0, label, feat_flat)
        sums_c.index_add_(0, label, comp_flat)
        centers_o = sums_o / counts.view(-1, 1)
        centers_c = sums_c / counts.view(-1, 1)
        delta = (centers_o - centers_c).detach()
        centers_comp_batch = centers_c

    # -------- 3) delta 归一化 / 系数 --------
    if getattr(cfg.Auto_F3C, "NORM_DELTA", True):
        delta = delta / (delta.norm(dim=1, keepdim=True) + 1e-6)
    delta = delta * float(getattr(cfg.Auto_F3C, "DELTA_COEF", 1.0))

    # -------- 4) 类级可学习强度 --------
    gamma_shift = aligner()                                                    # [C]
    shift_scale = gamma_shift[label].view(N, 1)                                # [N,1]

    count_beta = float(getattr(cfg.Auto_F3C, "COUNT_BETA", 0.0))
    if count_beta > 0.0:
        # 用“当前epoch累计计数”作为近似（与你的 prev/EMA 逻辑兼容）
        if aligner.acc_counts is not None:
            freq = aligner.acc_counts.float().clamp_min(1.0)   # [C]
        else:
            freq = torch.bincount(label, minlength=num_classes).float().to(device).clamp_min(1.0)
        gate_c = (freq / freq.max()).pow(count_beta)           # [C]∈(0,1]
        shift_scale = shift_scale * gate_c[label].view(N, 1)   # 叠乘类级门控

    # （可选）样本级距离闸门
    dist_beta = float(getattr(cfg.Auto_F3C, "DIST_BETA", 0.0))
    if dist_beta > 0.0:
        if getattr(cfg.Auto_F3C, "USE_EMA_CENTER", True) and \
           hasattr(aligner, "mu_comp") and aligner.mu_comp is not None:
            center_comp = aligner.mu_comp                                      # [C,D]
        else:
            if centers_comp_batch is None:
                sums_c = torch.zeros(num_classes, D, device=device, dtype=dtype)
                counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)
                sums_c.index_add_(0, label, comp_flat)
                centers_comp_batch = sums_c / counts.view(-1, 1)
            center_comp = centers_comp_batch
        with torch.no_grad():
            dist = (comp_flat - center_comp[label]).pow(2).sum(1).sqrt()       # [N]
            tmp = torch.zeros(num_classes, device=device); tmp.index_add_(0, label, dist)
            counts = torch.bincount(label, minlength=num_classes).clamp(min=1).to(device)
            mean_d = (tmp / counts.float()).clamp_min(1e-6)                    # [C]
            gate = torch.sigmoid(dist_beta * (mean_d[label] - dist)).view(N, 1)
        shift_scale = shift_scale * gate

    # -------- 5) 软对齐 ramp：从 ALIGN_START_EPOCH 开始放开 --------
    if now_epoch <= align_start:
        coef = 0.0
    else:
        coef = 1.0 if ramp <= 0 else min(1.0, (now_epoch - align_start) / float(ramp))

    # -------- 6) 应用并还原形状 --------
    comp_flat_aligned = comp_flat + coef * shift_scale * delta[label]          # [N,D]
    return comp_flat_aligned.view(raw_shape)