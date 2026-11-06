import torch

def fccr(feature, label, num_classes, cfg, **kwargs):
    """
    Feature Clusters Compression (FCCR) with center alignment.
    - 先按策略压缩（放大）特征；
    - 再计算“原始 vs 压缩后”的簇中心差，并把压缩特征沿该方向整体平移，避免簇整体偏移。
    """
    c_type = cfg.FCCR.C_TYPE
    if c_type == 'edc':
        new_features = equal_diff_compress_with_center_align(
            feature, label, num_classes, cfg.FCCR.GAMMA
        )
    else:
        raise Exception('Error compression type.')
    return new_features


def equal_diff_compress_with_center_align(feature, label, num_classes, gamma):
    """
    等差压缩 + 簇中心对齐
    feature: [N, ...] 任意维度，压缩与对齐都基于展平后的通道维进行。
    """
    device = feature.device
    dtype  = feature.dtype
    N      = feature.shape[0]
    raw_shape = feature.shape

    # 1) 计算每类的缩放因子 tau_k（等差策略）
    #    tau[k] = 1 + gamma - k * (gamma / num_classes)
    classes = torch.arange(num_classes, device=device, dtype=dtype)
    tau_per_class = 1.0 + gamma - classes * (gamma / float(num_classes))  # [C]

    # 2) 取出本 batch 样本对应的 tau 并进行逐样本缩放（等价于“线性放大”，从而让原始特征被动“压缩”）
    tau_batch = tau_per_class[label].view(N, *([1] * (feature.ndim - 1)))  # [N,1,1,...]
    feat_flat = feature.view(N, -1)        # [N, D]
    comp_flat = (feat_flat * tau_batch.view(N, 1))  # 压缩后的（实际是放大后的）特征

    # 3) 计算 batch 内每个“出现过的类别”的原始/压缩簇中心
    D = comp_flat.size(1)
    # 按类别求和
    sums_orig = torch.zeros(num_classes, D, device=device, dtype=dtype)
    sums_comp = torch.zeros_like(sums_orig)
    counts = torch.bincount(label, minlength=num_classes).to(device=device).clamp(min=1)  # 防除零

    sums_orig.index_add_(0, label, feat_flat)  # Σ x
    sums_comp.index_add_(0, label, comp_flat)  # Σ x'

    centers_orig = sums_orig / counts.view(-1, 1)  # μ
    centers_comp = sums_comp / counts.view(-1, 1)  # μ'

    # 4) 平移量 = μ - μ'（把压缩后的簇中心拉回原始中心）
    delta = centers_orig - centers_comp  # [C, D]

    # 5) 对每个样本，按其标签加上对应类别的平移量
    comp_flat_aligned = comp_flat + delta[label]  # [N, D]

    # 6) 还原形状
    new_features = comp_flat_aligned.view(raw_shape)
    return new_features