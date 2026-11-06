import math
import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ["densenet_backbone"]

# ---- DenseNet 组件 ----
class _BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop = drop_rate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop > 0:
            out = nn.functional.dropout(out, p=self.drop, training=self.training)
        return torch.cat([x, out], dim=1)

class _BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super().__init__()
        inter = out_planes * 4
        self.bn1  = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter, kernel_size=1, bias=False)
        self.bn2  = nn.BatchNorm2d(inter)
        self.conv2 = nn.Conv2d(inter, out_planes, kernel_size=3, padding=1, bias=False)
        self.drop = drop_rate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop > 0:
            out = nn.functional.dropout(out, p=self.drop, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop > 0:
            out = nn.functional.dropout(out, p=self.drop, training=self.training)
        return torch.cat([x, out], dim=1)

class _Transition(nn.Module):
    def __init__(self, in_planes, out_planes, drop_rate=0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.drop = drop_rate
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.drop > 0:
            out = nn.functional.dropout(out, p=self.drop, training=self.training)
        return self.pool(out)

class _DenseBlock(nn.Module):
    def __init__(self, n_layers, in_planes, growth, block_cls, drop_rate=0.0):
        super().__init__()
        layers = []
        n_layers = int(n_layers)
        for i in range(n_layers):
            layers.append(block_cls(in_planes + i * growth, growth, drop_rate))
        self.block = nn.Sequential(*layers)
    def forward(self, x):
        return self.block(x)

# ---- 主干 ----
class DenseNetBackbone(nn.Module):
    """
    DenseNet 主干（不含 GAP / FC）
    输入: B x C x 32 x 32 （C=cfg.BACKBONE.IN_CHANNELS）
    输出: B x out_channels x H x W  （默认 depth=100, growth=12 时 out_channels=342, H=W=8）
    """
    def __init__(self, cfg,
                 pretrain=True, pretrained_model="", last_layer_stride=1):
        super().__init__()
        bb = getattr(cfg, "BACKBONE", cfg)

        in_ch       = int(getattr(bb, "IN_CHANNELS", 3))
        depth       = int(getattr(bb, "DEPTH", 100))
        growth      = int(getattr(bb, "GROWTH_RATE", 12))
        reduction   = float(getattr(bb, "REDUCTION", 0.5))
        bottleneck  = bool(getattr(bb, "BOTTLENECK", True))
        drop_rate   = float(getattr(bb, "DROP_RATE", 0.0))

        # CIFAR 风格 DenseNet：depth = 6n + 4（含 bottleneck 时每块 n/2 个层单元）
        n = (depth - 4) / 3
        block_cls = _BottleneckBlock if bottleneck else _BasicBlock
        if bottleneck: n = n / 2

        # stem
        in_planes = 2 * growth
        self.conv1 = nn.Conv2d(in_ch, in_planes, kernel_size=3, stride=1, padding=1, bias=False)

        # block1 -> trans1
        self.block1 = _DenseBlock(n, in_planes, growth, block_cls, drop_rate)
        in_planes = int(in_planes + n * growth)
        out_planes = int(math.floor(in_planes * reduction))
        self.trans1 = _Transition(in_planes, out_planes, drop_rate)
        in_planes = out_planes

        # block2 -> trans2
        self.block2 = _DenseBlock(n, in_planes, growth, block_cls, drop_rate)
        in_planes = int(in_planes + n * growth)
        out_planes = int(math.floor(in_planes * reduction))
        self.trans2 = _Transition(in_planes, out_planes, drop_rate)
        in_planes = out_planes

        # block3 (最后一块不再接 transition)
        self.block3 = _DenseBlock(n, in_planes, growth, block_cls, drop_rate)
        in_planes = int(in_planes + n * growth)

        # 末端 BN+ReLU（不做 GAP；把特征图交给上层 MODULE）
        self.bn_last = nn.BatchNorm2d(in_planes)
        self.relu    = nn.ReLU(inplace=True)

        self.out_channels = in_planes  # 供 Network.get_feature_length 使用（建议读取）

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        if pretrain and pretrained_model:
            self.load_model(pretrained_model)

    def load_model(self, pretrain_path: str):
        """
        只加载 shape 匹配的键；自动剥离常见前缀（module./backbone./encoder_q. 等）
        """
        print(f"Loading DenseNet backbone weights from {pretrain_path} ...")
        state = torch.load(pretrain_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        own = self.state_dict()
        new_dict = OrderedDict()
        for k, v in state.items():
            k2 = k
            for pref in ("module.", "encoder_q.", "backbone.", "features."):
                if k2.startswith(pref):
                    k2 = k2[len(pref):]
            # 输入通道不同会导致 conv1 不匹配；跳过即可
            if k2 in own and own[k2].shape == v.shape:
                new_dict[k2] = v

        missing, unexpected = self.load_state_dict(new_dict, strict=False)
        print(f"Backbone loaded. missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x, **kwargs):
        out = self.conv1(x)

        # block1
        out = self.block1(out)
        out = self.trans1(out)
        if kwargs.get("layer", None) == "layer1":
            out = kwargs["coef"] * out + (1 - kwargs["coef"]) * out[kwargs["index"]]

        # block2
        out = self.block2(out)
        out = self.trans2(out)
        if kwargs.get("layer", None) == "layer2":
            out = kwargs["coef"] * out + (1 - kwargs["coef"]) * out[kwargs["index"]]

        # block3（最后一块）
        out = self.block3(out)
        out = self.relu(self.bn_last(out))
        if kwargs.get("layer", None) == "layer3":
            out = kwargs["coef"] * out + (1 - kwargs["coef"]) * out[kwargs["index"]]

        # 不做 GAP；直接返回特征图
        return out

def densenet_backbone(
    cfg,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=1,   # DenseNet CIFAR 版没有用到，可忽略
):
    return DenseNetBackbone(cfg, pretrain=pretrain, pretrained_model=pretrained_model,
                            last_layer_stride=last_layer_stride)