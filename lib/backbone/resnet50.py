import torch
import torch.nn as nn

# ------------------------------
# 基础 Block（与项目现有风格一致）
# ------------------------------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, padding=1, bias=False, stride=stride)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False, stride=1)
        self.bn2   = nn.BatchNorm2d(planes)
        # downsample
        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(out + identity)
        return out


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)

        if stride != 1 or self.expansion * planes != inplanes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )
        else:
            self.downsample = None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        residual = self.downsample(x) if self.downsample is not None else x
        out = self.relu(out + residual)
        return out


# ------------------------------
# ResNet 主体（backbone 版本：不带 GAP/FC）
# ------------------------------
class ResNet(nn.Module):
    def __init__(
        self,
        cfg,
        block_type,
        num_blocks,
        last_layer_stride=2,   # 与原项目保持一致
    ):
        super().__init__()
        self.block    = block_type
        self.inplanes = 64

        # 从 cfg 读取输入通道数与 stem 类型
        in_ch      = int(getattr(getattr(cfg, "BACKBONE", {}), "IN_CHANNELS", 3))
        cifar_stem = bool(getattr(getattr(cfg, "BACKBONE", {}), "CIFAR_STEM", True))  # 小图默认 True

        if cifar_stem:
            # CIFAR 风格：3x3, stride=1，无 MaxPool，更适合 32x32
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.relu  = nn.ReLU(inplace=True)
            self.pool  = None     # 没有首层最大池化
        else:
            # 标准 ImageNet 风格：7x7 stride=2 + MaxPool
            self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1   = nn.BatchNorm2d(64)
            self.relu  = nn.ReLU(inplace=True)
            self.pool  = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 四个 stage
        self.layer1 = self._make_layer(num_blocks[0],  64, stride=1)
        self.layer2 = self._make_layer(num_blocks[1], 128, stride=2)
        self.layer3 = self._make_layer(num_blocks[2], 256, stride=2)
        self.layer4 = self._make_layer(num_blocks[3], 512, stride=last_layer_stride)

    def _make_layer(self, num_block, planes, stride=1):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(self.block(self.inplanes, planes, stride=s))
            self.inplanes = planes * self.block.expansion
        return nn.Sequential(*layers)

    # 兼容你工程内的“分层 mixup”钩子：在各 layer 末尾判断
    def forward(self, x, **kwargs):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.pool is not None:
            out = self.pool(out)

        out = self.layer1(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer1':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        out = self.layer2(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer2':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        out = self.layer3(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer3':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        out = self.layer4(out)
        if 'layer' in kwargs and kwargs['layer'] == 'layer4':
            out = kwargs['coef'] * out + (1 - kwargs['coef']) * out[kwargs['index']]

        # 不做 GAP/FC，保持“backbone 输出 feature map”的约定
        return out

    # 预训练参数加载（自动处理 conv1 的通道数不匹配）
    def load_model(self, pretrain):
        print(f"Loading Backbone pretrain model from {pretrain} ......")
        model_dict    = self.state_dict()
        pretrain_dict = torch.load(pretrain, map_location="cpu")
        pretrain_dict = pretrain_dict.get("state_dict", pretrain_dict)

        from collections import OrderedDict
        new_dict = OrderedDict()
        for k, v in pretrain_dict.items():
            # 去掉 'module.' 前缀
            if k.startswith("module."):
                k = k[7:]
            # 只加载 backbone 相关参数（排除 fc / classifier）
            if "fc" in k or "classifier" in k:
                continue
            # 旧工程有 "backbone." 前缀时去掉
            k = k.replace("backbone.", "")

            # 特判 conv1 通道数不一致
            if k == "conv1.weight" and v.size(1) != self.conv1.weight.size(1):
                v = self._adapt_input_conv_weight(v, self.conv1.weight.size(1))
                print(f"  [adapt] conv1 weight expanded: {v.shape}")
            new_dict[k] = v

        # 合并并加载
        model_dict.update(new_dict)
        self.load_state_dict(model_dict, strict=False)
        print("Backbone model has been loaded......")

    @staticmethod
    def _adapt_input_conv_weight(w, in_channels_target):
        """
        将预训练的 conv1 权重（通常是 [out, 3, k, k]）适配到任意输入通道数。
        策略：
        - 若 target 通道数是原来的倍数：直接 repeat 后截断；
        - 否则：对原通道求均值，然后在通道维上 repeat 到目标通道数。
        """
        out_c, in_c, k1, k2 = w.shape
        if in_channels_target == in_c:
            return w
        if in_channels_target % in_c == 0:
            repeat_times = in_channels_target // in_c
            w = w.repeat(1, repeat_times, 1, 1)
            return w[:, :in_channels_target, :, :]
        # 非整数倍：用均值填充
        mean_w = w.mean(dim=1, keepdim=True)                     # [out,1,k,k]
        w_exp  = mean_w.repeat(1, in_channels_target, 1, 1)      # [out,target,k,k]
        return w_exp


# 工厂函数：与原工程保持一致的接口
def res50(
    cfg,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=2,
):
    """
    用法：BACKBONE.TYPE 设为 'res50'
    - 若想用 150 通道输入：在 cfg 里设 BACKBONE.IN_CHANNELS: 150
    - 32×32 小图建议 CIFAR_STEM=True（默认就是 True）
    """
    net = ResNet(
        cfg=cfg,
        block_type=BottleNeck,
        num_blocks=[3, 4, 6, 3],
        last_layer_stride=last_layer_stride,
    )
    if pretrain and pretrained_model != "":
        net.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return net