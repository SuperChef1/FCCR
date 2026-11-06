import math
import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ["resnext50_backbone"]

def conv3x3(in_planes, out_planes, stride=1, groups=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, groups=groups)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=32, base_width=4, is_last=False):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, bias=False, groups=groups)
        self.bn2   = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x, relu=True):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out) if relu else out

class ResNeXtBackbone(nn.Module):
    """
    ResNeXt-50 特征主干（不含分类头）
    - 输入: B x C x 32 x 32（C 由 cfg.BACKBONE.IN_CHANNELS 指定，默认 3；150 也支持）
    - 输出: B x 2048 x 1 x 1（默认 layer4 stride=2；若要更大空间可把 last_layer_stride 设为 1）
    """
    def __init__(self, cfg, block=Bottleneck, layers=(3,4,6,3),
                 groups=32, width_per_group=4,
                 pretrain=True, pretrained_model="", last_layer_stride=2):
        super().__init__()
        self.cfg = cfg
        in_ch = int(getattr(getattr(cfg, "BACKBONE", object()), "IN_CHANNELS", 3))
        self.groups = int(getattr(getattr(cfg, "BACKBONE", object()), "GROUPS", groups))
        self.base_width = int(getattr(getattr(cfg, "BACKBONE", object()), "WIDTH_PER_GROUP", width_per_group))

        self.inplanes = 64
        # 适配任意输入通道
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.relu  = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 支持可变最后一层步幅
        self.layer4 = self._make_layer(block, 512, layers[3], stride=last_layer_stride)

        self.out_channels = 512 * block.expansion  # 2048

        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.0)
                m.bias.data.zero_()

        if pretrain and pretrained_model:
            self.load_model(pretrained_model)

    def _make_layer(self, block, planes, blocks, stride=1, is_last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample,
                        groups=self.groups, base_width=self.base_width)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                is_last=(is_last and i == blocks - 1)))
        return nn.Sequential(*layers)

    def load_model(self, pretrain_path: str):
        """
        仅加载与当前形状匹配的卷积/BN权重；会自动跳过第一层 conv1 形状不匹配的情况（例如 IN_CHANNELS!=3）。
        兼容常见 key 前缀：module., encoder_q., backbone., features.
        """
        print(f"Loading ResNeXt backbone weights from {pretrain_path} ...")
        state = torch.load(pretrain_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        own = self.state_dict()
        new_dict = OrderedDict()
        for k, v in state.items():
            k2 = k
            for pref in ("module.", "encoder_q.", "backbone."):
                if k2.startswith(pref):
                    k2 = k2[len(pref):]
            # 只接收网络里存在且 shape 匹配的键
            if k2 in own and own[k2].shape == v.shape:
                new_dict[k2] = v

        missing, unexpected = self.load_state_dict(new_dict, strict=False)
        print(f"Backbone loaded. missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x, **kwargs):
        # stem
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # layers（支持 combiner 的中间层插混）
        x = self.layer1(x)
        if kwargs.get("layer", None) == "layer1":
            x = kwargs["coef"] * x + (1 - kwargs["coef"]) * x[kwargs["index"]]

        x = self.layer2(x)
        if kwargs.get("layer", None) == "layer2":
            x = kwargs["coef"] * x + (1 - kwargs["coef"]) * x[kwargs["index"]]

        x = self.layer3(x)
        if kwargs.get("layer", None) == "layer3":
            x = kwargs["coef"] * x + (1 - kwargs["coef"]) * x[kwargs["index"]]

        x = self.layer4(x)
        if kwargs.get("layer", None) == "layer4":
            x = kwargs["coef"] * x + (1 - kwargs["coef"]) * x[kwargs["index"]]

        return x

def resnext50_backbone(
    cfg,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=2,
):
    return ResNeXtBackbone(cfg,
                           block=Bottleneck,
                           layers=(3, 4, 6, 3),
                           groups=int(getattr(getattr(cfg, "BACKBONE", object()), "GROUPS", 32)),
                           width_per_group=int(getattr(getattr(cfg, "BACKBONE", object()), "WIDTH_PER_GROUP", 4)),
                           pretrain=pretrain,
                           pretrained_model=pretrained_model,
                           last_layer_stride=last_layer_stride)