import torch
import torch.nn as nn
from collections import OrderedDict

__all__ = ["vgg16_backbone"]

_VGG_CFG = {
    # VGG-16 (D)
    "D": [64, 64, "M",
          128, 128, "M",
          256, 256, 256, "M",
          512, 512, 512, "M",
          512, 512, 512, "M"],
}

def _make_layers(cfg_list, in_channels=3, use_bn=True):
    layers = []
    ch = in_channels
    for v in cfg_list:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv = nn.Conv2d(ch, v, kernel_size=3, padding=1, bias=False)
            if use_bn:
                layers += [conv, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv, nn.ReLU(inplace=True)]
            ch = v
    return nn.Sequential(*layers)

class VGG16Backbone(nn.Module):
    """
    只做特征提取，不含分类头。
    - 输入: B x C x 32 x 32（C 可配置，默认 3；支持 150）
    - 输出: B x 512 x 1 x 1 （VGG16 在 32x32 上经过 5 次池化）
    """
    def __init__(self, cfg,
                 pretrain: bool = True,
                 pretrained_model: str = "",
                 last_layer_stride: int = 2,   # 兼容签名，不使用
                 ):
        super().__init__()
        in_ch = int(getattr(getattr(cfg, "BACKBONE", object()), "IN_CHANNELS", 3))
        self.features = _make_layers(_VGG_CFG["D"], in_channels=in_ch, use_bn=True)

        # 若你一定要输出更大的空间尺寸（比如 512x2x2），可以把最后一个 MaxPool 替换成 stride=1，
        # 这里只保持标准 VGG16 配置：最后输出 1x1。
        self.out_channels = 512

        # 记录输入通道，加载预训练时跳过 conv1 不匹配
        self.in_channels = in_ch

        if pretrain and pretrained_model:
            self.load_model(pretrained_model)

    def load_model(self, pretrain_path: str):
        """
        只加载与 self.features 形状匹配的权重；若 in_channels!=3，会自动跳过第一层 conv。
        """
        print(f"Loading VGG16 backbone weights from {pretrain_path} ...")
        state = torch.load(pretrain_path, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]

        # 允许来自 torchvision 的键名：可能是 "features.X.weight/bias"
        # 只挑 features.* 且 shape 匹配的参数
        own = self.state_dict()
        new_dict = OrderedDict()
        for k, v in state.items():
            if not k.startswith("features."):
                continue
            if k in own and own[k].shape == v.shape:
                new_dict[k] = v
        missing, unexpected = self.load_state_dict(new_dict, strict=False)
        print(f"Backbone loaded. missing={len(missing)} unexpected={len(unexpected)}")

    def forward(self, x, **kwargs):
        return self.features(x)

def vgg16_backbone(cfg, pretrain=True, pretrained_model="", last_layer_stride=2):
    return VGG16Backbone(cfg, pretrain=pretrain, pretrained_model=pretrained_model,
                         last_layer_stride=last_layer_stride)