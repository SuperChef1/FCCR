import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# BasicBlock（CIFAR 版）
# ---------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option="B"):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # 下采样的捷径分支（用 1x1 conv，稳定省事）
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = F.relu(out, inplace=True)
        return out


# ---------------------------
# ResNet-32 (CIFAR 风格)
# 输入默认 3 通道，如需别的通道数可在 cfg 里加 INPUT_CHANNELS
# ---------------------------
class ResNet32(nn.Module):
    def __init__(self, cfg=None, block=BasicBlock, num_blocks=(5, 5, 5)):
        super(ResNet32, self).__init__()
        in_ch = 150

        self.in_planes = 16
        # CIFAR stem：3x3, stride=1, 不做 maxpool
        self.conv1 = nn.Conv2d(in_ch, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(16)

        # 三个 stage：输出通道 16/32/64；在第2/3个stage做 stride=2 下采样
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)

        # 这里不做 GAP/FC，保持“骨干只输出特征图”的约定
        # 你的 Network 会接上 MODULE(GAP) 与 CLASSIFIER

        # 初始化
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _make_layer(self, block, planes, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, option="B"))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def load_model(self, pretrain):
        """
        兼容你工程的加载接口（如果有预训练就会被 Network 调用）。
        预训练权重里若带有 'module.' 或 'backbone.' 前缀，会自动剥离。
        """
        print(f"Loading Backbone pretrain model from {pretrain} ......")
        model_dict = self.state_dict()
        ckpt = torch.load(pretrain, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)

        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            if k.startswith("module."):
                k = k[7:]
            k = k.replace("backbone.", "")  # 兼容有 backbone. 前缀的情况
            # 只加载骨干里的键
            if k in model_dict and v.size() == model_dict[k].size():
                new_state[k] = v
        model_dict.update(new_state)
        self.load_state_dict(model_dict, strict=False)
        print("Backbone model has been loaded......")

    def forward(self, x, **kwargs):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)

        out = self.layer1(out)
        if "layer" in kwargs and kwargs["layer"] == "layer1":
            idx, coef = kwargs["index"], kwargs["coef"]
            out = coef * out + (1 - coef) * out[idx]

        out = self.layer2(out)
        if "layer" in kwargs and kwargs["layer"] == "layer2":
            idx, coef = kwargs["index"], kwargs["coef"]
            out = coef * out + (1 - coef) * out[idx]

        out = self.layer3(out)
        if "layer" in kwargs and kwargs["layer"] == "layer3":
            idx, coef = kwargs["index"], kwargs["coef"]
            out = coef * out + (1 - coef) * out[idx]

        return out


# 供 Network 调用的工厂函数（签名与 res50/resnet18_backbone 对齐）
def res32(
    cfg,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=2,   # CIFAR 架构这里不用，但保留参数以兼容调用
):
    model = ResNet32(cfg, block=BasicBlock, num_blocks=(5, 5, 5))
    if pretrain and pretrained_model:
        model.load_model(pretrain=pretrained_model)
    else:
        print("Choose to train from scratch")
    return model