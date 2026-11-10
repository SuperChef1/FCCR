"""BBN-specific ResNet-32 backbone for the 150-channel LAM input.

This follows the official CIFAR BBN split: the first four blocks of the
third stage are shared, while the final residual block is duplicated into a
conventional branch (CB) and a re-balancing branch (RB).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet32 import BasicBlock


class BBNResNet32(nn.Module):
    def __init__(self, cfg=None, block=BasicBlock, num_blocks=(5, 5, 5)):
        super().__init__()
        if num_blocks[2] < 2:
            raise ValueError("BBN ResNet-32 requires at least two blocks in layer3")

        in_channels = int(getattr(getattr(cfg, "BACKBONE", None), "IN_CHANNELS", 150))
        self.in_planes = 16
        self.conv1 = nn.Conv2d(
            in_channels, 16, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(16)

        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        # The first four layer3 blocks are shared. Only the final block is
        # branch-specific, matching the official BBN CIFAR implementation.
        self.layer3 = self._make_layer(block, 64, num_blocks[2] - 1, stride=2)
        self.cb_block = block(self.in_planes, self.in_planes, stride=1, option="B")
        self.rb_block = block(self.in_planes, self.in_planes, stride=1, option="B")

        self._init_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for current_stride in strides:
            layers.append(
                block(self.in_planes, planes, stride=current_stride, option="B")
            )
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)

    def _shared_forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)

    def forward(self, x, **kwargs):
        shared = self._shared_forward(x)
        if kwargs.get("feature_cb", False):
            return self.cb_block(shared)
        if kwargs.get("feature_rb", False):
            return self.rb_block(shared)

        cb_feature = self.cb_block(shared)
        rb_feature = self.rb_block(shared)
        return torch.cat((cb_feature, rb_feature), dim=1)

    def load_model(self, pretrain):
        """Load compatible shared weights without altering branch topology."""
        checkpoint = torch.load(pretrain, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        model_state = self.state_dict()
        compatible = {}
        for key, value in state.items():
            if key.startswith("module."):
                key = key[7:]
            key = key.replace("backbone.", "")
            if key in model_state and model_state[key].shape == value.shape:
                compatible[key] = value
        model_state.update(compatible)
        self.load_state_dict(model_state, strict=True)


def bbn_res32(
    cfg,
    pretrain=True,
    pretrained_model="",
    last_layer_stride=2,
):
    del last_layer_stride  # kept for compatibility with the model factory
    model = BBNResNet32(cfg, block=BasicBlock, num_blocks=(5, 5, 5))
    if pretrain and pretrained_model:
        model.load_model(pretrained_model)
    else:
        print("Choose to train BBN ResNet-32 from scratch")
    return model
