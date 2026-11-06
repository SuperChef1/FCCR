# #!/user/bin/env python3
# # -*- coding: utf-8 -*-
# import torch
# import torch.nn as nn
# from model import classifier
#
#
# def conv3x3(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, use_cos=False, cos_temp=8):
#         super(ResNet, self).__init__()
#         self.inplanes = 64
#         # 修改初始卷积层，使用3x3卷积核代替7x7
#         self.conv1 = nn.Conv2d(150, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入: 150x32x32 输出: 64x32x32
#         self.bn1 = nn.BatchNorm2d(64)
#         self.relu = nn.ReLU(inplace=True)
#         # 去除原始的最大池化层
#         self.layer1 = self._make_layer(block, 64, layers[0])  # 输入: 64x32x32 输出: 64x32x32
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 输入: 64x32x32 输出: 128x16x16
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 输入: 128x16x16 输出: 256x8x8
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 输入: 256x8x8 输出: 512x4x4
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输入: 512x4x4 输出: 512x1x1
#         self.fc = nn.Linear(512 * block.expansion, num_classes)  # 输入: 512 输出: num_classes
#         self.use_cos = use_cos
#         self.cos_classifier = classifier.Classifier(512 * block.expansion, num_classes, cos_temp)
#
#     def _make_layer(self, block, planes, blocks, stride=1):
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         feats = x
#         if self.use_cos:
#             x = self.cos_classifier(x)
#         else:
#             x = self.fc(x)
#         return x, torch.sum(torch.abs(feats), 1).reshape(-1, 1)
#
#
# def ResNet18(num_classes, use_cos, cos_temp):
#     return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_cos, cos_temp)

# !/user/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from model import classifier


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, use_cos=False, cos_temp=8):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.use_cos = use_cos
        # 修改初始卷积层，使用3x3卷积核代替7x7
        self.conv1 = nn.Conv2d(150, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 输入: 150x32x32 输出: 64x32x32
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.LeakyReLU(inplace=True)
        # 去除原始的最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 输入: 64x32x32 输出: 64x32x32
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # 输入: 64x32x32 输出: 128x16x16
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # 输入: 128x16x16 输出: 256x8x8
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # 输入: 256x8x8 输出: 512x4x4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 输入: 512x4x4 输出: 512x1x1

        if self.use_cos:
            self.cos_classifier = classifier.Classifier(512 * block.expansion, num_classes, cos_temp)
        else:
            self.fc = nn.Linear(512 * block.expansion, num_classes)  # 输入: 512 输出: num_classes
        # self.cos_classifier = classifier.Classifier(512 * block.expansion, num_classes, cos_temp)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)  # 输入: 512 输出: num_classes

        # 初始化所有层权重
        self._initialize_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        """自定义权重初始化逻辑"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 卷积层使用Kaiming初始化，适配ReLU激活
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BN层权重初始化为1，偏置为0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 分类层使用较小的高斯初始化（适配不平衡数据）
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
        # 如果使用余弦分类器，单独初始化（假设classifier模块已有初始化逻辑）
        if self.use_cos:
            self.cos_classifier._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        feats = x
        if self.use_cos:
            x = self.cos_classifier(x)
        else:
            x = self.fc(x)
        return x, torch.sum(torch.abs(feats), 1).reshape(-1, 1)


def ResNet18(num_classes, use_cos, cos_temp):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, use_cos, cos_temp)
