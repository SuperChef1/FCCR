from .resnet_cifar import res32_cifar
from .oltr_resnet import res10
from .MyBackobone import AlexNet
from .MyBackobone import alexnet_backbone
from .resnet_18 import resnet18_backbone
from .resnet_18 import ResNetBackbone
from .resnet50 import res50
from .resnet32 import res32
from .resnet32_bbn import bbn_res32
from .VGG16 import vgg16_backbone
from .Resnext50 import resnext50_backbone
from .densenet import densenet_backbone
__all__ = [
    'res50', 'res32', 'bbn_res32', 'res10', 'resnext50_backbone','AlexNet','alexnet_backbone','resnet18_backbone','ResNetBackbone','vgg16_backbone', 'densenet_backbone'
]
