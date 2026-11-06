import torch.nn as nn
import torch

class AlexNet(nn.Module):
    def __init__(self, num_classes=3, init_weights=False):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(150, 96, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )

    def forward(self, x, **kwargs):
        x = self.features(x)
        return x

def alexnet_backbone(cfg, pretrain=True, pretrained_model="", last_layer_stride=2):
    model = AlexNet(num_classes=cfg.NUM_CLASSES if hasattr(cfg, "NUM_CLASSES") else 3)
    if pretrain and pretrained_model != "":
        model.load_state_dict(torch.load(pretrained_model))
        print("Loaded pretrained weights from", pretrained_model)
    else:
        print("Using randomly initialized weights for AlexNet")
    return model