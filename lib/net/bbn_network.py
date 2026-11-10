"""Isolated full BBN network used only by BBN configurations."""

from collections import OrderedDict
import os

import torch
import torch.nn as nn

from backbone import bbn_res32
from fcc.fcc_functions import fcc
from fcc.fccr_functions import fccr
from modules import FCNorm, GAP, Identity, LWS


class BBNNetwork(nn.Module):
    """Bilateral-branch ResNet-32 with one joint classifier.

    The classifier consumes the concatenation of conventional-branch and
    re-balancing-branch features. During cumulative training the two halves
    are scaled by the BBN coefficient before concatenation; at inference both
    halves are used equally, matching the official BBN implementation.
    """

    def __init__(self, cfg, mode="train", num_classes=1000):
        super().__init__()
        if str(cfg.BACKBONE.TYPE).lower() != "bbn_res32":
            raise ValueError("BBNNetwork requires BACKBONE.TYPE='bbn_res32'")

        self.cfg = cfg
        self.mode = mode
        self.num_classes = num_classes
        pretrain = (
            mode == "train"
            and cfg.RESUME_MODEL == ""
            and cfg.BACKBONE.PRETRAINED_MODEL != ""
        )
        self.backbone = bbn_res32(
            cfg,
            pretrain=pretrain,
            pretrained_model=cfg.BACKBONE.PRETRAINED_MODEL,
            last_layer_stride=2,
        )
        self.module = self._get_module()
        self.classifier = self._get_classifier()

        if cfg.NETWORK.PRETRAINED and os.path.isfile(cfg.NETWORK.PRETRAINED_MODEL):
            self.load_model(cfg.NETWORK.PRETRAINED_MODEL)

    def forward(self, x, **kwargs):
        if kwargs.get("classifier_flag", False):
            return self.classifier(x)
        if kwargs.get("feature_cb", False):
            return self.extract_branch_feature(x, branch="cb", **kwargs)
        if kwargs.get("feature_rb", False):
            return self.extract_branch_feature(x, branch="rb", **kwargs)
        if kwargs.get("feature_fcc_flag", False):
            return self.extract_joint_feature(x, feature_mode="fcc", **kwargs)
        if kwargs.get("feature_fccr_flag", False):
            return self.extract_joint_feature(x, feature_mode="fccr", **kwargs)
        if kwargs.get("feature_flag", False):
            return self.extract_joint_feature(x)

        feature = self.extract_joint_feature(x)
        return self.classifier(feature)

    def extract_branch_feature(self, x, branch, **kwargs):
        branch_kwarg = {"feature_cb": True} if branch == "cb" else {"feature_rb": True}
        feature_map = self.backbone(x, **branch_kwarg)

        feature_mode = str(kwargs.get("feature_mode", "raw")).lower()
        labels = kwargs.get("label")
        now_epoch = int(kwargs.get("now_epoch", 0))
        if feature_mode == "fcc":
            if labels is None:
                raise ValueError("BBN FCC branch extraction requires labels")
            if self.cfg.FCC.WORK and now_epoch >= self.cfg.FCC.START_EPOCH:
                feature_map = fcc(
                    feature_map,
                    labels,
                    self.num_classes,
                    self.cfg,
                    now_epoch=now_epoch,
                )
        elif feature_mode == "fccr":
            if labels is None:
                raise ValueError("BBN FCCR branch extraction requires labels")
            if self.cfg.FCCR.WORK and now_epoch >= self.cfg.FCCR.START_EPOCH:
                feature_map = fccr(
                    feature_map,
                    labels,
                    self.num_classes,
                    self.cfg,
                    now_epoch=now_epoch,
                )
        elif feature_mode != "raw":
            raise ValueError(f"Unsupported BBN feature mode: {feature_mode}")

        feature = self.module(feature_map)
        return feature.view(feature.shape[0], -1)

    def extract_joint_feature(self, x, feature_mode="raw", **kwargs):
        feature_map = self.backbone(x)
        feature_mode = str(feature_mode).lower()
        labels = kwargs.get("label")
        now_epoch = int(kwargs.get("now_epoch", 0))
        if feature_mode == "fcc":
            if labels is None:
                raise ValueError("BBN FCC joint extraction requires labels")
            if self.cfg.FCC.WORK and now_epoch >= self.cfg.FCC.START_EPOCH:
                feature_map = fcc(
                    feature_map, labels, self.num_classes, self.cfg,
                    now_epoch=now_epoch,
                )
        elif feature_mode == "fccr":
            if labels is None:
                raise ValueError("BBN FCCR joint extraction requires labels")
            if self.cfg.FCCR.WORK and now_epoch >= self.cfg.FCCR.START_EPOCH:
                feature_map = fccr(
                    feature_map, labels, self.num_classes, self.cfg,
                    now_epoch=now_epoch,
                )
        elif feature_mode != "raw":
            raise ValueError(f"Unsupported BBN joint feature mode: {feature_mode}")
        feature = self.module(feature_map)
        return feature.view(feature.shape[0], -1)

    def _branch_feature_length(self):
        if self.cfg.MODULE.TYPE == "GAP":
            return 64
        if self.cfg.MODULE.TYPE == "Identity":
            height, width = tuple(self.cfg.INPUT_SIZE)
            height = (int(height) + 3) // 4
            width = (int(width) + 3) // 4
            return 64 * height * width
        raise NotImplementedError(f"Unsupported module: {self.cfg.MODULE.TYPE}")

    def get_feature_length(self):
        return 2 * self._branch_feature_length()

    def _get_module(self):
        if self.cfg.MODULE.TYPE == "GAP":
            return GAP()
        if self.cfg.MODULE.TYPE == "Identity":
            return Identity()
        raise NotImplementedError(f"Unsupported module: {self.cfg.MODULE.TYPE}")

    def _get_classifier(self):
        feature_length = self.get_feature_length()
        bias = self.cfg.CLASSIFIER.BIAS
        if self.cfg.CLASSIFIER.TYPE == "FCNorm":
            return FCNorm(feature_length, self.num_classes)
        if self.cfg.CLASSIFIER.TYPE in ("FC", "cRT"):
            return nn.Linear(feature_length, self.num_classes, bias=bias)
        if self.cfg.CLASSIFIER.TYPE == "LWS":
            return LWS(feature_length, self.num_classes, bias=bias)
        raise NotImplementedError(
            f"Unsupported BBN classifier: {self.cfg.CLASSIFIER.TYPE}"
        )

    def freeze_backbone(self):
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def load_backbone_model(self, backbone_path=""):
        self.backbone.load_model(backbone_path)

    def load_model(self, model_path, tau_norm=False, tau=1):
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        cleaned = OrderedDict()
        for key, value in state.items():
            if key.startswith("module."):
                key = key[7:]
            if key == "classifier.weight" and tau_norm:
                value = value / torch.pow(
                    torch.norm(value, 2, 1, keepdim=True), tau
                )
            cleaned[key] = value
        self.load_state_dict(cleaned, strict=True)
        print("BBN model has been loaded...")
