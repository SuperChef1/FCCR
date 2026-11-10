"""Training combiner for the isolated full BBN execution path."""

import torch

from core.evaluate import accuracy


class BBNCombiner:
    _FEATURE_MODE = {
        "bbn": "raw",
        "bbn_fcc": "fcc",
        "bbn_fccr": "fccr",
    }

    def __init__(self, cfg, device, num_class_list=None):
        del num_class_list
        self.cfg = cfg
        self.device = device
        self.type = str(cfg.TRAIN.COMBINER.TYPE).lower()
        if self.type not in self._FEATURE_MODE:
            raise ValueError(
                f"BBNNetwork requires one of {sorted(self._FEATURE_MODE)}, "
                f"got {cfg.TRAIN.COMBINER.TYPE!r}"
            )
        self.epoch_number = int(cfg.TRAIN.MAX_EPOCH)
        self.div_epoch = (
            100 * (self.epoch_number // 100 + 1)
            if self.epoch_number in (90, 180)
            else self.epoch_number
        )
        self.epoch = 1
        self.softmax = torch.nn.Softmax(dim=1)
        print(f"BBN combiner type: {self.type}")

    def update(self, epoch):
        self.epoch = int(epoch)

    def cumulative_weight(self):
        progress = (self.epoch - 1) / float(self.div_epoch)
        return 1.0 - progress ** 2

    def forward(self, model, criterion, image, label, meta, **kwargs):
        if "sample_image" not in meta or "sample_label" not in meta:
            raise ValueError(
                "Full BBN training requires sample_image and sample_label from "
                "LAMDatasetBBN"
            )

        image_cb = image.to(self.device)
        label_cb = label.to(self.device)
        image_rb = meta["sample_image"].to(self.device)
        label_rb = meta["sample_label"].to(self.device)
        feature_mode = self._FEATURE_MODE[self.type]

        feature_cb = model(
            image_cb,
            feature_cb=True,
            feature_mode=feature_mode,
            label=label_cb,
            now_epoch=self.epoch,
        )
        feature_rb = model(
            image_rb,
            feature_rb=True,
            feature_mode=feature_mode,
            label=label_rb,
            now_epoch=self.epoch,
        )

        alpha = self.cumulative_weight()
        mixed_feature = 2.0 * torch.cat(
            (alpha * feature_cb, (1.0 - alpha) * feature_rb), dim=1
        )
        output = model(mixed_feature, classifier_flag=True)
        loss = (
            alpha * criterion(output, label_cb, feature=mixed_feature)
            + (1.0 - alpha) * criterion(output, label_rb, feature=mixed_feature)
        )

        prediction = torch.argmax(self.softmax(output), dim=1)
        accuracy_cb = accuracy(
            prediction.detach().cpu().numpy(), label_cb.detach().cpu().numpy()
        )[0]
        accuracy_rb = accuracy(
            prediction.detach().cpu().numpy(), label_rb.detach().cpu().numpy()
        )[0]
        now_acc = alpha * accuracy_cb + (1.0 - alpha) * accuracy_rb

        if kwargs.get("return_logits", False):
            # The same fused logits are supervised by both BBN labels. Returning
            # both copies keeps the existing per-class diagnostic code isolated
            # from the BBN training implementation.
            stats_logits = torch.cat((output.detach(), output.detach()), dim=0)
            stats_labels = torch.cat((label_cb.detach(), label_rb.detach()), dim=0)
            return loss, now_acc, stats_logits, stats_labels
        return loss, now_acc
