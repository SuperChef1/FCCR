"""LAM dataset adapter for full bilateral-branch BBN training."""

import random

import numpy as np

from .lam_dataset import LAMDataset


class LAMDatasetBBN(LAMDataset):
    """Return a natural sample plus a reverse-sampled BBN companion.

    The validation split remains identical to :class:`LAMDataset` and never
    performs reverse sampling. This keeps BBN data behavior isolated from all
    non-BBN experiments.
    """

    def __init__(self, mode="train", cfg=None, **kwargs):
        super().__init__(mode=mode, cfg=cfg, **kwargs)
        self.reverse_classes = None
        self.reverse_p = None

        if mode == "train":
            if cfg is None:
                raise ValueError("LAMDatasetBBN requires a configuration")
            sampler_type = str(cfg.TRAIN.SAMPLER.TYPE).lower()
            reverse_type = str(cfg.TRAIN.SAMPLER.BBN_SAMPLER.TYPE).lower()
            if sampler_type != "bbn sampler" or reverse_type != "reverse":
                raise ValueError(
                    "LAMDatasetBBN requires TRAIN.SAMPLER.TYPE='bbn sampler' "
                    "and BBN_SAMPLER.TYPE='reverse'"
                )

            self.reverse_classes = np.array(
                sorted(self.class_dict.keys()), dtype=np.int64
            )
            counts = np.array(
                [len(self.class_dict[int(cls)]) for cls in self.reverse_classes],
                dtype=np.float64,
            )
            if np.any(counts <= 0):
                raise ValueError("BBN reverse sampling requires every class to be non-empty")
            inverse_frequency = counts.max() / counts
            self.reverse_p = inverse_frequency / inverse_frequency.sum()
            print(
                "[BBN Sampling] class counts: "
                f"{counts.astype(int).tolist()}, reverse probabilities: "
                f"{self.reverse_p.tolist()}"
            )

    def __getitem__(self, index):
        image, label, meta = super().__getitem__(index)
        if self.mode != "train":
            return image, label, meta

        reverse_class = int(
            np.random.choice(self.reverse_classes, p=self.reverse_p)
        )
        reverse_index = random.choice(self.class_dict[reverse_class])
        meta["sample_image"] = self.input_data[reverse_index]
        meta["sample_label"] = int(self.label[reverse_index])
        meta["sample_image_id"] = int(self.pos_indices[reverse_index])
        return image, label, meta
