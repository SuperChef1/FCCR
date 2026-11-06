import numpy as np
import torch, math
from core.evaluate import accuracy
from net import Network
from torch import nn

class Combiner:
    def __init__(self, cfg, device, num_class_list=None):
        self.cfg = cfg
        self.type = cfg.TRAIN.COMBINER.TYPE
        self.device = device
        self.num_class_list = torch.FloatTensor(num_class_list)
        self.epoch_number = cfg.TRAIN.MAX_EPOCH
        self.func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)
        self.initilize_all_parameters()


    def initilize_all_parameters(self):
        self.alpha = self.cfg.TRAIN.COMBINER.ALPHA
        self.manifold_mix_up_location = self.cfg.TRAIN.COMBINER.MANIFOLD_MIX_UP.LOCATION
        self.remix_kappa = self.cfg.TRAIN.COMBINER.REMIX.KAPPA
        self.remix_tau = self.cfg.TRAIN.COMBINER.REMIX.TAU
        print('_'*100)
        print('combiner type: ', self.type)
        print('alpha in combiner: ', self.alpha)
        if self.type in ['manifold_mix_up', 'manifold_mix_up_fcc']:
            if 'res32' in self.cfg.BACKBONE.TYPE:
                assert self.manifold_mix_up_location in ['layer1','layer2','layer3','pool','fc']
            else:
                assert self.manifold_mix_up_location in ['layer1','layer2','layer3','pool','fc','layer4']
            print('location in manifold mixup: ', self.manifold_mix_up_location)
        if self.type == 'remix':
            print('kappa in remix: ', self.remix_kappa)
            print('tau in remix: ', self.remix_tau)
        print('_'*100)

        # ---- MixUp 可控旋钮 ----
        mu = getattr(self.cfg.TRAIN.COMBINER, "MIXUP", None)
        self.mix_prob = getattr(mu, "PROB", 0.7) if mu else 1.0
        self.mix_l_minmax = tuple(getattr(mu, "L_MINMAX", (0.3, 0.7))) if mu else (0.3, 0.7)
        self.mix_start_epoch = getattr(mu, "START_EPOCH", 10) if mu else 0
        self.mix_end_epoch = getattr(mu, "END_EPOCH", self.epoch_number) if mu else self.epoch_number
        self.mix_class_gamma = getattr(mu, "CLASS_AWARE_GAMMA", 1.0) if mu else 0.0
        self.mix_same_prob = getattr(mu, "SAME_CLASS_PROB", 0.3) if mu else 0.0
        self.mix_asym = getattr(mu, "ASYMMETRIC", True) if mu else False

        # ---- Remix 专属旋钮（可与上面共用思路）----
        rm = getattr(self.cfg.TRAIN.COMBINER, "REMIX", None)
        self.remix_prob = getattr(rm, "PROB", 0.7) if rm else 1.0
        self.remix_l_minmax = tuple(getattr(rm, "L_MINMAX", (0.35, 0.65))) if rm else (0.35, 0.65)
        self.remix_start_epoch = getattr(rm, "START_EPOCH", 10) if rm else 0
        self.remix_end_epoch = getattr(rm, "END_EPOCH", self.epoch_number) if rm else self.epoch_number

    def update(self, epoch):
        self.epoch = epoch
    
    def forward(self, model, criterion, image, label, meta, **kwargs):
        return eval("self.{}".format(self.type))(
            model, criterion, image, label, meta, **kwargs
        )

    def _mix_active(self, epoch, prob, start_ep, end_ep):
        if not (start_ep <= epoch <= end_ep):
            return False
        # 用 torch.rand 保持可控随机（配合你的种子）
        return (torch.rand(1).item() < prob)

    def _class_aware_alpha(self, labels, base_alpha):
        if self.mix_class_gamma <= 1e-8 or self.num_class_list is None:
            return base_alpha
        n = self.num_class_list.to(labels.device)  # [C]
        counts = n[labels].clamp_min(1.0)  # [B]
        scale = torch.pow(counts / (n.max() + 1e-12), self.mix_class_gamma)
        # 逐样本 α，防止极端
        return (float(base_alpha) * scale).clamp(1e-3, 10.0)  # [B]

    def _sample_lambda(self, labels, alpha_base, l_minmax, asymmetric=False):
        # 逐样本 α → 采多次取均值，得到 batch 统一的 λ（稳定）
        if isinstance(alpha_base, torch.Tensor):
            vals = []
            for a in alpha_base:
                a = float(a.item())
                vals.append(np.random.beta(a, a))
            lam = float(np.mean(vals))
        else:
            a = float(alpha_base)
            lam = np.random.beta(a, a)
        if asymmetric:
            lam = 0.5 + 0.5 * lam  # 偏向“保留锚样本”
        lo, hi = l_minmax
        return float(np.clip(lam, lo, hi))

    def _pair_index(self, labels, same_class_prob=0.0):
        B = labels.size(0)
        idx = torch.randperm(B, device=labels.device)
        if same_class_prob <= 1e-8:
            return idx
        mask = (torch.rand(B, device=labels.device) < same_class_prob)
        # 简单同类就近替换（若类内只有1个就保持原 idx）
        for i in torch.where(mask)[0]:
            same = torch.where(labels == labels[i])[0]
            if len(same) > 1:
                # 随机挑一个不同于自己的同类样本
                j = same[torch.randint(0, len(same), (1,), device=labels.device)]
                if j == i and len(same) > 1:
                    j = same[(torch.arange(len(same), device=labels.device) + 1) % len(same)][0]
                idx[i] = j
        return idx


    def default(self, model, criterion, image, label, meta, **kwargs):
        return_logits = kwargs.get("return_logits", False)
        image, label = image.to(self.device), label.to(self.device)
        if 'sample_image' in meta and 'sample_label' in meta:
            image_b = meta["sample_image"].to(self.device)
            label_b = meta["sample_label"].to(self.device)
            image = torch.cat([image, image_b], dim=0)
            label = torch.cat([label, label_b])

        feature = model(image, feature_flag=True)
        output = model(feature, classifier_flag=True, label=label)

        loss = criterion(output, label, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        if return_logits:
            return loss, now_acc, output.detach(), label.detach()
        return loss, now_acc

    def fcc(self, model, criterion, image, label, meta, **kwargs):
        return_logits = kwargs.get("return_logits", False)
        image, label = image.to(self.device), label.to(self.device)
        if 'sample_image' in meta and 'sample_label' in meta:
            image_b = meta["sample_image"].to(self.device)
            label_b = meta["sample_label"].to(self.device)
            image = torch.cat([image, image_b], dim=0)
            label = torch.cat([label, label_b])

        feature = model(image, feature_fcc_flag=True, label=label, now_epoch=self.epoch)
        output = model(feature, classifier_flag=True, label=label)

        loss = criterion(output, label, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        if return_logits:
            return loss, now_acc, output.detach(), label.detach()
        return loss, now_acc

    def fccr(self, model, criterion, image, label, meta, **kwargs):
        return_logits = kwargs.get("return_logits", False)
        image, label = image.to(self.device), label.to(self.device)
        if 'sample_image' in meta and 'sample_label' in meta:
            image_b = meta["sample_image"].to(self.device)
            label_b = meta["sample_label"].to(self.device)
            image = torch.cat([image, image_b], dim=0)
            label = torch.cat([label, label_b])

        feature = model(image, feature_fccr_flag=True, label=label, now_epoch=self.epoch)
        output = model(feature, classifier_flag=True, label=label)

        loss = criterion(output, label, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        if return_logits:
            return loss, now_acc, output.detach(), label.detach()
        return loss, now_acc

    def auto_f3c(self, model, criterion, image, label, meta, **kwargs):
        """
        当 cfg.TRAIN.COMBINER.TYPE 直接设为 'auto_f3c' 时走这里。
        与 fcc/f3c 结构一致，只是调用 feature_auto_f3c_flag。
        """
        image, label = image.to(self.device), label.to(self.device)
        if 'sample_image' in meta and 'sample_label' in meta:
            image_b = meta["sample_image"].to(self.device)
            label_b = meta["sample_label"].to(self.device)
            image = torch.cat([image, image_b], dim=0)
            label = torch.cat([label, label_b])

        feature = model(image, feature_auto_f3c_flag=True, label=label, now_epoch=self.epoch)
        output = model(feature, classifier_flag=True, label=label)

        loss = criterion(output, label, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]

        return loss, now_acc

    def mix_up(self, model, criterion, image, label, meta, **kwargs):
        """
        Vanilla MixUp（无FCC/FCCR），接入：
          - 概率启用/关闭（PROB）
          - 训练阶段窗口（START/END_EPOCH）
          - 同类配对概率（SAME_CLASS_PROB）
          - 类感知 alpha（CLASS_AWARE_GAMMA）
          - λ 截断窗口（L_MINMAX）
          - 非对称混合（ASYMMETRIC）
        """
        import numpy as np
        image, label = image.to(self.device), label.to(self.device)

        # 1) 是否启用（epoch + prob）
        if not self._mix_active(self.epoch, self.mix_prob, self.mix_start_epoch, self.mix_end_epoch):
            feature = model(image, feature_flag=True)
            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            now_acc = accuracy(torch.argmax(self.func(output), 1).cpu().numpy(), label.cpu().numpy())[0]
            return loss, now_acc

        # 2) 同类/异类配对
        idx = self._pair_index(label, self.mix_same_prob)
        img_b = image[idx]
        lab_b = label[idx]

        # 3) 类感知 α → 采样 λ → 截断/非对称
        alpha_ps = self._class_aware_alpha(label, self.alpha)  # 标量或 [B]
        lam = self._sample_lambda(label, alpha_ps, self.mix_l_minmax, asymmetric=self.mix_asym)

        # 4) 输入层混合
        mixed_image = lam * image + (1.0 - lam) * img_b

        # 5) 前向与损失（标准 mixup 的双标签 CE）
        feature = model(mixed_image, feature_flag=True)
        output = model(feature, classifier_flag=True, label=label)
        loss = lam * criterion(output, label, feature=feature) \
               + (1.0 - lam) * criterion(output, lab_b, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = lam * accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0] \
                  + (1.0 - lam) * accuracy(now_result.cpu().numpy(), lab_b.cpu().numpy())[0]
        return loss, now_acc

    def manifold_mix_up_fcc(self, model, criterion, image, label, meta, **kwargs):
        """
        Manifold MixUp + FCC/FCCR：
        - layer1/2/3：在骨干层里先做 mixup → 再做 FCC/FCCR → module → flatten → FC
        - pool：先 backbone+module，再在池化特征图处 mixup → FCC/FCCR → flatten → FC
        - fc：先 backbone+module → FCC/FCCR → flatten → FC 得 logits，再在 logits 上线性混合
        """
        import numpy as np
        from fcc.fcc_functions import fcc as _fcc
        from fcc.fccr_functions import fccr as _fccr

        image, label = image.to(self.device), label.to(self.device)
        l = np.random.beta(self.alpha, self.alpha)
        idx = torch.randperm(image.size(0), device=image.device)
        label_a, label_b = label, label[idx]

        # --- 解包 DataParallel ---
        core = model.module if isinstance(model, nn.DataParallel) else model
        where = self.manifold_mix_up_location  # 'layer1'/'layer2'/'layer3'/'pool'/'fc'

        use_fccr = hasattr(self.cfg, "FCCR") and getattr(self.cfg.FCCR, "WORK", False)
        use_fcc  = hasattr(self.cfg, "FCC")  and getattr(self.cfg.FCC,  "WORK", False)

        def _apply_fcc_like(feat_map):
            # 注意用 core.num_classes（Network 里有 num_classes）
            if use_fccr:
                return _fccr(feat_map, label, core.num_classes, self.cfg, now_epoch=self.epoch)
            if use_fcc:
                return _fcc( feat_map, label, core.num_classes, self.cfg, now_epoch=self.epoch)
            return feat_map

        if where in ["layer1", "layer2", "layer3"]:
            # 在骨干中间层做 manifold mixup
            feat = core.backbone.forward(image, index=idx, layer=where, coef=l)  # [N,C,H,W]
            feat = _apply_fcc_like(feat)
            feat = core.module(feat)                       # GAP / Identity
            feat = feat.view(feat.size(0), -1)             # flatten
            output = model(feat, classifier_flag=True, label=label)

        elif where == "pool":
            # pool 处混合：先提特征再混合
            feat_a = core.module(core.backbone(image))
            feat_b = core.module(core.backbone(image[idx]))

            feat_a = _apply_fcc_like(feat_a)  # 注意这里传 label_a
            feat_b = _apply_fcc_like(feat_b)  # 这里传 label_b

            feat = l * feat_a + (1.0 - l) * feat_b
            feat = feat.view(feat.size(0), -1)
            output = model(feat, classifier_flag=True)

        elif where == "fc":
            # 先做 FCC/FCCR，再分类，最后在 logits 处线性混合
            feat_a = core.module(core.backbone(image))
            feat_b = core.module(core.backbone(image[idx]))

            feat_a = _apply_fcc_like(feat_a)  # 注意这里传 label_a
            feat_b = _apply_fcc_like(feat_b)  # 这里传 label_b

            feat = l * feat_a + (1.0 - l) * feat_b
            feat = feat.view(feat.size(0), -1)
            output = model(feat, classifier_flag=True)

        else:
            raise ValueError(f"Unknown manifold_mix_up location: {where}")

        loss = l * criterion(output, label_a) + (1.0 - l) * criterion(output, label_b)
        now_result = torch.argmax(self.func(output), 1)
        now_acc = (
            l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
            + (1.0 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )
        return loss, now_acc

    def remix(self, model, criterion, image, label, meta, **kwargs):
        """
        Vanilla Remix（无FCC/FCCR），接入与 mixup 相同的“旋钮”，并保留 Remix 的 kappa/tau 规则。
        """
        import numpy as np
        assert self.num_class_list is not None, "num_class_list is required"
        image, label = image.to(self.device), label.to(self.device)

        # 1) 是否启用（epoch + prob）
        if not self._mix_active(self.epoch, self.remix_prob, self.remix_start_epoch, self.remix_end_epoch):
            feature = model(image, feature_flag=True)
            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            now_acc = accuracy(torch.argmax(self.func(output), 1).cpu().numpy(), label.cpu().numpy())[0]
            return loss, now_acc

        # 2) 同类/异类配对
        idx = self._pair_index(label, self.mix_same_prob)
        img_b = image[idx]
        label_b = label[idx]

        # 3) 类感知 α → 采样 λ → 截断/非对称（Remix 用自己的 L_MINMAX）
        alpha_ps = self._class_aware_alpha(label, self.alpha)
        lam = self._sample_lambda(label, alpha_ps, self.remix_l_minmax, asymmetric=self.mix_asym)

        # 4) 输入层混合
        mixed_image = lam * image + (1.0 - lam) * img_b

        # 5) 前向
        feature = model(mixed_image, feature_flag=True)
        output = model(feature, classifier_flag=True)

        # 6) Remix 的类别频次重加权
        l_list = torch.full((image.size(0),), lam, device=self.device, dtype=torch.float32)
        n_counts = self.num_class_list.to(label.device)
        n_i, n_j = n_counts[label], n_counts[label_b].float()
        if lam < self.remix_tau:
            l_list[n_i / n_j >= self.remix_kappa] = 0
        if 1.0 - lam < self.remix_tau:
            l_list[(n_i * self.remix_kappa) / n_j <= 1] = 1

        # 7) 双标签 CE
        loss = l_list * criterion(output, label, feature=feature) \
               + (1.0 - l_list) * criterion(output, label_b, feature=feature)
        loss = loss.mean()

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (l_list * accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]
                   + (1.0 - l_list) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]).mean()
        return loss, now_acc

    def remix_fcc(self, model, criterion, image, label, meta, **kwargs):
        """
        Remix + FCCR/FCC：
        - 保留 Remix 的类频次重加权 (kappa/tau)
        - 加入 概率/λ截断/延迟/同类配对/类感知α/非对称混合，使混合更“克制”，利于 FCCR 胜出
        """
        import numpy as np
        assert self.num_class_list is not None, "num_class_list is required"
        image, label = image.to(self.device), label.to(self.device)

        # 1) 是否启用（epoch 窗口 + 概率）
        if not self._mix_active(self.epoch, self.remix_prob, self.remix_start_epoch, self.remix_end_epoch):
            # 不启用就按 FCCR/FCC 正常前向
            use_fccr = hasattr(self.cfg, "FCCR") and getattr(self.cfg.FCCR, "WORK", False)
            use_fcc = hasattr(self.cfg, "FCC") and getattr(self.cfg.FCC, "WORK", False)
            if use_fccr:
                feature = model(image, feature_fccr_flag=True, label=label, now_epoch=self.epoch)
            elif use_fcc:
                feature = model(image, feature_fcc_flag=True, label=label, now_epoch=self.epoch)
            else:
                feature = model(image, feature_flag=True)
            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            now_acc = accuracy(torch.argmax(self.func(output), 1).cpu().numpy(), label.cpu().numpy())[0]
            return loss, now_acc

        # 2) 同类/异类配对
        idx = self._pair_index(label, self.mix_same_prob)
        img_b = image[idx]
        label_b = label[idx]

        # 3) 类感知 α → 采样 λ → 截断/非对称（Remix 也使用更克制的 λ）
        alpha_ps = self._class_aware_alpha(label, self.alpha)
        lam = self._sample_lambda(label, alpha_ps, self.remix_l_minmax, asymmetric=self.mix_asym)

        # 4) 输入层混合
        mixed_image = lam * image + (1.0 - lam) * img_b

        # 5) FCCR / FCC 特征
        use_fccr = hasattr(self.cfg, "FCCR") and getattr(self.cfg.FCCR, "WORK", False)
        use_fcc = hasattr(self.cfg, "FCC") and getattr(self.cfg.FCC, "WORK", False)
        if use_fccr:
            feature = model(mixed_image, feature_fccr_flag=True, label=label, now_epoch=self.epoch)
        elif use_fcc:
            feature = model(mixed_image, feature_fcc_flag=True, label=label, now_epoch=self.epoch)
        else:
            feature = model(mixed_image, feature_flag=True)

        output = model(feature, classifier_flag=True)

        # 6) Remix 的类别频次重加权（在“克制”的 λ 基础上）
        l_list = torch.empty(image.shape[0]).fill_(lam).float().to(self.device)
        n_counts = self.num_class_list.to(label.device)  # 避免 CPU/GPU 索引不一致
        n_i, n_j = n_counts[label], n_counts[label_b].float()
        if lam < self.remix_tau:
            l_list[n_i / n_j >= self.remix_kappa] = 0
        if 1.0 - lam < self.remix_tau:
            l_list[(n_i * self.remix_kappa) / n_j <= 1] = 1

        # 7) 双标签 CE
        loss = l_list * criterion(output, label, feature=feature) \
               + (1.0 - l_list) * criterion(output, label_b, feature=feature)
        loss = loss.mean()

        now_result = torch.argmax(self.func(output), 1)
        now_acc = (l_list * accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]
                   + (1.0 - l_list) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]).mean()
        return loss, now_acc


    def bbn_mix(self, model, criterion, image, label, meta, **kwargs):
        r"""
        Reference:
            Zhou et al. BBN: Bilateral-Branch Network with Cumulative Learning for Long-Tailed Visual Recognition, CVPR 2020.

        We combine the sampling method of BBN, which consists of a uniform sampler and a reverse sampler, with input mixup.

        For more details about these two samplers, you can read the original paper https://arxiv.org/abs/1912.02413.
        """
        l = np.random.beta(self.alpha, self.alpha) # beta distribution

        image_a, image_b = image.to(self.device), meta["sample_image"].to(self.device)
        label_a, label_b = label.to(self.device), meta["sample_label"].to(self.device)

        # mix up two image
        mixed_image = l * image_a + (1 - l) * image_b

        mixed_output = model(mixed_image)

        loss = l * criterion(mixed_output, label_a) + (1 - l) * criterion(mixed_output, label_b)

        now_result = torch.argmax(self.func(mixed_output), 1)
        now_acc = (
                l * accuracy(now_result.cpu().numpy(), label_a.cpu().numpy())[0]
                + (1 - l) * accuracy(now_result.cpu().numpy(), label_b.cpu().numpy())[0]
        )
        return loss, now_acc

    def dive(self, model, criterion, image, label, meta, **kwargs):

        if not hasattr(self, 'model_t'):
            print('Loading the teacher model in DiVE')
            self.model_t = Network(self.cfg, mode="test", num_classes=len(self.num_class_list))
            self.model_t.load_model(self.cfg.TRAIN.COMBINER.DIVE.TEACHER_MODEL)
            self.model_t = torch.nn.DataParallel(self.model_t).cuda()
            self.model_t.eval()

        image, label = image.to(self.device), label.to(self.device)
        output_s = model(image)

        with torch.no_grad():
            output_t = self.model_t(image)

        loss = criterion(output_s, output_t, label)
        now_result = torch.argmax(self.func(output_s), 1)
        now_acc = accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0]
        return loss, now_acc

    def mix_up_fcc(self, model, criterion, image, label, meta, **kwargs):
        """Input MixUp + FCCR/FCC（优先用 FCCR；否则用 FCC；都没开就退化成普通 mixup）"""
        import numpy as np
        image, label = image.to(self.device), label.to(self.device)

        # 1) 是否启用（epoch 窗口 + 概率）
        if not self._mix_active(self.epoch, self.mix_prob, self.mix_start_epoch, self.mix_end_epoch):
            # 不启用就按 FCCR/FCC 正常前向
            use_fccr = hasattr(self.cfg, "FCCR") and getattr(self.cfg.FCCR, "WORK", False)
            use_fcc = hasattr(self.cfg, "FCC") and getattr(self.cfg.FCC, "WORK", False)
            if use_fccr:
                feature = model(image, feature_fccr_flag=True, label=label, now_epoch=self.epoch)
            elif use_fcc:
                feature = model(image, feature_fcc_flag=True, label=label, now_epoch=self.epoch)
            else:
                feature = model(image, feature_flag=True)
            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            now_acc = accuracy(torch.argmax(self.func(output), 1).cpu().numpy(), label.cpu().numpy())[0]
            return loss, now_acc

        # 2) 同类/异类配对
        idx = self._pair_index(label, self.mix_same_prob)
        img_b = image[idx]
        lab_b = label[idx]

        # 3) 类感知 α → 采样 λ → 截断/非对称
        alpha_ps = self._class_aware_alpha(label, self.alpha)  # 标量或 [B]
        lam = self._sample_lambda(label, alpha_ps, self.mix_l_minmax, asymmetric=self.mix_asym)

        # 4) 输入层混合
        mixed_image = lam * image + (1.0 - lam) * img_b

        # 5) FCCR/FCC 特征提取
        use_fccr = hasattr(self.cfg, "FCCR") and getattr(self.cfg.FCCR, "WORK", False)
        use_fcc = hasattr(self.cfg, "FCC") and getattr(self.cfg.FCC, "WORK", False)
        if use_fccr:
            feature = model(mixed_image, feature_fccr_flag=True, label=label, now_epoch=self.epoch)
        elif use_fcc:
            feature = model(mixed_image, feature_fcc_flag=True, label=label, now_epoch=self.epoch)
        else:
            feature = model(mixed_image, feature_flag=True)

        output = model(feature, classifier_flag=True, label=label)

        # 6) 双标签 CE（标准 mixup）
        loss = lam * criterion(output, label, feature=feature) \
               + (1.0 - lam) * criterion(output, lab_b, feature=feature)

        now_result = torch.argmax(self.func(output), 1)
        now_acc = lam * accuracy(now_result.cpu().numpy(), label.cpu().numpy())[0] \
                  + (1.0 - lam) * accuracy(now_result.cpu().numpy(), lab_b.cpu().numpy())[0]
        return loss, now_acc


