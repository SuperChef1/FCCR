import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import _init_paths
from dataset.lam_dataset import LAMDataset
from copy import deepcopy
from config import cfg as _cfg, update_config
from net import Network
from utils.repro import setup_repro, make_seeded_loader


# ---------- PCA ----------
class _PCA2D_np:
    def __init__(self):
        self.mean_ = None
        self.components_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0, keepdims=True)
        Xc = X - self.mean_
        _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[:2, :]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean_) @ self.components_.T

    def inverse_transform(self, Z):
        Z = np.asarray(Z, dtype=np.float64)
        return Z @ self.components_ + self.mean_


def make_pca():
    try:
        from sklearn.decomposition import PCA
        class _SKPCA:
            def __init__(self):
                self.pca = PCA(n_components=2, random_state=0)
            def fit(self, X):
                self.pca.fit(X); return self
            def transform(self, X):
                return self.pca.transform(X)
            def inverse_transform(self, Z):
                return self.pca.inverse_transform(Z)
        return _SKPCA()
    except Exception:
        return _PCA2D_np()


# ---------- utils ----------
def load_cfg(cfg_path):
    c = deepcopy(_cfg)
    args = argparse.Namespace(cfg=cfg_path, opts=[])
    update_config(c, args)
    return c


def collect_fixed_samples(cfg_node, split="valid", max_points=5000, seed=111):
    ds = eval(cfg_node.DATASET.DATASET)(split, cfg_node)
    num_classes = ds.get_num_classes()

    loader = make_seeded_loader(
        ds,
        batch_size=getattr(cfg_node.TEST, "BATCH_SIZE", 64),
        num_workers=1,
        shuffle=False,
        pin_memory=getattr(cfg_node, "PIN_MEMORY", False),
        drop_last=False,
        seed=seed,
    )

    xs, ys = [], []
    seen = 0
    for images, gt, _ in loader:
        xs.append(images)
        ys.append(gt)
        seen += images.size(0)
        if seen >= max_points:
            break

    x = torch.cat(xs, dim=0)[:max_points]
    y = torch.cat(ys, dim=0)[:max_points]
    return x, y, num_classes


def filter_pair(x_cpu, y_cpu, cls_a=1, cls_b=2):
    y = y_cpu.detach().cpu().numpy().astype(int)
    keep = np.isin(y, [cls_a, cls_b])
    if keep.sum() == 0:
        raise RuntimeError(f"No samples found for labels {{{cls_a},{cls_b}}}.")
    x_f = x_cpu[keep]
    y_f = y_cpu[keep]
    # 二值化：cls_a -> 0, cls_b -> 1
    y_bin = (y_f == cls_b).long()
    return x_f, y_f, y_bin


@torch.no_grad()
def run_one(cfg_node, x_cpu, y_cpu, y_bin_cpu, num_classes, mode_tag: str, cls_a=1, cls_b=2):
    device = torch.device("cuda")

    model = Network(cfg_node, mode="test", num_classes=num_classes)
    model_file = os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=False, tau=1.0)
    model = torch.nn.DataParallel(model).to(device).eval()

    x = x_cpu.to(device)
    y = y_cpu.to(device)

    now_epoch = int(getattr(cfg_node.TRAIN, "MAX_EPOCH", 9999))

    if mode_tag == "fcc":
        feat = model(x, feature_fcc_flag=True, label=y, now_epoch=now_epoch)
    elif mode_tag == "fccr":
        feat = model(x, feature_fccr_flag=True, label=y, now_epoch=now_epoch)
    else:
        feat = model(x, feature_flag=True)

    logits = model(feat, classifier_flag=True, label=y)          # [N,3]
    logits_np = logits.detach().cpu().numpy()

    # 仅取(1,2)子向量做二分类
    pair_logits = logits_np[:, [cls_a, cls_b]]                   # [N,2]
    pred_bin = pair_logits.argmax(axis=1).astype(int)            # 0->cls_a, 1->cls_b

    return {
        "cfg": cfg_node,
        "mode": mode_tag,
        "model": model,
        "model_file": model_file,
        "feat": feat.detach().cpu().numpy(),
        "pair_logits": pair_logits,
        "y_true_pair": y_cpu.detach().cpu().numpy().astype(int),  # 原始标签(1/2)
        "y_true_bin": y_bin_cpu.detach().cpu().numpy().astype(int),
        "y_pred_bin": pred_bin,
        "cls_a": cls_a,
        "cls_b": cls_b,
    }


def bin_f1(y_true_bin, y_pred_bin, pos=1):
    eps = 1e-12
    yt = np.asarray(y_true_bin, dtype=int)
    yp = np.asarray(y_pred_bin, dtype=int)
    tp = np.sum((yt == pos) & (yp == pos))
    fp = np.sum((yt != pos) & (yp == pos))
    fn = np.sum((yt == pos) & (yp != pos))
    p = tp / (tp + fp + eps)
    r = tp / (tp + fn + eps)
    f1 = 2*p*r / (p + r + eps)
    return float(f1), float(p), float(r)


@torch.no_grad()
def predict_grid_pair(dp_model, grid_feat, cls_a=1, cls_b=2, chunk=20000):
    device = torch.device("cuda")
    out = []
    M = grid_feat.shape[0]
    for i in range(0, M, chunk):
        xx = torch.from_numpy(grid_feat[i:i+chunk]).float().to(device)
        lg = dp_model(xx, classifier_flag=True)   # [m,3]
        lg = lg.detach().cpu().numpy()[:, [cls_a, cls_b]]  # [m,2]
        out.append(lg)
    pair_logits = np.concatenate(out, axis=0)
    return pair_logits.argmax(axis=1)  # 0/1


def plot_panel(ax, pack, pca, grid=350, q=0.05, title=""):
    feat = pack["feat"]
    pair_logits = pack["pair_logits"]  # [N,2]
    y_true_bin = pack["y_true_bin"]
    y_pred_bin = pack["y_pred_bin"]
    dp_model = pack["model"]
    cls_a, cls_b = pack["cls_a"], pack["cls_b"]

    Z = pca.transform(feat)

    # 决策区域（仅1/2二分类）
    x_min, x_max = Z[:,0].min(), Z[:,0].max()
    y_min, y_max = Z[:,1].min(), Z[:,1].max()
    pad_x = 0.08 * (x_max - x_min + 1e-9)
    pad_y = 0.08 * (y_max - y_min + 1e-9)
    x_min, x_max = x_min - pad_x, x_max + pad_x
    y_min, y_max = y_min - pad_y, y_max + pad_y

    xs = np.linspace(x_min, x_max, grid)
    ys = np.linspace(y_min, y_max, grid)
    xx, yy = np.meshgrid(xs, ys)
    grid_2d = np.stack([xx.ravel(), yy.ravel()], axis=1)
    grid_feat = pca.inverse_transform(grid_2d)
    grid_pred = predict_grid_pair(dp_model, grid_feat, cls_a=cls_a, cls_b=cls_b).reshape(grid, grid)
    ax.contourf(xx, yy, grid_pred, levels=np.arange(2+1)-0.5, alpha=0.22)

    # 散点（按二分类真值上色：0=cls_a,1=cls_b）
    ax.scatter(Z[:,0], Z[:,1], c=y_true_bin, s=14)

    # 边界点：|logit_b - logit_a| 最小的q分位（最接近分界）
    margin = (pair_logits[:,1] - pair_logits[:,0]).astype(np.float64)  # >0倾向cls_b
    abs_m = np.abs(margin)
    thr = np.quantile(abs_m, q)
    boundary = abs_m <= thr
    ax.scatter(Z[boundary,0], Z[boundary,1],
               facecolors="none", edgecolors="k", s=55, linewidths=1.2)

    # 指标（以“大缺陷=cls_b”为正类 pos=1）
    f1, prec, rec = bin_f1(y_true_bin, y_pred_bin, pos=1)
    wrong = int((y_true_bin != y_pred_bin).sum())
    total = int(len(y_true_bin))

    ax.set_title(
        f"{title} (only labels {cls_a} vs {cls_b})\n"
        f"Wrong {wrong}/{total} | Large({cls_b}) F1 {f1:.3f} P {prec:.3f} R {rec:.3f} | Boundary q={q:.2f}",
        fontsize=10
    )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_base",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\alexnet\LAM_baseline_alexnet.yaml",
        type=str,)
    ap.add_argument("--cfg_fcc",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\alexnet\LAM_fcc_0.5_start1_alexnet.yaml",
        type=str,)
    ap.add_argument("--cfg_fccr",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\alexnet\LAM_fccr_0.5_start1_alexnet.yaml",
        type=str,)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--cls_a", type=int, default=1)
    ap.add_argument("--cls_b", type=int, default=2)
    ap.add_argument("--max_points", type=int, default=5000)
    ap.add_argument("--grid", type=int, default=350)
    ap.add_argument("--q", type=float, default=0.05)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out_dir", default="./vis_boundary")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    setup_repro(111)

    cfg_base = load_cfg(args.cfg_base)
    cfg_fcc  = load_cfg(args.cfg_fcc)
    cfg_fccr = load_cfg(args.cfg_fccr)

    # 固定同一批样本（先取大一些，再筛(1,2)）
    x_cpu, y_cpu, num_classes = collect_fixed_samples(cfg_base, split=args.split, max_points=args.max_points, seed=111)
    x_pair, y_pair, y_bin = filter_pair(x_cpu, y_cpu, cls_a=args.cls_a, cls_b=args.cls_b)

    base_pack = run_one(cfg_base, x_pair, y_pair, y_bin, num_classes, mode_tag="base", cls_a=args.cls_a, cls_b=args.cls_b)
    fcc_pack  = run_one(cfg_fcc,  x_pair, y_pair, y_bin, num_classes, mode_tag="fcc",  cls_a=args.cls_a, cls_b=args.cls_b)
    fccr_pack = run_one(cfg_fccr, x_pair, y_pair, y_bin, num_classes, mode_tag="fccr", cls_a=args.cls_a, cls_b=args.cls_b)

    # 共享 PCA：三者特征拼接拟合，保证同一坐标系
    all_feat = np.concatenate([base_pack["feat"], fcc_pack["feat"], fccr_pack["feat"]], axis=0)
    pca = make_pca().fit(all_feat)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"baseline_fcc_fccr_pair{args.cls_a}{args.cls_b}_{args.split}.png")

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), dpi=160)
    plot_panel(axes[0], base_pack, pca, grid=args.grid, q=args.q, title="Baseline")
    plot_panel(axes[1], fcc_pack,  pca, grid=args.grid, q=args.q, title="FCC")
    plot_panel(axes[2], fccr_pack, pca, grid=args.grid, q=args.q, title="FCCR")

    fig.suptitle("Binary slice visualization in PCA-projected classifier-input feature space (shared PCA)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()