
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import _init_paths  # 保持你工程的导入机制一致
from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_bbn import LAMDatasetBBN
from copy import deepcopy
from config import cfg as _cfg, update_config
from net import build_network
from utils.repro import setup_repro, make_seeded_loader
from utils.utils import get_category_list


# ---------------------------
# PCA（sklearn优先，否则numpy兜底）
# ---------------------------
class _PCA2D_np:
    def __init__(self):
        self.mean_ = None
        self.components_ = None  # [2, D]

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


# ---------------------------
# tail 类：按训练集频次三分位（3类时tail=最少那类）
# ---------------------------
def head_mid_tail_sets(cfg_node):
    train_set = eval(cfg_node.DATASET.DATASET)("train", cfg_node)
    num_classes = train_set.get_num_classes()
    ann = train_set.get_annotations()
    freq, _ = get_category_list(ann, num_classes, cfg_node)
    freq = np.asarray(freq, dtype=int)
    order = np.argsort(freq)
    C = len(order)
    k = max(1, C // 3)
    tail = set(order[:k].tolist())
    mid  = set(order[k:-k].tolist()) if C - 2*k > 0 else set()
    head = set(order[-k:].tolist())
    return head, mid, tail, freq


# ---------------------------
# 一次性固定同一批样本：取valid前max_points（shuffle=False）
# ---------------------------
def collect_fixed_samples(cfg_node, split="valid", max_points=3000, seed=111):
    ds = eval(cfg_node.DATASET.DATASET)(split, cfg_node)
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
    return x, y, ds.get_num_classes()


# ---------------------------
# 抽取特征+logits（显式用 base/fcc/fccr 的特征分支）
# ---------------------------
@torch.no_grad()
def run_one(cfg_node, x_cpu, y_cpu, num_classes, mode_tag: str):
    device = torch.device("cuda")

    model = build_network(cfg_node, mode="test", num_classes=num_classes)
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

    logits = model(feat, classifier_flag=True, label=y)

    return {
        "cfg": cfg_node,
        "mode": mode_tag,
        "model": model,
        "model_file": model_file,
        "feat": feat.detach().cpu().numpy(),      # [N,D]
        "logits": logits.detach().cpu().numpy(),  # [N,C]
        "y_true": y_cpu.detach().cpu().numpy().astype(int),
        "y_pred": logits.argmax(dim=1).detach().cpu().numpy().astype(int),
    }


def true_margin(logits, y_true):
    logits = np.asarray(logits, dtype=np.float64)
    y_true = np.asarray(y_true, dtype=int)
    N, C = logits.shape
    lt = logits[np.arange(N), y_true]
    tmp = logits.copy()
    tmp[np.arange(N), y_true] = -1e18
    lo = tmp.max(axis=1)
    return lt - lo


@torch.no_grad()
def predict_grid(dp_model, grid_feat, chunk=20000):
    device = torch.device("cuda")
    out = []
    M = grid_feat.shape[0]
    for i in range(0, M, chunk):
        xx = torch.from_numpy(grid_feat[i:i+chunk]).float().to(device)
        lg = dp_model(xx, classifier_flag=True)
        out.append(lg.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


# ---------------------------
# 指标：Macro-F1 / Tail-F1 / High-risk Specificity(=tail TNR)
# ---------------------------
def macro_f1(y_true, y_pred, C):
    eps = 1e-12
    f1s = []
    for c in range(C):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2*p*r / (p + r + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def tail_f1(y_true, y_pred, tail_set):
    eps = 1e-12
    mask = np.isin(y_true, list(tail_set))
    if mask.sum() == 0:
        return 0.0
    yt = y_true[mask]
    yp = y_pred[mask]
    # tail里只有1类时，直接用该类的F1
    classes = sorted(list(set(yt.tolist())))
    f1s = []
    for c in classes:
        tp = np.sum((yt == c) & (yp == c))
        fp = np.sum((yt != c) & (yp == c))
        fn = np.sum((yt == c) & (yp != c))
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2*p*r / (p + r + eps)
        f1s.append(f1)
    return float(np.mean(f1s))


def high_risk_specificity(y_true, y_pred, tail_set):
    # 把 tail 视为“正类”，High-risk Specificity = TN/(TN+FP)
    mask_pos = np.isin(y_true, list(tail_set))
    pos = mask_pos
    neg = ~mask_pos
    tn = np.sum(neg & ~np.isin(y_pred, list(tail_set)))
    fp = np.sum(neg &  np.isin(y_pred, list(tail_set)))
    return float(tn / (tn + fp + 1e-12))


def plot_panel(ax, pack, pca, tail_set, grid=350, title=""):
    feat = pack["feat"]
    logits = pack["logits"]
    y_true = pack["y_true"]
    y_pred = pack["y_pred"]
    C = logits.shape[1]
    dp_model = pack["model"]

    Z = pca.transform(feat)

    # 决策区域
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

    grid_logits = predict_grid(dp_model, grid_feat)
    grid_pred = grid_logits.argmax(axis=1).reshape(grid, grid)
    ax.contourf(xx, yy, grid_pred, levels=np.arange(C+1)-0.5, alpha=0.22)

    # 散点（按真值类别上色）
    ax.scatter(Z[:,0], Z[:,1], c=y_true, s=14)

    # tail 边界点：tail true-margin 最低5%
    m = true_margin(logits, y_true)
    tail_mask = np.isin(y_true, list(tail_set))
    if tail_mask.sum() > 0:
        thr = np.quantile(m[tail_mask], 0.05)
        boundary = tail_mask & (m <= thr)
        ax.scatter(Z[boundary,0], Z[boundary,1],
                   facecolors="none", edgecolors="k",
                   s=55, linewidths=1.2)

    # 指标写标题
    mf1 = macro_f1(y_true, y_pred, C)
    tf1 = tail_f1(y_true, y_pred, tail_set)
    hrs = high_risk_specificity(y_true, y_pred, tail_set)

    tail_wrong = int(((y_pred != y_true) & tail_mask).sum())
    tail_total = int(tail_mask.sum())

    ax.set_title(
        f"{title}\nTail wrong {tail_wrong}/{tail_total} | Macro-F1 {mf1:.3f} | Tail-F1 {tf1:.3f} | HR-Spec {hrs:.3f}",
        fontsize=10
    )
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")


def load_cfg(cfg_path):
    c = deepcopy(_cfg)
    args = argparse.Namespace(cfg=cfg_path, opts=[])
    update_config(c, args)
    return c


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
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--max_points", type=int, default=3000)
    ap.add_argument("--grid", type=int, default=350)
    ap.add_argument("--out_dir", default="./vis_boundary")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    setup_repro(111)

    cfg_base = load_cfg(args.cfg_base)
    cfg_fcc  = load_cfg(args.cfg_fcc)
    cfg_fccr = load_cfg(args.cfg_fccr)

    # tail 集合：统一用 baseline 的 train 统计（同数据集时三者一致）
    _, _, tail_set, _ = head_mid_tail_sets(cfg_base)

    # 固定同一批样本（用 baseline 的 dataset 来取）
    x_cpu, y_cpu, C = collect_fixed_samples(cfg_base, split=args.split, max_points=args.max_points, seed=111)
    # 写入 NUM_CLASSES，防止有些cfg里没填
    base_pack = run_one(cfg_base, x_cpu, y_cpu, C, mode_tag="base")
    fcc_pack = run_one(cfg_fcc, x_cpu, y_cpu, C, mode_tag="fcc")
    fccr_pack = run_one(cfg_fccr, x_cpu, y_cpu, C, mode_tag="fccr")

    # 共享 PCA：用三者特征拼接 fit（同一坐标系）
    all_feat = np.concatenate([base_pack["feat"], fcc_pack["feat"], fccr_pack["feat"]], axis=0)
    pca = make_pca().fit(all_feat)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"baseline_fcc_fccr_boundary_{args.split}.png")

    fig, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), dpi=160)
    plot_panel(axes[0], base_pack, pca, tail_set, grid=args.grid, title="Baseline")
    plot_panel(axes[1], fcc_pack,  pca, tail_set, grid=args.grid, title="FCC")
    plot_panel(axes[2], fccr_pack, pca, tail_set, grid=args.grid, title="FCCR")

    fig.suptitle("Decision regions in PCA-projected classifier-input feature space (same samples, shared PCA)", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[SAVE] {out_png}")

    # 可选：单独保存三张图（你需要的话把下面注释取消）
    # for ax, name in zip(axes, ["baseline","fcc","fccr"]):
    #     fig2 = plt.figure(figsize=(6.2,5.2), dpi=160)
    #     ax2 = fig2.add_subplot(111)
    #     # 重新画一次（或从axes复制，这里省事起见重画）
    # plt.close("all")


if __name__ == "__main__":
    main()
