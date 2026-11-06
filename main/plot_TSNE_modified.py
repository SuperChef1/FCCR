import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

import _init_paths  # 你的工程路径注入

from copy import deepcopy
from config import cfg as _cfg, update_config
from net import Network

# ---- 兼容导入：你的工程里可能是 utils.repro，也可能是 repro.py ----
try:
    from utils.repro import setup_repro, make_seeded_loader
except Exception:
    from repro import setup_repro, make_seeded_loader

# ---- dataset：按你现有习惯直接 import（不要 eval），避免 __init__.py 未导出 LAMDataset 的坑 ----
from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_balance import LAMDataset_balanced
from dataset.lam_dataset_LT import LAMDataset_LT


# -------------------------
# Config / Dataset helpers
# -------------------------
def load_cfg(cfg_path: str):
    c = deepcopy(_cfg)
    args = argparse.Namespace(cfg=cfg_path, opts=[])
    update_config(c, args)
    return c


def build_dataset(cfg_node, split: str):
    name = cfg_node.DATASET.DATASET
    if name == "LAMDataset":
        return LAMDataset(split, cfg_node)
    if name == "LAMDataset_balanced":
        return LAMDataset_balanced(split, cfg_node)
    if name == "LAMDataset_LT":
        return LAMDataset_LT(split, cfg_node)
    # 兜底：你如果 cfg 写了别的名字，这里直接报出来
    raise ValueError(f"Unsupported DATASET.DATASET='{name}'. Please add it in build_dataset().")


@torch.no_grad()
def collect_fixed_samples(cfg_node, split="valid", max_points=2500, seed=111):
    """
    固定同一批样本：shuffle=False，从 split 顺序截断 max_points。
    """
    ds = build_dataset(cfg_node, split)
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


# -------------------------
# Model forward helpers
# -------------------------
def _get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def extract_feat_logits(cfg_node, x_cpu, y_cpu, num_classes, mode_tag: str):
    """
    mode_tag: "base" / "fcc" / "fccr"
    返回：
      feat_np  [N,D]
      logits_np[N,C]
      y_true_np[N]
      y_pred_np[N]
      margin_np[N] (top1-top2)
      model_file
    """
    device = _get_device()

    model = Network(cfg_node, mode="test", num_classes=num_classes)
    model_file = os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=False, tau=1.0)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model).to(device)
    else:
        model = model.to(device)
    model.eval()

    x = x_cpu.to(device)
    y = y_cpu.to(device)

    now_epoch = int(getattr(cfg_node.TRAIN, "MAX_EPOCH", 9999))

    # ---- 抽 classifier-input feature ----
    if mode_tag == "fcc":
        feat = model(x, feature_flag=True, label=y, now_epoch=now_epoch)
    elif mode_tag == "fccr":
        feat = model(x, feature_flag=True, label=y, now_epoch=now_epoch)
    else:
        feat = model(x, feature_flag=True)

    # ---- logits ----
    # 统一传 label=y，确保你的 forward 分支需要 label 时也不会报错
    logits = model(feat, classifier_flag=True, label=y)  # [N,C]

    feat_np = feat.detach().cpu().numpy()
    logits_np = logits.detach().cpu().numpy()
    y_true_np = y_cpu.detach().cpu().numpy().astype(int)

    # y_pred
    y_pred_np = logits_np.argmax(axis=1).astype(int)

    # margin = top1 - top2（越小越危险）
    # 用 partition 比 sort 快
    top2 = np.partition(logits_np, kth=-2, axis=1)[:, -2:]  # 未排序的 top2
    top1v = top2.max(axis=1)
    top2v = top2.min(axis=1)
    margin_np = (top1v - top2v).astype(np.float64)

    return feat_np, logits_np, y_true_np, y_pred_np, margin_np, model_file


# -------------------------
# t-SNE helpers
# -------------------------
def run_tsne(feat_np, pca_dim=50, perplexity=30, seed=111):
    """
    常用稳健流程：PCA -> t-SNE
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = feat_np.astype(np.float64)

    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    Z = TSNE(
        n_components=2,
        perplexity=float(perplexity),
        learning_rate="auto",
        init="pca",
        random_state=seed,
        verbose=0,
    ).fit_transform(X)
    return Z


def margin_to_sizes(margin, size_min=10.0, size_max=90.0, qlo=0.05, qhi=0.95):
    """
    把 margin 映射为点大小：
      margin 小 -> 点大（危险点更醒目）
    用分位数裁剪，避免极值把尺度拉爆。
    """
    m = np.asarray(margin, dtype=np.float64)
    lo = np.quantile(m, qlo)
    hi = np.quantile(m, qhi)
    denom = max(hi - lo, 1e-12)
    t = np.clip((m - lo) / denom, 0.0, 1.0)   # margin 越大 t 越大
    inv = 1.0 - t                              # margin 越小 inv 越大
    sizes = size_min + (size_max - size_min) * inv
    return sizes


def plot_panel(ax, Z, y_true, y_pred, margin, title,
               size_min=10, size_max=90, qlo=0.05, qhi=0.95,
               show_err_ring=True):
    """
    颜色=真值类别；大小=margin(低margin大)；错分=黑圈
    """
    sizes = margin_to_sizes(margin, size_min=size_min, size_max=size_max, qlo=qlo, qhi=qhi)
    wrong = (y_pred != y_true)

    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y_true, s=sizes)

    if show_err_ring and wrong.any():
        ax.scatter(
            Z[wrong, 0], Z[wrong, 1],
            s=sizes[wrong] * 1.15,
            facecolors="none", edgecolors="k", linewidths=1.2
        )

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # ---- legend：class + misclassified ----
    classes = sorted(list(set(y_true.tolist())))
    handles = []
    labels = []
    for c in classes:
        handles.append(plt.Line2D(
            [0], [0], marker='o', linestyle='', markersize=6,
            markerfacecolor=sc.cmap(sc.norm(c)), markeredgecolor='none'
        ))
        labels.append(f"class {c}")

    if show_err_ring:
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=7,
                                  markerfacecolor='none', markeredgecolor='k'))
        labels.append("misclassified")

    ax.legend(handles, labels, loc="best", fontsize=9, framealpha=0.9)

    # 标注错分数量（不想显示可删）
    ax.text(
        0.02, 0.02,
        f"wrong: {int(wrong.sum())}/{len(y_true)}",
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom"
    )


# -------------------------
# Main
# -------------------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg_base",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_baseline_resnet18.yaml",
        type=str,)
    ap.add_argument("--cfg_fcc",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_fcc_1_start1_resnet18.yaml",
        type=str,)
    ap.add_argument("--cfg_fccr",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_fccr_1_start50_resnet18.yaml",
        type=str,)
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])

    ap.add_argument("--max_points", type=int, default=2500)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--pca_dim", type=int, default=50)
    ap.add_argument("--seed", type=int, default=111)

    ap.add_argument("--size_min", type=float, default=10.0)
    ap.add_argument("--size_max", type=float, default=90.0)
    ap.add_argument("--margin_qlo", type=float, default=0.05)
    ap.add_argument("--margin_qhi", type=float, default=0.95)

    ap.add_argument("--gpus", default="0")
    ap.add_argument("--out_dir", default="./vis_tsne")
    return ap.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    setup_repro(args.seed)

    cfg_base = load_cfg(args.cfg_base)
    cfg_fcc  = load_cfg(args.cfg_fcc)
    cfg_fccr = load_cfg(args.cfg_fccr)

    # 固定同一批样本（保证可比）
    x_cpu, y_cpu, num_classes = collect_fixed_samples(
        cfg_base, split=args.split, max_points=args.max_points, seed=args.seed
    )

    # 三个模型：抽特征 + logits + pred + margin
    feat_b, lg_b, yt, yp_b, m_b, mf_b = extract_feat_logits(cfg_base, x_cpu, y_cpu, num_classes, mode_tag="base")
    feat_f, lg_f, _,  yp_f, m_f, mf_f = extract_feat_logits(cfg_fcc,  x_cpu, y_cpu, num_classes, mode_tag="fcc")
    feat_r, lg_r, _,  yp_r, m_r, mf_r = extract_feat_logits(cfg_fccr, x_cpu, y_cpu, num_classes, mode_tag="fccr")

    # t-SNE：分别对三个模型跑（更像“每个模型一个嵌入空间”）
    Z_b = run_tsne(feat_b, pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)
    Z_f = run_tsne(feat_f, pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)
    Z_r = run_tsne(feat_r, pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"tsne_3class_margin_err_{args.split}.png")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=160)

    plot_panel(
        axes[0], Z_b, yt, yp_b, m_b,
        title=f"Baseline\n{os.path.basename(mf_b)}",
        size_min=args.size_min, size_max=args.size_max,
        qlo=args.margin_qlo, qhi=args.margin_qhi,
        show_err_ring=True
    )
    plot_panel(
        axes[1], Z_f, yt, yp_f, m_f,
        title=f"FCC\n{os.path.basename(mf_f)}",
        size_min=args.size_min, size_max=args.size_max,
        qlo=args.margin_qlo, qhi=args.margin_qhi,
        show_err_ring=True
    )
    plot_panel(
        axes[2], Z_r, yt, yp_r, m_r,
        title=f"FCCR\n{os.path.basename(mf_r)}",
        size_min=args.size_min, size_max=args.size_max,
        qlo=args.margin_qlo, qhi=args.margin_qhi,
        show_err_ring=True
    )

    fig.suptitle("t-SNE of classifier-input features (3-class) with margin-size + misclassified rings", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()