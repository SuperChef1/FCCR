import _init_paths
import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from dataset.lam_dataset import LAMDataset

from copy import deepcopy
from config import cfg as _cfg, update_config
from net import Network
from utils.repro import setup_repro, make_seeded_loader


def load_cfg(cfg_path: str):
    c = deepcopy(_cfg)
    args = argparse.Namespace(cfg=cfg_path, opts=[])
    update_config(c, args)
    return c


@torch.no_grad()
def collect_fixed_samples(cfg_node, split="valid", max_points=2500, seed=111):
    """
    取同一批样本：shuffle=False，从 split 顺序截断 max_points。
    返回: x_cpu [N,C,H,W], y_cpu [N], num_classes
    """
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


@torch.no_grad()
def extract_feature(cfg_node, x_cpu, y_cpu, num_classes, mode_tag: str):
    """
    mode_tag: base / fcc / fccr
    返回：feat [N,D] numpy, y_true [N] numpy
    """
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

    feat_np = feat.detach().cpu().numpy()
    y_np = y_cpu.detach().cpu().numpy().astype(int)
    return feat_np, y_np, model_file


def run_tsne(feat_np, pca_dim=50, perplexity=30, seed=111):
    """
    标准做法：先 PCA 到 pca_dim，再 t-SNE 到 2D。
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = feat_np.astype(np.float64)

    # (可选) 先中心化/标准化：这里用 PCA 自带中心化即可
    if pca_dim is not None and pca_dim > 0 and pca_dim < X.shape[1]:
        X = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        verbose=0,
    ).fit_transform(X)
    return Z


def plot_panel(ax, Z, y, title=""):
    """
    只画散点聚类，不画边界。颜色按类别。
    """
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, s=12)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("t-SNE-1")
    ax.set_ylabel("t-SNE-2")

    # legend：按出现的类别做
    classes = sorted(list(set(y.tolist())))
    handles = []
    labels = []
    for c in classes:
        handles.append(plt.Line2D([0], [0], marker='o', linestyle='', markersize=6,
                                  markerfacecolor=sc.cmap(sc.norm(c)), markeredgecolor='none'))
        labels.append(f"class {c}")
    ax.legend(handles, labels, loc="best", fontsize=9, framealpha=0.9)


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

    # 取同一批样本（用 baseline 的 dataset 取，三者公平）
    x_cpu, y_cpu, num_classes = collect_fixed_samples(
        cfg_base, split=args.split, max_points=args.max_points, seed=args.seed
    )

    # 三个模型各自抽特征
    feat_base, y, mf_base = extract_feature(cfg_base, x_cpu, y_cpu, num_classes, mode_tag="base")
    feat_fcc,  _, mf_fcc  = extract_feature(cfg_fcc,  x_cpu, y_cpu, num_classes, mode_tag="fcc")
    feat_fccr, _, mf_fccr = extract_feature(cfg_fccr, x_cpu, y_cpu, num_classes, mode_tag="fccr")

    # t-SNE：分别对三者做（形态更像你截图那种“每个模型一个面板”）
    Z_base = run_tsne(feat_base, pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)
    Z_fcc  = run_tsne(feat_fcc,  pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)
    Z_fccr = run_tsne(feat_fccr, pca_dim=args.pca_dim, perplexity=args.perplexity, seed=args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, f"tsne_3class_baseline_fcc_fccr_{args.split}.png")

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.6), dpi=160)
    plot_panel(axes[0], Z_base, y, title=f"Baseline\n{os.path.basename(mf_base)}")
    plot_panel(axes[1], Z_fcc,  y, title=f"FCC\n{os.path.basename(mf_fcc)}")
    plot_panel(axes[2], Z_fccr, y, title=f"FCCR\n{os.path.basename(mf_fccr)}")

    fig.suptitle("t-SNE visualization of classifier-input features (3-class)", fontsize=13)
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches="tight")
    print(f"[SAVE] {out_png}")


if __name__ == "__main__":
    main()