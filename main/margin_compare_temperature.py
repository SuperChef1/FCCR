import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

import _init_paths  # 确保工程根路径加入 sys.path

from config import cfg as default_cfg
from net import build_network
from utils.repro import setup_repro, make_seeded_loader

# 只保留你目前会用到的 LAMDataset
from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_bbn import LAMDatasetBBN


# ----------------------------
# Repro + safe deserialization
# ----------------------------
def _enable_numpy_safe_unpickle():
    """
    兼容你 ckpt 里可能出现的 numpy scalar（参考你现有 eval 脚本/compare2.py 的做法）
    """
    to_allow = []
    for attr in ("_core", "core"):
        try:
            to_allow.append(getattr(np, attr).multiarray.scalar)
        except Exception:
            pass
    if to_allow:
        torch.serialization.add_safe_globals(to_allow)


def load_cfg(yaml_path: str):
    """
    基于默认 cfg clone 一份，再 merge yaml，最后 freeze。
    这样不会污染全局 cfg，也不会触发 immutable 修改报错。
    """
    c = default_cfg.clone()
    c.defrost()
    c.merge_from_file(yaml_path)
    c.freeze()
    return c


def build_valid_set(cfg_node):
    """
    你的 LAMDataset 里：mode != "train" 走 test split。
    这里统一用 "test" 作为验证/测试集。
    """
    ds_name = cfg_node.DATASET.DATASET
    if ds_name != "LAMDataset":
        raise ValueError(f"当前脚本只保留 LAMDataset，你的 cfg.DATASET.DATASET={ds_name}")
    return LAMDataset("test", cfg_node)


def _get_model_ckpt_path(cfg_node):
    return os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE)


@torch.no_grad()
def collect_logits_labels(cfg_node, device, valid_set, seed=111):
    """
    跑一遍验证集，收集 logits 和 labels（整集），用于温度标定 + margin 统计。
    """
    num_classes = valid_set.get_num_classes()
    model = build_network(cfg_node, mode="test", num_classes=num_classes)

    ckpt_path = _get_model_ckpt_path(cfg_node)
    model.load_model(
        ckpt_path,
        tau_norm=bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False))
        if hasattr(cfg_node.TEST, "TAU_NORM") else False,
        tau=float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0))
        if hasattr(cfg_node.TEST, "TAU_NORM") else 1.0,
    )

    model = torch.nn.DataParallel(model).to(device).eval()

    loader = make_seeded_loader(
        valid_set,
        batch_size=cfg_node.TEST.BATCH_SIZE,
        num_workers=int(getattr(cfg_node.TEST, "NUM_WORKERS", 1)),
        shuffle=False,
        pin_memory=bool(getattr(cfg_node, "PIN_MEMORY", False)),
        drop_last=False,
        seed=seed,
    )

    logits_list, labels_list = [], []
    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        z = model(x).float().detach().cpu()
        logits_list.append(z)
        labels_list.append(y.detach().cpu().long())

    logits = torch.cat(logits_list, dim=0)   # [N,C]
    labels = torch.cat(labels_list, dim=0)   # [N]
    return logits, labels


def find_T_star(logits_dev, labels_dev):
    """
    在验证集上求温度 T*：min CE(logits/T, y)
    （完全仿照 compare2.py 的核心思路）
    """
    T = torch.tensor(1.0, device=logits_dev.device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    y = labels_dev.long()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits_dev / T, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        # 极少数情况下 LBFGS 会因为数值问题中断；此时退化为 T=1
        pass

    return T.detach().clamp(min=1e-3, max=100.0)


@torch.no_grad()
def logit_margins_from_logits(logits_scaled: torch.Tensor, labels: torch.Tensor):
    """
    logit margin 定义：m = z_y - max_{c!=y} z_c
    logits_scaled: [N,C]（已经除以 T*）
    labels: [N]
    return: margins [N]
    """
    N, C = logits_scaled.shape
    y = labels.long().view(-1)  # [N]

    true_logit = logits_scaled[torch.arange(N), y]  # [N]

    # max other
    onehot = F.one_hot(y, num_classes=C).bool()
    max_other = logits_scaled.masked_fill(onehot, float("-inf")).max(dim=1).values  # [N]

    margins = true_logit - max_other
    return margins


def summarize_tail_distribution(tail_margins_np: np.ndarray):
    """
    tail logit margin 的“分布统计”：
    - mean/std/min/max
    - 分位数（含 5%）
    - 额外给出：<=0 的比例（直接误判/边界反转风险）
    """
    if tail_margins_np.size == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "q01": np.nan,
            "q05": np.nan,
            "q10": np.nan,
            "q25": np.nan,
            "q50": np.nan,
            "q75": np.nan,
            "q90": np.nan,
            "q95": np.nan,
            "q99": np.nan,
            "le0_ratio": np.nan,
        }

    qs = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    qv = np.quantile(tail_margins_np, qs)

    return {
        "n": int(tail_margins_np.size),
        "mean": float(np.mean(tail_margins_np)),
        "std": float(np.std(tail_margins_np)),
        "min": float(np.min(tail_margins_np)),
        "max": float(np.max(tail_margins_np)),
        "q01": float(qv[0]),
        "q05": float(qv[1]),  # 你要的 Tail margin 5% 分位数
        "q10": float(qv[2]),
        "q25": float(qv[3]),
        "q50": float(qv[4]),
        "q75": float(qv[5]),
        "q90": float(qv[6]),
        "q95": float(qv[7]),
        "q99": float(qv[8]),
        "le0_ratio": float(np.mean(tail_margins_np <= 0.0)),
    }


def eval_one_model_margins(cfg_node, device, valid_set, tail_cls=2, seed=111):
    """
    返回该模型的：
    - T*
    - mean logit margin（logits/T*）
    - tail logit margin 分布统计（logits/T*）
    """
    logits_cpu, labels_cpu = collect_logits_labels(cfg_node, device, valid_set, seed=seed)

    # 温度标定（在 GPU 上做优化更快）
    logits_dev = logits_cpu.to(device)
    labels_dev = labels_cpu.to(device)
    T_star = find_T_star(logits_dev, labels_dev).detach().cpu().item()

    logits_scaled = (logits_cpu / float(T_star)).float()  # z/T*
    probs = torch.softmax(logits_scaled, dim=1)  # p(x;T*)

    N, C = probs.shape
    y = labels_cpu.long().view(-1)
    p_true = probs[torch.arange(N), y]  # p_y

    onehot = F.one_hot(y, num_classes=C).bool()
    p_other_max = probs.masked_fill(onehot, float("-inf")).max(dim=1).values

    margins = (p_true - p_other_max).cpu().numpy()  # Δp ∈ [-1,1]

    mean_margin = float(np.mean(margins))

    tail_mask = (labels_cpu.numpy().astype(int) == int(tail_cls))
    tail_margins = margins[tail_mask]
    tail_stats = summarize_tail_distribution(tail_margins)

    return {
        "name": cfg_node.NAME,
        "ckpt": _get_model_ckpt_path(cfg_node),
        "T_star": float(T_star),
        "mean_logit_margin_T": mean_margin,
        "tail_cls": int(tail_cls),
        "tail_stats": tail_stats,
    }


def _fmt(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "nan"
    return f"{x:.{nd}f}"


def save_report(path_txt, results_list):
    os.makedirs(os.path.dirname(path_txt), exist_ok=True)
    with open(path_txt, "w", encoding="utf-8") as f:
        for r in results_list:
            ts = r["tail_stats"]
            f.write("=" * 80 + "\n")
            f.write(f"Model: {r['name']}\n")
            f.write(f"CKPT : {r['ckpt']}\n")
            f.write(f"T*   : {_fmt(r['T_star'], 6)}\n")
            f.write(f"Mean logit margin (logits/T*): {_fmt(r['mean_logit_margin_T'], 6)}\n")
            f.write(f"Tail cls = {r['tail_cls']}\n")
            f.write(f"Tail N   = {ts['n']}\n")
            f.write(f"Tail mean/std/min/max (logits/T*): "
                    f"{_fmt(ts['mean'],6)} / {_fmt(ts['std'],6)} / {_fmt(ts['min'],6)} / {_fmt(ts['max'],6)}\n")
            f.write("Tail quantiles (logits/T*): "
                    f"q01={_fmt(ts['q01'],6)}  q05={_fmt(ts['q05'],6)}  q10={_fmt(ts['q10'],6)}  "
                    f"q25={_fmt(ts['q25'],6)}  q50={_fmt(ts['q50'],6)}  q75={_fmt(ts['q75'],6)}  "
                    f"q90={_fmt(ts['q90'],6)}  q95={_fmt(ts['q95'],6)}  q99={_fmt(ts['q99'],6)}\n")
            f.write(f"Tail (margin<=0) ratio: {_fmt(ts['le0_ratio'],6)}\n")
        f.write("=" * 80 + "\n")


def main():
    ap = argparse.ArgumentParser("Temperature-normalized logit-margin stats for baseline/FCC/FCCR")
    ap.add_argument("--cfg_base", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_baseline_resnet18.yaml", type=str)
    ap.add_argument("--cfg_fcc", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_fcc_3_start1_resnet18.yaml", type=str)
    ap.add_argument("--cfg_fccr", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_fccr_3_start1_resnet18.yaml", type=str)
    ap.add_argument("--gpus", type=str, default="0", help="CUDA_VISIBLE_DEVICES, e.g. '0' or '0,1'")
    ap.add_argument("--seed", type=int, default=111, help="repro seed (default=111)")
    ap.add_argument("--tail_cls", type=int, default=2, help="tail class id (default=2)")
    ap.add_argument("--out", type=str, default="./margin_reports", help="output folder for txt report")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda")

    setup_repro(args.seed)
    _enable_numpy_safe_unpickle()

    cfg_base = load_cfg(args.cfg_base)
    cfg_fcc = load_cfg(args.cfg_fcc)
    cfg_fccr = load_cfg(args.cfg_fccr)

    # 三者共享同一份 valid_set（确保同一批样本对齐比较）
    valid_set = build_valid_set(cfg_base)

    results = []
    for cfg_node in (cfg_base, cfg_fcc, cfg_fccr):
        r = eval_one_model_margins(
            cfg_node=cfg_node,
            device=device,
            valid_set=valid_set,
            tail_cls=args.tail_cls,
            seed=args.seed,
        )
        results.append(r)

    # 控制台简表
    print("\n" + "=" * 100)
    print(f"[Tail cls = {args.tail_cls}]  Temperature-normalized logit margins (logits/T*)")
    print("-" * 100)
    print(f"{'Model':<28} {'T*':>10} {'MeanMargin':>14} {'TailMean':>14} {'TailQ05':>14} {'Tail<=0':>10}")
    for r in results:
        ts = r["tail_stats"]
        print(f"{r['name']:<28} "
              f"{_fmt(r['T_star'],6):>10} "
              f"{_fmt(r['mean_logit_margin_T'],6):>14} "
              f"{_fmt(ts['mean'],6):>14} "
              f"{_fmt(ts['q05'],6):>14} "
              f"{_fmt(ts['le0_ratio'],6):>10}")
    print("=" * 100 + "\n")

    # 保存详细报告
    out_txt = os.path.join(args.out, f"margin_temp_stats_tail{args.tail_cls}.txt")
    save_report(out_txt, results)
    print(f"[Saved] {out_txt}")


if __name__ == "__main__":
    # 可选：进一步收紧确定性（你之前也用过）
    torch.use_deterministic_algorithms(True, warn_only=True)
    main()
