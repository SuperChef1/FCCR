# compare_by_3cfg.py
import os, argparse, numpy as np, torch
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn.functional as F
import _init_paths
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support

from config import cfg as default_cfg  # 默认 config 节点
from net import build_network
from utils.repro import setup_repro, make_seeded_loader
from utils.utils import valid_logger, get_category_list

# 你的数据集工厂（保持原脚本一致；cfg 里选 LAMDataset 就只会用到它）
from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_bbn import LAMDatasetBBN
from dataset.lam_dataset_balance import LAMDataset_balanced
from dataset.lam_dataset_LT import LAMDataset_LT

SEED = 111
setup_repro(SEED)

# 兼容 numpy 安全反序列化（与现有 eval 脚本一致）
_to_allow = []
for attr in ("_core", "core"):
    try:
        _to_allow.append(getattr(np, attr).multiarray.scalar)
    except Exception:
        pass
if _to_allow:
    torch.serialization.add_safe_globals(_to_allow)


def load_cfg(yaml_path: str):
    """基于默认 cfg 克隆一份，再 merge 当前 yaml。避免全局 cfg 被覆盖。"""
    c = default_cfg.clone()
    c.defrost()
    c.merge_from_file(yaml_path)
    c.freeze()
    return c


def build_dataset(split: str, cfg_node):
    """按 cfg 的 DATASET.DATASET 名称构造数据集。"""
    ds_name = cfg_node.DATASET.DATASET
    return eval(ds_name)(split, cfg_node)


@torch.no_grad()
def infer_one_cfg(model_cfg, device, valid_set):
    """
    用单个 cfg 的 TEST 参数在同一个 valid_set 上跑推理。
    返回 preds [N], labels [N], image_ids [N]
    """
    num_classes = valid_set.get_num_classes()
    model = build_network(model_cfg, mode="test", num_classes=num_classes)

    model_file = os.path.join(
        model_cfg.OUTPUT_DIR, model_cfg.NAME, "models", model_cfg.TEST.MODEL_FILE
    )
    model.load_model(
        model_file,
        tau_norm=bool(getattr(model_cfg.TEST.TAU_NORM, "USE_TAU_NORM", False))
        if hasattr(model_cfg.TEST, "TAU_NORM") else False,
        tau=float(getattr(model_cfg.TEST.TAU_NORM, "TAU", 1.0))
        if hasattr(model_cfg.TEST, "TAU_NORM") else 1.0,
    )
    model = torch.nn.DataParallel(model).to(device)
    model.eval()

    loader = make_seeded_loader(
        valid_set,
        batch_size=model_cfg.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=getattr(model_cfg, "PIN_MEMORY", False),
        drop_last=False,
        seed=SEED,
    )

    preds, labels, image_ids = [], [], []
    for images, gt, meta in tqdm(loader, desc=f"Infer [{model_cfg.NAME}]"):
        images = images.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy().astype(int)
        preds.append(pred)
        labels.append(gt.numpy().astype(int))
        if "image_id" in meta:
            image_ids.extend([int(x) for x in meta["image_id"]])
        else:
            start = len(image_ids)
            image_ids.extend(list(range(start, start + pred.shape[0])))

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    image_ids = np.array(image_ids)
    if image_ids.shape[0] != preds.shape[0]:
        image_ids = np.arange(preds.shape[0])
    return preds, labels, image_ids


@torch.no_grad()
def eval_margin_stats(cfg_node, device, valid_set, eps_list=(0.0, 0.2, 0.5)):
    """
    Logit-margin: m = logit_y - max_{c!=y} logit_c  （true-class margin）
    返回：mean_margin, near_boundary_ratios(|m|<=eps), per_class_rows
    """
    num_classes = valid_set.get_num_classes()
    model = build_network(cfg_node, mode="test", num_classes=num_classes)
    model.load_model(
        os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE),
        tau_norm=bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False))
        if hasattr(cfg_node.TEST, "TAU_NORM") else False,
        tau=float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0))
        if hasattr(cfg_node.TEST, "TAU_NORM") else 1.0
    )
    model = torch.nn.DataParallel(model).to(device).eval()

    loader = make_seeded_loader(
        valid_set,
        batch_size=cfg_node.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        seed=SEED
    )

    sums, cnt = 0.0, 0
    band = [0 for _ in eps_list]
    per_cls = {c: {"sum": 0.0, "cnt": 0, "band": [0 for _ in eps_list]} for c in range(num_classes)}

    for images, gt, _ in loader:
        images = images.to(device)
        y = gt.to(device).long()                  # [B]
        logits = model(images).float()            # [B,C]

        mask = F.one_hot(y, num_classes=logits.size(1)).bool()
        max_other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        margin = logits.gather(1, y.view(-1, 1)).squeeze(1) - max_other  # [B]

        sums += margin.sum().item()
        cnt += margin.numel()

        for i, eps in enumerate(eps_list):
            band[i] += int(((margin >= -eps) & (margin <= eps)).sum().item())

        for c in range(num_classes):
            m = (y == c)
            if m.any():
                mvals = margin[m]
                per_cls[c]["sum"] += float(mvals.sum().item())
                per_cls[c]["cnt"] += int(mvals.numel())
                for i, eps in enumerate(eps_list):
                    per_cls[c]["band"][i] += int(((mvals >= -eps) & (mvals <= eps)).sum().item())

    mean_margin = sums / max(1, cnt)
    band_ratio = [b / max(1, cnt) for b in band]

    per_rows = []
    for c in range(num_classes):
        ccnt = per_cls[c]["cnt"]
        cmargin = per_cls[c]["sum"] / max(1, ccnt)
        cband = [x / max(1, ccnt) for x in per_cls[c]["band"]]
        per_rows.append([c, ccnt, f"{cmargin:.6f}"] + [f"{r:.6f}" for r in cband])

    return mean_margin, band_ratio, per_rows


@torch.no_grad()
def eval_tail_logit_margin_stats(cfg_node, device, valid_set, tail_cls=2, hist_bins=40):
    """
    Tail logit margin（true-class margin）分布统计：
      仅对 label==tail_cls 的样本统计 margin = logit_y - max_{c!=y} logit_c
    返回 dict：
      count/mean/std/min/p05/p25/p50/p75/p95/max + hist_edges/hist_counts
    """
    num_classes = valid_set.get_num_classes()
    model = build_network(cfg_node, mode="test", num_classes=num_classes)
    model.load_model(
        os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE),
        tau_norm=bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False))
        if hasattr(cfg_node.TEST, "TAU_NORM") else False,
        tau=float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0))
        if hasattr(cfg_node.TEST, "TAU_NORM") else 1.0,
    )
    model = torch.nn.DataParallel(model).to(device).eval()

    loader = make_seeded_loader(
        valid_set,
        batch_size=cfg_node.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        seed=SEED,
    )

    tail_margins = []
    for images, gt, _ in loader:
        images = images.to(device)
        y = gt.to(device).long()
        logits = model(images).float()

        mask = F.one_hot(y, num_classes=logits.size(1)).bool()
        max_other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        margin = logits.gather(1, y.view(-1, 1)).squeeze(1) - max_other  # [B]

        m = (y == int(tail_cls))
        if m.any():
            tail_margins.append(margin[m].detach().cpu())

    if len(tail_margins) == 0:
        return {
            "tail_cls": int(tail_cls),
            "count": 0,
            "mean": 0.0, "std": 0.0, "min": 0.0,
            "p05": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p95": 0.0,
            "max": 0.0,
            "hist_edges": None,
            "hist_counts": None,
        }

    vals = torch.cat(tail_margins, dim=0).numpy().astype(np.float64)

    p05, p25, p50, p75, p95 = np.quantile(vals, [0.05, 0.25, 0.50, 0.75, 0.95])
    hist_counts, hist_edges = np.histogram(vals, bins=int(hist_bins))

    return {
        "tail_cls": int(tail_cls),
        "count": int(vals.size),
        "mean": float(vals.mean()),
        "std": float(vals.std(ddof=0)),
        "min": float(vals.min()),
        "p05": float(p05),   # 你要的 Tail margin 5% 分位数
        "p25": float(p25),
        "p50": float(p50),
        "p75": float(p75),
        "p95": float(p95),
        "max": float(vals.max()),
        "hist_edges": hist_edges,
        "hist_counts": hist_counts,
    }


@torch.no_grad()
def _collect_logits_labels(cfg_node, device, valid_set):
    """收集整份验证集的 logits/labels，供温度标定与概率-margin计算"""
    num_classes = valid_set.get_num_classes()
    model = build_network(cfg_node, mode="test", num_classes=num_classes)
    model.load_model(
        os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE),
        tau_norm=bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False)) if hasattr(cfg_node.TEST, "TAU_NORM") else False,
        tau=float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0)) if hasattr(cfg_node.TEST, "TAU_NORM") else 1.0,
    )
    model = torch.nn.DataParallel(model).to(device).eval()

    loader = make_seeded_loader(
        valid_set,
        batch_size=cfg_node.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        seed=SEED
    )
    logits_list, labels_list = [], []
    for x, y, _ in loader:
        z = model(x.to(device)).float()
        logits_list.append(z.cpu())
        labels_list.append(y.cpu())
    return torch.cat(logits_list, 0), torch.cat(labels_list, 0)


def _find_T_star(logits, labels):
    """在验证集上搜索温度 T 以最小化 CE(logits/T, labels)"""
    T = torch.tensor(1.0, device=logits.device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    y = labels.long()

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits / T, y)
        loss.backward()
        return loss

    try:
        opt.step(closure)
    except Exception:
        pass
    return T.detach().clamp(min=1e-3, max=100.0)


@torch.no_grad()
def eval_prob_margin_stats(cfg_node, device, valid_set, eps_prob=(0.02, 0.05, 0.10), tau_prob=(0.5, 0.7, 0.8)):
    """
    概率口径的 margin：p_y - max_{c≠y} p_c（先做温度缩放）
    返回：mean_pmargin, near_boundary_ratios, safe_margin_coverages, per_class_rows, T*
    """
    logits, labels = _collect_logits_labels(cfg_node, device, valid_set)
    T_star = _find_T_star(logits.to(device), labels.to(device)).cpu()

    probs = torch.softmax(logits / T_star, dim=1)   # [N,C]
    y = labels.view(-1, 1)
    py = probs.gather(1, y).squeeze(1)              # [N]
    max_other = probs.scatter(1, y, -1).max(1).values
    pmargin = py - max_other                        # [-1,1]

    mean_pmargin = pmargin.mean().item()
    near_ratios = [(pmargin.abs() <= e).float().mean().item() for e in eps_prob]
    cover_ratios = [(pmargin >= t).float().mean().item() for t in tau_prob]

    C = probs.size(1)
    per_rows = []
    for c in range(C):
        m = (labels == c)
        if m.any():
            vals = pmargin[m]
            row = [int(c), int(m.sum().item()), f"{vals.mean().item():.6f}"] \
                  + [f"{((vals.abs() <= e).float().mean().item()):.6f}" for e in eps_prob] \
                  + [f"{((vals >= t).float().mean().item()):.6f}" for t in tau_prob]
        else:
            row = [int(c), 0, "0.000000"] + ["0.000000"] * (len(eps_prob) + len(tau_prob))
        per_rows.append(row)

    return mean_pmargin, near_ratios, cover_ratios, per_rows, T_star.item()


def to_ok(preds, labels):
    return (preds == labels).astype(np.int32)


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def infer_head_class_from_train(cfg_node):
    """
    基于训练集的类别频次推断头类。
    返回: head_cls(类ID), head_cnt(训练集中该类样本数), counts_train(各类训练频次数组)
    """
    train_set = build_dataset("train", cfg_node)
    annotations = train_set.get_annotations()
    num_classes = train_set.get_num_classes()
    num_class_list, _ = get_category_list(annotations, num_classes, cfg_node)
    head_cls = int(np.argmax(num_class_list))
    head_cnt = int(num_class_list[head_cls])
    return head_cls, head_cnt, np.array(num_class_list, dtype=int)


def main():
    ap = argparse.ArgumentParser("Compare baseline/fcc/fccr with three cfgs")
    ap.add_argument("--cfg_base", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_baseline_res32.yaml", type=str)
    ap.add_argument("--cfg_fcc", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_fcc_3_start1_resnet32.yaml", type=str)
    ap.add_argument("--cfg_fccr", required=False,
                    default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_fccr_3_start1_resnet32.yaml", type=str)
    ap.add_argument("--gpus", required=False, default="0", type=str)

    ap.add_argument("--gamma", type=float, default=3,
                    help="仅用于命名标记：写入输出文件名/文件夹，例如 0.1 -> gamma_0p1",
                    required=False)

    ap.add_argument("--head_cls", type=int, default=None,
                    help="强制指定头类ID；若不提供，则默认从训练集频次推断（推荐）。")
    ap.add_argument("--mid_cls", type=int, default=1, help="中类ID")
    ap.add_argument("--tail_cls", type=int, default=2, help="尾类ID（用于 Tail margin 统计）")

    ap.add_argument("--tail_hist_bins", type=int, default=40, help="Tail logit margin 直方图 bins")
    args = ap.parse_args()

    def _gamma_tag(g):
        if g is None:
            return None
        s = str(g).replace(".", "p")
        return f"gamma_{s}"

    tag = _gamma_tag(args.gamma)
    suffix = f"_{tag}" if tag is not None else ""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RED = "\033[31m"
    RESET = "\033[0m"

    def emit(msg: str):
        print(f"{RED}{msg}{RESET}")

    # 读取三份 cfg
    cfg_base = load_cfg(args.cfg_base)
    cfg_fcc = load_cfg(args.cfg_fcc)
    cfg_fccr = load_cfg(args.cfg_fccr)

    logger, _ = valid_logger(cfg_base)

    out_dir = os.path.join(
        cfg_base.OUTPUT_DIR, cfg_base.NAME, "valid",
        "compare_3cfg" + (f"_{tag}" if tag is not None else "")
    )
    os.makedirs(out_dir, exist_ok=True)

    # 验证集：用 baseline 的 cfg 构造一份（确保样本顺序一致）
    valid_set = build_dataset("valid", cfg_base)

    # 三个模型分别推理（各自用自己的 TEST 参数）
    preds_base, labels, image_ids = infer_one_cfg(cfg_base, device, valid_set)
    preds_fcc, labels2, _ = infer_one_cfg(cfg_fcc, device, valid_set)
    preds_fccr, labels3, _ = infer_one_cfg(cfg_fccr, device, valid_set)

    assert np.array_equal(labels, labels2) and np.array_equal(labels, labels3), \
        "三次推理的标签不一致（请确保三者使用同一验证集划分，且 DataLoader 不打乱）"

    num_classes = valid_set.get_num_classes()
    counts_val = np.bincount(labels.astype(int), minlength=num_classes)

    # 头/中/尾类 ID
    if args.head_cls is not None:
        head_cls = int(args.head_cls)
        logger.info(f"[HEAD] forced by args: head_cls={head_cls} | valid_count={int(counts_val[head_cls])}")
    else:
        head_cls, head_cnt_train, counts_train = infer_head_class_from_train(cfg_base)
        logger.info(f"[HEAD] by TRAIN freq: head_cls={head_cls} | train_count={int(head_cnt_train)} | "
                    f"valid_count={int(counts_val[head_cls])} | train_counts={counts_train.tolist()} | "
                    f"valid_counts={counts_val.tolist()}")
    mid_cls = int(args.mid_cls)
    tail_cls = int(args.tail_cls)
    assert all(c in range(num_classes) for c in [head_cls, mid_cls, tail_cls]), \
        "head/mid/tail 类别ID越界，请检查。"

    # =========================
    # A) 三者：logit margin 全局均值 + near-boundary mass
    # =========================
    eps_list = (0.0, 0.2, 0.5)
    mm_base, band_base, per_rows_mm_base = eval_margin_stats(cfg_base, device, valid_set, eps_list)
    mm_fcc, band_fcc, per_rows_mm_fcc = eval_margin_stats(cfg_fcc, device, valid_set, eps_list)
    mm_fccr, band_fccr, per_rows_mm_fccr = eval_margin_stats(cfg_fccr, device, valid_set, eps_list)

    emit("===== Logit-margin (true-class) & Near-Boundary Mass =====")
    emit(f"Mean margin:  Base={mm_base:.6f} | FCC={mm_fcc:.6f} | FCCR={mm_fccr:.6f}")
    emit("Near-boundary ratio (|margin|<=eps):")
    for eps, r0, r1, r2 in zip(eps_list, band_base, band_fcc, band_fccr):
        emit(f"  eps={eps:.1f}: Base={r0:.6f} | FCC={r1:.6f} | FCCR={r2:.6f}")
    emit("===========================================================")

    save_csv(os.path.join(out_dir, f"margin_per_class_BASE{suffix}.csv"),
             ["class_id", "count", "mean_margin"] + [f"M_eps{e}" for e in eps_list],
             per_rows_mm_base)
    save_csv(os.path.join(out_dir, f"margin_per_class_FCC{suffix}.csv"),
             ["class_id", "count", "mean_margin"] + [f"M_eps{e}" for e in eps_list],
             per_rows_mm_fcc)
    save_csv(os.path.join(out_dir, f"margin_per_class_FCCR{suffix}.csv"),
             ["class_id", "count", "mean_margin"] + [f"M_eps{e}" for e in eps_list],
             per_rows_mm_fccr)

    # =========================
    # B) Tail logit margin：分布统计 + 5% 分位数（tail=label 2 或 args.tail_cls）
    # =========================
    ts_base = eval_tail_logit_margin_stats(cfg_base, device, valid_set, tail_cls=tail_cls, hist_bins=args.tail_hist_bins)
    ts_fcc = eval_tail_logit_margin_stats(cfg_fcc, device, valid_set, tail_cls=tail_cls, hist_bins=args.tail_hist_bins)
    ts_fccr = eval_tail_logit_margin_stats(cfg_fccr, device, valid_set, tail_cls=tail_cls, hist_bins=args.tail_hist_bins)

    emit("===== Tail logit-margin distribution (true-class margin) =====")
    def _print_tail(tag, d):
        emit(
            f"{tag}: tail={d['tail_cls']} n={d['count']} | mean={d['mean']:.6f} std={d['std']:.6f} "
            f"| min={d['min']:.6f} p05={d['p05']:.6f} p50={d['p50']:.6f} p95={d['p95']:.6f} max={d['max']:.6f}"
        )
    _print_tail("Baseline", ts_base)
    _print_tail("FCC     ", ts_fcc)
    _print_tail("FCCR    ", ts_fccr)
    emit("=============================================================")

    save_csv(
        os.path.join(out_dir, f"tail_logit_margin_stats{suffix}.csv"),
        ["model", "tail_cls", "count", "mean", "std", "min", "p05", "p25", "p50", "p75", "p95", "max"],
        [
            ["Baseline", tail_cls, ts_base["count"],
             f"{ts_base['mean']:.6f}", f"{ts_base['std']:.6f}", f"{ts_base['min']:.6f}",
             f"{ts_base['p05']:.6f}", f"{ts_base['p25']:.6f}", f"{ts_base['p50']:.6f}",
             f"{ts_base['p75']:.6f}", f"{ts_base['p95']:.6f}", f"{ts_base['max']:.6f}"],
            ["FCC", tail_cls, ts_fcc["count"],
             f"{ts_fcc['mean']:.6f}", f"{ts_fcc['std']:.6f}", f"{ts_fcc['min']:.6f}",
             f"{ts_fcc['p05']:.6f}", f"{ts_fcc['p25']:.6f}", f"{ts_fcc['p50']:.6f}",
             f"{ts_fcc['p75']:.6f}", f"{ts_fcc['p95']:.6f}", f"{ts_fcc['max']:.6f}"],
            ["FCCR", tail_cls, ts_fccr["count"],
             f"{ts_fccr['mean']:.6f}", f"{ts_fccr['std']:.6f}", f"{ts_fccr['min']:.6f}",
             f"{ts_fccr['p05']:.6f}", f"{ts_fccr['p25']:.6f}", f"{ts_fccr['p50']:.6f}",
             f"{ts_fccr['p75']:.6f}", f"{ts_fccr['p95']:.6f}", f"{ts_fccr['max']:.6f}"],
        ],
    )

    def _dump_hist(tag_name, d):
        if d["hist_edges"] is None:
            return
        edges = d["hist_edges"]
        cnts = d["hist_counts"]
        rows = []
        for i in range(len(cnts)):
            rows.append([f"{edges[i]:.6f}", f"{edges[i+1]:.6f}", int(cnts[i])])
        save_csv(
            os.path.join(out_dir, f"tail_margin_hist_{tag_name}{suffix}.csv"),
            ["bin_left", "bin_right", "count"],
            rows
        )

    _dump_hist("BASE", ts_base)
    _dump_hist("FCC", ts_fcc)
    _dump_hist("FCCR", ts_fccr)

    # =========================
    # C) 你原本的整体性能与事件统计（保持不动）
    # =========================
    ok_base = to_ok(preds_base, labels)
    ok_fcc = to_ok(preds_fcc, labels)
    ok_fccr = to_ok(preds_fccr, labels)

    mask_base_right = (ok_base == 1)
    base_right_fcc_wrong = np.where(mask_base_right & (ok_fcc == 0))[0]
    base_right_fccr_wrong = np.where(mask_base_right & (ok_fccr == 0))[0]
    fcc_wrong_fccr_correct = np.where((ok_fcc == 0) & (ok_fccr == 1))[0]
    fcc_right_fccr_wrong = np.where((ok_fcc == 1) & (ok_fccr == 0))[0]

    def macro_f1(pred):
        _, _, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
        return f1

    macro_base, macro_fcc, macro_fccr = macro_f1(preds_base), macro_f1(preds_fcc), macro_f1(preds_fccr)

    top1_base = ok_base.mean() * 100.0
    top1_fcc = ok_fcc.mean() * 100.0
    top1_fccr = ok_fccr.mean() * 100.0

    emit("===== Overall VALID (Top-1 | Macro-F1) =====")
    emit(f"Baseline : {top1_base:.2f}% | macro-F1 {macro_base:.6f}")
    emit(f"FCC      : {top1_fcc:.2f}% | macro-F1 {macro_fcc:.6f}  | Δ vs Base: {top1_fcc - top1_base:+.2f} pp, ΔF1 {macro_fcc - macro_base:+.6f}")
    emit(f"FCCR     : {top1_fccr:.2f}% | macro-F1 {macro_fccr:.6f} | Δ vs FCC : {top1_fccr - top1_fcc:+.2f} pp, ΔF1 {macro_fccr - macro_fcc:+.6f}")
    emit("============================================")

    def class_detail_and_gain(cls_id, cls_name):
        cls_mask = (labels == cls_id)
        gain = int(np.sum(ok_fccr[cls_mask]) - np.sum(ok_fcc[cls_mask]))
        cnt = int(np.sum(cls_mask))

        idx_all = np.where(cls_mask)[0]
        rows_all = [[int(image_ids[i]), int(labels[i]),
                     int(preds_base[i]), int(ok_base[i]),
                     int(preds_fcc[i]), int(ok_fcc[i]),
                     int(preds_fccr[i]), int(ok_fccr[i])]
                    for i in idx_all]
        save_csv(os.path.join(out_dir, f"{cls_name}_class_{cls_id}_detail{suffix}.csv"),
                 ["image_id", "label", "pred_base", "ok_base", "pred_fcc", "ok_fcc", "pred_fccr", "ok_fccr"],
                 rows_all)

        def dump_subset_for_class(global_idx, subname):
            idx = np.intersect1d(global_idx, idx_all)
            if idx.size == 0:
                return
            rows = [[int(image_ids[i]), int(labels[i]),
                     int(preds_base[i]), int(preds_fcc[i]), int(preds_fccr[i])]
                    for i in idx.tolist()]
            save_csv(os.path.join(out_dir, f"{cls_name}_class_{cls_id}__{subname}{suffix}.csv"),
                     ["image_id", "label", "pred_base", "pred_fcc", "pred_fccr"], rows)

        dump_subset_for_class(base_right_fcc_wrong, "baseline_right__fcc_wrong")
        dump_subset_for_class(base_right_fccr_wrong, "baseline_right__fccr_wrong")
        dump_subset_for_class(fcc_wrong_fccr_correct, "fcc_wrong__fccr_correct")
        dump_subset_for_class(fcc_right_fccr_wrong, "fcc_right__fccr_wrong")

        return gain, cnt

    head_gain, head_cnt = class_detail_and_gain(head_cls, "head")
    mid_gain, mid_cnt = class_detail_and_gain(mid_cls, "mid")
    tail_gain, tail_cnt = class_detail_and_gain(tail_cls, "tail")

    def pct(a, b):
        return 0.0 if b == 0 else 100.0 * a / b

    N = len(labels)
    head_gain_pct_in_head = pct(head_gain, head_cnt)
    head_gain_pct_total = pct(head_gain, N)
    mid_gain_pct_in_mid = pct(mid_gain, mid_cnt)
    mid_gain_pct_total = pct(mid_gain, N)
    tail_gain_pct_in_tail = pct(tail_gain, tail_cnt)
    tail_gain_pct_total = pct(tail_gain, N)

    emit(f"[Head class={head_cls}] gain={head_gain} | {head_gain_pct_in_head:.2f}% of head ({head_cnt}) | {head_gain_pct_total:.2f}% of all ({N})")
    emit(f"[Mid  class={mid_cls}]  gain={mid_gain}  | {mid_gain_pct_in_mid:.2f}% of mid ({mid_cnt})   | {mid_gain_pct_total:.2f}% of all ({N})")
    emit(f"[Tail class={tail_cls}] gain={tail_gain} | {tail_gain_pct_in_tail:.2f}% of tail ({tail_cnt}) | {tail_gain_pct_total:.2f}% of all ({N})")

    save_csv(
        os.path.join(out_dir, f"class_gain_stats{suffix}.csv"),
        ["class_name", "class_id", "count", "gain", "gain_pct_in_class", "gain_pct_in_all"],
        [
            ["head", head_cls, head_cnt, head_gain, f"{head_gain_pct_in_head:.2f}%", f"{head_gain_pct_total:.2f}%"],
            ["mid", mid_cls, mid_cnt, mid_gain, f"{mid_gain_pct_in_mid:.2f}%", f"{mid_gain_pct_total:.2f}%"],
            ["tail", tail_cls, tail_cnt, tail_gain, f"{tail_gain_pct_in_tail:.2f}%", f"{tail_gain_pct_total:.2f}%"],
        ]
    )

    n_base_right = int(mask_base_right.sum())
    n_fcc_wrong = int((ok_fcc == 0).sum())
    n_fcc_right = int((ok_fcc == 1).sum())

    n_b2f = int(len(base_right_fcc_wrong))
    n_b2r = int(len(base_right_fccr_wrong))
    n_fix = int(len(fcc_wrong_fccr_correct))
    n_break = int(len(fcc_right_fccr_wrong))

    b2f_src = pct(n_b2f, n_base_right)
    b2f_total = pct(n_b2f, N)
    b2r_src = pct(n_b2r, n_base_right)
    b2r_total = pct(n_b2r, N)
    fix_src = pct(n_fix, n_fcc_wrong)
    fix_total = pct(n_fix, N)
    break_src = pct(n_break, n_fcc_right)
    break_total = pct(n_break, N)

    emit(f"Baseline正确→FCC错误:  {n_b2f} | {b2f_src:.2f}% of baseline-correct ({n_base_right}) | {b2f_total:.2f}% of all ({N})")
    emit(f"Baseline正确→FCCR错误: {n_b2r} | {b2r_src:.2f}% of baseline-correct ({n_base_right}) | {b2r_total:.2f}% of all ({N})")
    emit(f"FCC错误→FCCR纠正:      {n_fix} | {fix_src:.2f}% of FCC-wrong ({n_fcc_wrong}) | {fix_total:.2f}% of all ({N})")
    emit(f"FCC正确→FCCR错误:      {n_break} | {break_src:.2f}% of FCC-right ({n_fcc_right}) | {break_total:.2f}% of all ({N})")

    save_csv(
        os.path.join(out_dir, f"event_stats{suffix}.csv"),
        ["event", "count", "src_den", "src_pct", "all_den", "all_pct"],
        [
            ["baseline_right_to_fcc_wrong", n_b2f, n_base_right, f"{b2f_src:.2f}%", N, f"{b2f_total:.2f}%"],
            ["baseline_right_to_fccr_wrong", n_b2r, n_base_right, f"{b2r_src:.2f}%", N, f"{b2r_total:.2f}%"],
            ["fcc_wrong_to_fccr_correct", n_fix, n_fcc_wrong, f"{fix_src:.2f}%", N, f"{fix_total:.2f}%"],
            ["fcc_right_to_fccr_wrong", n_break, n_fcc_right, f"{break_src:.2f}%", N, f"{break_total:.2f}%"],
            ["head_gain_fccr_minus_fcc", head_gain, head_cnt, f"{head_gain_pct_in_head:.2f}%", N, f"{head_gain_pct_total:.2f}%"],
        ]
    )

    # 每类 top1
    per_cls_rows = []
    for c in range(num_classes):
        m = (labels == c)
        cnt = int(m.sum())
        if cnt == 0:
            acc_b = acc_f = acc_r = 0.0
        else:
            acc_b = float(ok_base[m].mean())
            acc_f = float(ok_fcc[m].mean())
            acc_r = float(ok_fccr[m].mean())
        per_cls_rows.append([c, cnt, f"{acc_b * 100:.2f}%", f"{acc_f * 100:.2f}%", f"{acc_r * 100:.2f}%"])

    save_csv(os.path.join(out_dir, f"per_class_top1{suffix}.csv"),
             ["class_id", "count", "base_top1", "fcc_top1", "fccr_top1"],
             per_cls_rows)

    # 文本总结（加上 Tail p05）
    with open(os.path.join(out_dir, f"summary{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(f"[N={N}]\n")
        f.write(f"top1: base={top1_base:.2f}% | fcc={top1_fcc:.2f}% | fccr={top1_fccr:.2f}%\n")
        f.write(f"macro-F1: base={macro_base:.6f} | fcc={macro_fcc:.6f} | fccr={macro_fccr:.6f}\n\n")

        f.write("Logit-margin (true-class):\n")
        f.write(f"  mean: base={mm_base:.6f} | fcc={mm_fcc:.6f} | fccr={mm_fccr:.6f}\n")
        for eps, r0, r1, r2 in zip(eps_list, band_base, band_fcc, band_fccr):
            f.write(f"  near(|m|<= {eps:.1f}): base={r0:.6f} | fcc={r1:.6f} | fccr={r2:.6f}\n")
        f.write("\n")

        f.write(f"Tail(logit margin) distribution (tail_cls={tail_cls}):\n")
        f.write(f"  Baseline: n={ts_base['count']} mean={ts_base['mean']:.6f} p05={ts_base['p05']:.6f} p50={ts_base['p50']:.6f} p95={ts_base['p95']:.6f}\n")
        f.write(f"  FCC     : n={ts_fcc['count']} mean={ts_fcc['mean']:.6f} p05={ts_fcc['p05']:.6f} p50={ts_fcc['p50']:.6f} p95={ts_fcc['p95']:.6f}\n")
        f.write(f"  FCCR    : n={ts_fccr['count']} mean={ts_fccr['mean']:.6f} p05={ts_fccr['p05']:.6f} p50={ts_fccr['p50']:.6f} p95={ts_fccr['p95']:.6f}\n")
        f.write("\n")

        f.write("Per-class top1 on VALID:\n")
        for c, cnt, b, fcc, fccr in per_cls_rows:
            f.write(f"  class {c:>2d} (n={cnt:>4d}): base={b} | fcc={fcc} | fccr={fccr}\n")

    print(f"\n✅ Done. Results saved under: {out_dir}\n")


if __name__ == "__main__":
    main()
