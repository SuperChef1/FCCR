# compare_by_3cfg.py
import os, argparse, numpy as np, torch
torch.use_deterministic_algorithms(True, warn_only=True)
import torch.nn.functional as F
import _init_paths
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from types import SimpleNamespace

from config import cfg as default_cfg  # 默认 config 节点
from net import build_network
from utils.repro import setup_repro, make_seeded_loader
from utils.utils import valid_logger, get_category_list

# 你的数据集工厂
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

    # 模型路径：OUTPUT_DIR / NAME / models / TEST.MODEL_FILE
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
            # 若数据集未提供 image_id，就按顺序占位；因为复用同一 valid_set，顺序可对齐
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
    num_classes = valid_set.get_num_classes()
    model = build_network(cfg_node, mode="test", num_classes=num_classes)
    model.load_model(os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE),
                     tau_norm=bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False))
                     if hasattr(cfg_node.TEST, "TAU_NORM") else False,
                     tau=float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0))
                     if hasattr(cfg_node.TEST, "TAU_NORM") else 1.0)
    model = torch.nn.DataParallel(model).to(device).eval()

    loader = make_seeded_loader(valid_set, batch_size=cfg_node.TEST.BATCH_SIZE,
                                num_workers=1, shuffle=False, pin_memory=False, drop_last=False, seed=SEED)

    sums, cnt = 0.0, 0
    band = [0 for _ in eps_list]
    per_cls = {c: {"sum": 0.0, "cnt": 0, "band": [0 for _ in eps_list]} for c in range(num_classes)}

    for images, gt, _ in loader:
        images = images.to(device)
        y = gt.to(device).long()                  # [B]
        logits = model(images).float()            # [B,C]
        # 取“非真类”的最大 logit
        mask = F.one_hot(y, num_classes=logits.size(1)).bool()
        max_other = logits.masked_fill(mask, float("-inf")).max(dim=1).values
        margin = logits.gather(1, y.view(-1,1)).squeeze(1) - max_other  # [B]

        sums += margin.sum().item()
        cnt  += margin.numel()

        for i, eps in enumerate(eps_list):
            band[i] += int(((margin >= -eps) & (margin <= eps)).sum().item())

        # 分类累计
        for c in range(num_classes):
            m = (y == c)
            if m.any():
                mvals = margin[m]
                per_cls[c]["sum"] += float(mvals.sum().item())
                per_cls[c]["cnt"] += int(mvals.numel())
                for i, eps in enumerate(eps_list):
                    per_cls[c]["band"][i] += int(((mvals >= -eps) & (mvals <= eps)).sum().item())

    mean_margin = sums / max(1, cnt)
    band_ratio  = [b / max(1, cnt) for b in band]
    per_rows = []
    for c in range(num_classes):
        ccnt = per_cls[c]["cnt"]
        cmargin = per_cls[c]["sum"] / max(1, ccnt)
        cband = [x / max(1, ccnt) for x in per_cls[c]["band"]]
        per_rows.append([c, ccnt, f"{cmargin:.4f}"] + [f"{r:.4f}" for r in cband])
    return mean_margin, band_ratio, per_rows

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

    loader = make_seeded_loader(valid_set, batch_size=cfg_node.TEST.BATCH_SIZE,
                                num_workers=1, shuffle=False, pin_memory=False, drop_last=False, seed=SEED)
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

    # 全局统计
    mean_pmargin = pmargin.mean().item()
    near_ratios = [ (pmargin.abs() <= e).float().mean().item() for e in eps_prob ]   # |pm|<=eps 的比例
    cover_ratios = [ (pmargin >= t).float().mean().item() for t in tau_prob ]        # pm>=tau 的覆盖率

    # 按类统计
    C = probs.size(1)
    per_rows = []
    for c in range(C):
        m = (labels == c)
        if m.any():
            vals = pmargin[m]
            row = [int(c), int(m.sum().item()), f"{vals.mean().item():.4f}"] \
                  + [f"{((vals.abs() <= e).float().mean().item()):.4f}" for e in eps_prob] \
                  + [f"{((vals >= t).float().mean().item()):.4f}" for t in tau_prob]
        else:
            row = [int(c), 0, "0.0000"] + ["0.0000"]*(len(eps_prob)+len(tau_prob))
        per_rows.append(row)

    return mean_pmargin, near_ratios, cover_ratios, per_rows, T_star.item()

def to_ok(preds, labels):  # 逐样本命中(0/1)
    return (preds == labels).astype(np.int32)


def save_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)

def infer_head_class_from_train(cfg_node):
    """
    基于训练集的类别频次推断头类。
    返回: head_cls(类ID), head_cnt(训练集中该类样本数), counts_train(各类训练频次数组)
    """
    train_set = build_dataset("train", cfg_node)
    annotations = train_set.get_annotations()           # 你的 数据集类 已有该接口
    num_classes = train_set.get_num_classes()
    num_class_list, _ = get_category_list(annotations, num_classes, cfg_node)
    head_cls = int(np.argmax(num_class_list))
    head_cnt = int(num_class_list[head_cls])
    return head_cls, head_cnt, np.array(num_class_list, dtype=int)

def infer_head_class_from_val(labels, num_classes):
    """
    基于验证集 labels 统计每类数量，返回：
    head_cls: 最多样本的类ID
    head_cnt: 该类在验证集中的样本数
    counts:   各类计数数组
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    head_cls = int(np.argmax(counts))
    head_cnt = int(counts[head_cls])
    return head_cls, head_cnt, counts

def main():
    ap = argparse.ArgumentParser("Compare baseline/fcc/fccr with three cfgs")
    ap.add_argument("--cfg_base",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_baseline_res32.yaml",
        type=str,)
    ap.add_argument("--cfg_fcc",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_fcc_3_start1_resnet32.yaml",
        type=str,)
    ap.add_argument("--cfg_fccr",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\Resnet32\LAM_fccr_3_start1_resnet32.yaml",
        type=str,)
    ap.add_argument("--gpus",
        help="decide which gpus to use",
        required=False,
        default='0',
        type=str,)
    ap.add_argument("--gamma", type=float, default=3,
                    help="仅用于命名标记：将当前比较实验的 gamma 值写入输出文件名/文件夹，例如 0.1 -> gamma_0p1",
                    required = False
    )

    ap.add_argument("--head_cls", type=int, default=None,
                    help="强制指定头类ID；若不提供，则默认从训练集频次推断（推荐）。")
    ap.add_argument("--mid_cls", type=int, default=1, help="中类ID（验证集上统计/导出用）")
    ap.add_argument("--tail_cls", type=int, default=2, help="尾类ID（验证集上统计/导出用）")

    args = ap.parse_args()
    def _gamma_tag(g):
        if g is None:
            return None
        # 0.1 -> gamma_0p1, 2 -> gamma_2
        s = str(g).replace('.', 'p')
        return f"gamma_{s}"

    tag = _gamma_tag(args.gamma)
    suffix = f"_{tag}" if tag is not None else ""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    device = torch.device("cuda")
    RED = "\033[31m";
    RESET = "\033[0m"

    def emit(msg: str):
        print(f"{RED}{msg}{RESET}")
    # 读取三份 cfg
    cfg_base  = load_cfg(args.cfg_base)
    cfg_fcc   = load_cfg(args.cfg_fcc)
    cfg_fccr  = load_cfg(args.cfg_fccr)

    # 日志与输出目录（以 baseline 的 OUTPUT_DIR/NAME 为基准）
    logger, _ = valid_logger(cfg_base)



    # 原 out_dir = os.path.join(cfg_base.OUTPUT_DIR, cfg_base.NAME, "valid", "compare_3cfg")
    # 改为（带上 gamma 子目录，便于分组管理）
    out_dir = os.path.join(
        cfg_base.OUTPUT_DIR, cfg_base.NAME, "valid",
        "compare_3cfg" + (f"_{tag}" if tag is not None else "")
    )
    os.makedirs(out_dir, exist_ok=True)

    # 验证集：用 baseline 的 cfg 构造一份（确保样本顺序一致）
    valid_set = build_dataset("valid", cfg_base)


    # 三个模型分别推理（各自用自己的 TEST 参数）
    preds_base, labels, image_ids = infer_one_cfg(cfg_base, device, valid_set)
    preds_fcc,  labels2, _        = infer_one_cfg(cfg_fcc,  device, valid_set)
    preds_fccr, labels3, _        = infer_one_cfg(cfg_fccr, device, valid_set)

    # —— 边界相关指标：FCC vs FCCR ——
    eps_list = (0.0, 0.2, 0.5)
    mm_fcc, band_fcc, per_rows_mm_fcc = eval_margin_stats(cfg_fcc, device, valid_set, eps_list)
    mm_fccr, band_fccr, per_rows_mm_fccr = eval_margin_stats(cfg_fccr, device, valid_set, eps_list)

    emit("===== Margin & Near-Boundary Mass =====")
    emit(f"Mean margin:  FCC={mm_fcc:.4f} | FCCR={mm_fccr:.4f} | Δ={mm_fccr - mm_fcc:+.4f}")
    emit("Near-boundary ratio (|margin|<=eps):")
    for eps, r1, r2 in zip(eps_list, band_fcc, band_fccr):
        emit(f"  eps={eps:.1f}: FCC={r1:.4f} | FCCR={r2:.4f} | Δ={r2 - r1:+.4f}")
    emit("========================================")

    save_csv(os.path.join(out_dir, f"margin_per_class_FCC{suffix}.csv"),
             ["class_id", "count", "mean_margin"] + [f"M_eps{e}" for e in eps_list], per_rows_mm_fcc)
    save_csv(os.path.join(out_dir, f"margin_per_class_FCCR{suffix}.csv"),
             ["class_id", "count", "mean_margin"] + [f"M_eps{e}" for e in eps_list], per_rows_mm_fccr)

    eps_prob = (0.02, 0.05, 0.10)
    tau_prob = (0.50, 0.70, 0.80)

    pm_fcc, nb_fcc, cov_fcc, per_rows_pm_fcc, T_fcc = eval_prob_margin_stats(cfg_fcc, device, valid_set, eps_prob,
                                                                             tau_prob)
    pm_fccr, nb_fccr, cov_fccr, per_rows_pm_fccr, T_fccr = eval_prob_margin_stats(cfg_fccr, device, valid_set, eps_prob,
                                                                                  tau_prob)

    emit("===== Calibrated Prob-Margin (after T-scaling) =====")
    emit(f"T*: FCC={T_fcc:.3f} | FCCR={T_fccr:.3f}")
    emit(f"Mean p-margin: FCC={pm_fcc:.4f} | FCCR={pm_fccr:.4f} | Δ={pm_fccr - pm_fcc:+.4f}")
    emit("Near-boundary mass (|pmargin|<=eps):")
    for e, r1, r2 in zip(eps_prob, nb_fcc, nb_fccr):
        emit(f"  eps={e:.2f}: FCC={r1 * 100:.2f}% | FCCR={r2 * 100:.2f}% | Δ={(r2 - r1) * 100:+.2f} pp")
    emit("Safe-margin coverage (pmargin>=τ):")
    for t, r1, r2 in zip(tau_prob, cov_fcc, cov_fccr):
        emit(f"  τ={t:.2f}: FCC={r1 * 100:.2f}% | FCCR={r2 * 100:.2f}% | Δ={(r2 - r1) * 100:+.2f} pp")
    emit("========================================")

    # 导出 CSV（概率口径）
    save_csv(os.path.join(out_dir, f"prob_margin_per_class_FCC{suffix}.csv"),
             ["class_id", "count", "mean_pmargin"] + [f"NB_eps{e}" for e in eps_prob] + [f"SMC_tau{t}" for t in
                                                                                         tau_prob],
             per_rows_pm_fcc)
    save_csv(os.path.join(out_dir, f"prob_margin_per_class_FCCR{suffix}.csv"),
             ["class_id", "count", "mean_pmargin"] + [f"NB_eps{e}" for e in eps_prob] + [f"SMC_tau{t}" for t in
                                                                                         tau_prob],
             per_rows_pm_fccr)

    # 标签一致性校验
    assert np.array_equal(labels, labels2) and np.array_equal(labels, labels3), \
        "三次推理的标签不一致（请确保三者使用同一验证集划分，且 DataLoader 不打乱）"
    num_classes = valid_set.get_num_classes()

    # 仅用于日志：记录验证集各类计数（不用于决定头类）
    counts_val = np.bincount(labels.astype(int), minlength=num_classes)

    if args.head_cls is not None:
        # 显式指定头类（例如 --head_cls 0）
        head_cls = int(args.head_cls)
        head_cnt_train = None
        logger.info(f"[HEAD] forced by args: head_cls={head_cls} | valid_count={int(counts_val[head_cls])}")
    else:
        # 默认：使用训练集频次决定头类
        head_cls, head_cnt_train, counts_train = infer_head_class_from_train(cfg_base)
        logger.info(f"[HEAD] by TRAIN freq: head_cls={head_cls} | train_count={int(head_cnt_train)} | "
                    f"valid_count={int(counts_val[head_cls])} | train_counts={counts_train.tolist()} | "
                    f"valid_counts={counts_val.tolist()}")
    mid_cls = int(args.mid_cls)
    tail_cls = int(args.tail_cls)
    assert all(c in range(num_classes) for c in [head_cls, mid_cls, tail_cls]), \
        "head/mid/tail 类别ID越界，请检查。"

    # 命中向量
    ok_base = to_ok(preds_base, labels)
    ok_fcc  = to_ok(preds_fcc,  labels)
    ok_fccr = to_ok(preds_fccr, labels)

    # (1) baseline 正确 → 在 fcc / fccr 错
    mask_base_right = (ok_base == 1)
    base_right_fcc_wrong  = np.where(mask_base_right & (ok_fcc  == 0))[0]
    base_right_fccr_wrong = np.where(mask_base_right & (ok_fccr == 0))[0]

    # (2) fcc 错 → fccr 纠正；fcc 对 → fccr 打错
    fcc_wrong_fccr_correct = np.where((ok_fcc == 0) & (ok_fccr == 1))[0]
    fcc_right_fccr_wrong   = np.where((ok_fcc == 1) & (ok_fccr == 0))[0]

    # macro-F1
    def macro_f1(pred):
        p, r, f1, _ = precision_recall_fscore_support(labels, pred, average="macro", zero_division=0)
        return f1
    macro_base, macro_fcc, macro_fccr = macro_f1(preds_base), macro_f1(preds_fcc), macro_f1(preds_fccr)

    top1_base = ok_base.mean() * 100.0
    top1_fcc = ok_fcc.mean() * 100.0
    top1_fccr = ok_fccr.mean() * 100.0

    emit("===== Overall VALID (Top-1 | Macro-F1) =====")
    emit(f"Baseline : {top1_base:.2f}% | macro-F1 {macro_base:.4f}")
    emit(
        f"FCC      : {top1_fcc:.2f}% | macro-F1 {macro_fcc:.4f}  | Δ vs Base: {top1_fcc - top1_base:+.2f} pp, ΔF1 {macro_fcc - macro_base:+.4f}")
    emit(
        f"FCCR     : {top1_fccr:.2f}% | macro-F1 {macro_fccr:.4f} | Δ vs FCC : {top1_fccr - top1_fcc:+.2f} pp, ΔF1 {macro_fccr - macro_fcc:+.4f}")
    emit("============================================")

    def class_detail_and_gain(cls_id, cls_name):
        """
        导出某一类的逐样本明细，并计算该类上 FCCR 相对 FCC 的净增益（判对数差值）。
        也额外导出该类在四个关注集合中的样本明细，便于诊断。
        """
        cls_mask = (labels == cls_id)

        # 该类的净增益：FCCR判对 - FCC判对
        gain = int(np.sum(ok_fccr[cls_mask]) - np.sum(ok_fcc[cls_mask]))
        cnt = int(np.sum(cls_mask))

        # 该类的逐样本总表（含 base/fcc/fccr 的预测与是否命中）
        idx_all = np.where(cls_mask)[0]
        rows_all = [[int(image_ids[i]), int(labels[i]),
                     int(preds_base[i]), int(ok_base[i]),
                     int(preds_fcc[i]), int(ok_fcc[i]),
                     int(preds_fccr[i]), int(ok_fccr[i])]
                    for i in idx_all]
        save_csv(os.path.join(out_dir, f"{cls_name}_class_{cls_id}_detail{suffix}.csv"),
                 ["image_id", "label", "pred_base", "ok_base", "pred_fcc", "ok_fcc", "pred_fccr", "ok_fccr"], rows_all)

        # 该类在四个关注集合中的细分（便于定位“被谁打错/纠正”）
        def dump_subset_for_class(global_idx, subname):
            idx = np.intersect1d(global_idx, idx_all)
            if idx.size == 0:  # 没有样本就不写
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

    # —— 对三类分别做统计与导出 ——
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

    # 也写进日志 / 控制台
    emit(
        f"[Head class={head_cls}] gain={head_gain} | {head_gain_pct_in_head:.2f}% of head ({head_cnt}) | {head_gain_pct_total:.2f}% of all ({N})")
    emit(
        f"[Mid  class={mid_cls}]  gain={mid_gain}  | {mid_gain_pct_in_mid:.2f}% of mid ({mid_cnt})   | {mid_gain_pct_total:.2f}% of all ({N})")
    emit(
        f"[Tail class={tail_cls}] gain={tail_gain} | {tail_gain_pct_in_tail:.2f}% of tail ({tail_cnt}) | {tail_gain_pct_total:.2f}% of all ({N})")

    # 可选：导出一张 class_gain_stats.csv
    save_csv(
        os.path.join(out_dir, f"class_gain_stats{suffix}.csv"),
        ["class_name", "class_id", "count", "gain", "gain_pct_in_class", "gain_pct_in_all"],
        [
            ["head", head_cls, head_cnt, head_gain, f"{head_gain_pct_in_head:.2f}%", f"{head_gain_pct_total:.2f}%"],
            ["mid", mid_cls, mid_cnt, mid_gain, f"{mid_gain_pct_in_mid:.2f}%", f"{mid_gain_pct_total:.2f}%"],
            ["tail", tail_cls, tail_cnt, tail_gain, f"{tail_gain_pct_in_tail:.2f}%", f"{tail_gain_pct_total:.2f}%"],
        ]
    )

    # —— 事件计数 + 占比（源集合/全体） ——
    N = len(labels)
    n_base_right = int(mask_base_right.sum())
    n_fcc_wrong = int((ok_fcc == 0).sum())
    n_fcc_right = int((ok_fcc == 1).sum())

    n_b2f = int(len(base_right_fcc_wrong))  # Baseline正→FCC错
    n_b2r = int(len(base_right_fccr_wrong))  # Baseline正→FCCR错
    n_fix = int(len(fcc_wrong_fccr_correct))  # FCC错→FCCR纠正
    n_break = int(len(fcc_right_fccr_wrong))  # FCC对→FCCR打错

    def pct(a, b):  # 百分比
        return 0.0 if b == 0 else 100.0 * a / b

    # 两种口径：源集合占比 & 全体占比
    b2f_src = pct(n_b2f, n_base_right)
    b2f_total = pct(n_b2f, N)

    b2r_src = pct(n_b2r, n_base_right)
    b2r_total = pct(n_b2r, N)

    fix_src = pct(n_fix, n_fcc_wrong)
    fix_total = pct(n_fix, N)

    break_src = pct(n_break, n_fcc_right)
    break_total = pct(n_break, N)

    # 头类净增益占比（对头类样本计数）及对全体的占比
    head_gain_pct_in_head = pct(head_gain, head_cnt if 'head_cnt' in locals() else int((labels == head_cls).sum()))
    head_gain_pct_total = pct(head_gain, N)

    # 打印到 logger 和控制台
    emit(
        f"Baseline正确→FCC错误:  {n_b2f} | {b2f_src:.2f}% of baseline-correct ({n_base_right}) | {b2f_total:.2f}% of all ({N})")
    emit(
        f"Baseline正确→FCCR错误: {n_b2r} | {b2r_src:.2f}% of baseline-correct ({n_base_right}) | {b2r_total:.2f}% of all ({N})")
    emit(
        f"FCC错误→FCCR纠正:      {n_fix} | {fix_src:.2f}% of FCC-wrong ({n_fcc_wrong}) | {fix_total:.2f}% of all ({N})")
    emit(
        f"FCC正确→FCCR错误:      {n_break} | {break_src:.2f}% of FCC-right ({n_fcc_right}) | {break_total:.2f}% of all ({N})")

    # 导出成 CSV
    save_csv(
        os.path.join(out_dir, f"event_stats{suffix}.csv"),
        ["event", "count", "src_den", "src_pct", "all_den", "all_pct"],
        [
            ["baseline_right_to_fcc_wrong", n_b2f, n_base_right, f"{b2f_src:.2f}%", N, f"{b2f_total:.2f}%"],
            ["baseline_right_to_fccr_wrong", n_b2r, n_base_right, f"{b2r_src:.2f}%", N, f"{b2r_total:.2f}%"],
            ["fcc_wrong_to_fccr_correct", n_fix, n_fcc_wrong, f"{fix_src:.2f}%", N, f"{fix_total:.2f}%"],
            ["fcc_right_to_fccr_wrong", n_break, n_fcc_right, f"{break_src:.2f}%", N, f"{break_total:.2f}%"],
            ["head_gain_fccr_minus_fcc", head_gain, head_cnt, f"{head_gain_pct_in_head:.2f}%", N,
             f"{head_gain_pct_total:.2f}%"],
        ]
    )

    # —— 每类 top1 准确率统计并导出/打印 ——
    per_cls_rows = []
    per_cls_log_lines = []
    for c in range(num_classes):
        m = (labels == c)
        cnt = int(m.sum())
        if cnt == 0:
            acc_b = acc_f = acc_r = 0.0
        else:
            acc_b = float(ok_base[m].mean())
            acc_f = float(ok_fcc[m].mean())
            acc_r = float(ok_fccr[m].mean())
        per_cls_rows.append([c, cnt,
                             f"{acc_b * 100:.2f}%", f"{acc_f * 100:.2f}%", f"{acc_r * 100:.2f}%"])
        per_cls_log_lines.append(
            f"  class {c:>2d} (n={cnt:>4d}): base={acc_b * 100:.2f}% | "
            f"fcc={acc_f * 100:.2f}% | fccr={acc_r * 100:.2f}%"
        )

    # 保存 CSV
    save_csv(os.path.join(out_dir, f"per_class_top1{suffix}.csv"),
             ["class_id", "count", "base_top1", "fcc_top1", "fccr_top1"],
             per_cls_rows)

    # 打印到 logger 和控制台
    emit("Per-class top1 on VALID:")
    for line in per_cls_log_lines:
        emit(line)


    # 文本总结
    with open(os.path.join(out_dir, f"summary{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(f"[N={len(labels)}]\n")
        f.write(
            f"top1: base={ok_base.mean() * 100:.2f}% | fcc={ok_fcc.mean() * 100:.2f}% | fccr={ok_fccr.mean() * 100:.2f}%\n")
        f.write(f"macro-F1: base={macro_base:.4f} | fcc={macro_fcc:.4f} | fccr={macro_fccr:.4f}\n")
        f.write(f"Baseline正确→FCC错误:  {n_b2f} | "
                f"{b2f_src:.2f}% of baseline-correct ({n_base_right}) | {b2f_total:.2f}% of all ({N})\n")
        f.write(f"Baseline正确→FCCR错误: {n_b2r} | "
                f"{b2r_src:.2f}% of baseline-correct ({n_base_right}) | {b2r_total:.2f}% of all ({N})\n")
        f.write(f"FCC错误→FCCR纠正:      {n_fix} | "
                f"{fix_src:.2f}% of FCC-wrong ({n_fcc_wrong}) | {fix_total:.2f}% of all ({N})\n")
        f.write(f"FCC正确→FCCR错误:      {n_break} | "
                f"{break_src:.2f}% of FCC-right ({n_fcc_right}) | {break_total:.2f}% of all ({N})\n")
        f.write("\n")
        f.write(f"[Head class={head_cls}] count={head_cnt} | gain={head_gain} | "
                f"{head_gain_pct_in_head:.2f}% of head | {head_gain_pct_total:.2f}% of all\n")
        f.write(f"[Mid  class={mid_cls}]  count={mid_cnt}  | gain={mid_gain}  | "
                f"{mid_gain_pct_in_mid:.2f}% of mid  | {mid_gain_pct_total:.2f}% of all\n")
        f.write(f"[Tail class={tail_cls}] count={tail_cnt} | gain={tail_gain} | "
                f"{tail_gain_pct_in_tail:.2f}% of tail | {tail_gain_pct_total:.2f}% of all\n")
        f.write("\nPer-class top1 on VALID:\n")
        for c, cnt, b, fcc, fccr in per_cls_rows:
            f.write(f"  class {c:>2d} (n={cnt:>4d}): base={b} | fcc={fcc} | fccr={fccr}\n")

    print(f"\n✅ Done. Results saved under: {out_dir}\n")


if __name__ == "__main__":
    main()
