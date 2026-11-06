# compare_by_3cfg.py
import torch.nn.functional as F
import torch.nn as nn
import os, argparse, numpy as np, torch
torch.use_deterministic_algorithms(True, warn_only=True)
import _init_paths
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support
from types import SimpleNamespace

from config import cfg as default_cfg  # 默认 config 节点
from net import Network
from utils.repro import setup_repro, make_seeded_loader
from utils.utils import valid_logger, get_category_list

# 你的数据集工厂
from dataset.lam_dataset import LAMDataset
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
    model = Network(model_cfg, mode="test", num_classes=num_classes)

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

def build_tau_vector(num_classes, class_counts, gamma=0.1, start_from_1=True):
    """
    依据训练时的等差压缩思路生成每类 τ_c。
    - class_counts: 长度 C 的训练集频次（可用你已有的 counts_train）
    - gamma: 压缩强度（与训练保持一致）
    - start_from_1: 若 True, rank 从 1..C；否则 0..C-1
    你若在训练期用了自定义 τ 公式，这里替换成一致的实现即可。
    """
    counts = np.asarray(class_counts).astype(float)
    # 频次大的 rank=1（头）…… 频次小的 rank=C（尾）
    order = np.argsort(-counts)
    rank = np.empty_like(order)
    rank[order] = np.arange(1, num_classes + 1) if start_from_1 else np.arange(num_classes)

    if start_from_1:
        # 论文里常见等差：τ_c = 1 + γ * (1 - (rank-1)/(C-1))
        denom = max(1, num_classes - 1)
        tau = 1.0 + gamma * (1.0 - (rank - 1) / denom)
    else:
        denom = max(1, num_classes - 1)
        tau = 1.0 + gamma * (1.0 - rank / denom)

    return torch.tensor(tau, dtype=torch.float32)


@torch.no_grad()
def eval_misclassified_area(model, valid_set, device, tau_vec, feature_getter=None, head_forward=None):
    model.eval()

    if feature_getter is None:
        assert hasattr(model.module, "backbone"), "请传入 feature_getter 或在 Network 中暴露 .backbone"
        feature_getter = lambda x: model.module.backbone(x)
    if head_forward is None:
        assert hasattr(model.module, "classifier"), "请传入 head_forward 或在 Network 中暴露 .classifier"
        head_forward = lambda f: model.module.classifier(f)

    def _get_cls_in_features(classifier: nn.Module):
        # 拿到分类头里第一个 Linear 的 in_features
        if isinstance(classifier, nn.Linear):
            return classifier.in_features
        for m in classifier.modules():
            if isinstance(m, nn.Linear):
                return m.in_features
        return None  # 没找到 Linear，就走“尽力而为”模式

    def _align_feat_to_classifier(feat, classifier):
        """
        将任意形状的 feat 对齐到 classifier 的输入维度：
        1) 先展平 [B, -1] 看是否等于 in_features
        2) 再 GAP (1d/2d) 看是否等于 in_features
        3) 两者都不行则报错提示实际尺寸
        """
        B = feat.size(0)
        exp = _get_cls_in_features(classifier)

        # 方案A：展平
        flat = feat.view(B, -1)
        if (exp is None) or (flat.size(1) == exp):
            return flat

        # 方案B：GAP 再展平
        if feat.dim() == 4:  # [B,C,H,W]
            pooled = F.adaptive_avg_pool2d(feat, 1).view(B, -1)
        elif feat.dim() == 3:  # [B,C,L]
            pooled = F.adaptive_avg_pool1d(feat, 1).view(B, -1)
        elif feat.dim() == 2:  # [B,D]
            pooled = feat
        else:
            pooled = flat

        if pooled.size(1) == exp:
            return pooled

        raise RuntimeError(
            f"Classifier expects in_features={exp}, "
            f"but got flat={flat.size(1)}, pooled={pooled.size(1)}. "
            f"请定制 feature_getter 或修改对齐逻辑。"
        )

    loader = make_seeded_loader(valid_set, batch_size=64, num_workers=1,
                                shuffle=False, pin_memory=False, drop_last=False, seed=SEED)

    nmf = nof = 0
    num_classes = valid_set.get_num_classes()
    per_cls = {c: {"nmf": 0, "nof": 0, "cnt": 0} for c in range(num_classes)}

    for images, gt, _ in loader:
        images = images.to(device)
        y = gt.to(device).long()               # [B]

        f_O_raw = feature_getter(images)  # 可能是 [B,C,H,W] / [B,C,L] / [B,D]
        f_O = _align_feat_to_classifier(f_O_raw, model.module.classifier)  # -> [B, in_features]
        tau_y = tau_vec.to(device)[y].view(-1, 1).type_as(f_O)  # [B,1]
        f_M = f_O * tau_y
        pred_O = head_forward(f_O).argmax(1)
        pred_M = head_forward(f_M).argmax(1)

        ok_M = (pred_M == y)
        ok_O = (pred_O == y)

        nmf += ok_M.sum().item()
        nof += ((~ok_O) & ok_M).sum().item()

        for c in range(num_classes):
            m = (y == c)
            if m.any():
                per_cls[c]["cnt"] += int(m.sum().item())
                per_cls[c]["nmf"] += int(ok_M[m].sum().item())
                per_cls[c]["nof"] += int(((~ok_O) & ok_M)[m].sum().item())

    ratio = nof / max(1, nmf)
    per_rows = []
    for c in range(num_classes):
        cnt = per_cls[c]["cnt"]
        nmf_c = per_cls[c]["nmf"]
        nof_c = per_cls[c]["nof"]
        ratio_c = (nof_c / max(1, nmf_c)) if nmf_c > 0 else 0.0
        per_rows.append([c, cnt, nmf_c, nof_c, f"{ratio_c:.4f}"])
    return nmf, nof, ratio, per_rows

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
    ap.add_argument("--gamma", type=float, default=0.1,
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

    if args.head_cls is None:
        # 你前面已经有 counts_train（infer_head_class_from_train 的返回）
        _, _, counts_train = infer_head_class_from_train(cfg_base)
    else:
        # 若不方便拿训练频次，也可用验证集 counts_val 近似；更严谨还是 counts_train
        counts_train = counts_val

    tau_vec_fcc = build_tau_vector(num_classes, counts_train, gamma=args.gamma, start_from_1=True)
    tau_vec_fccr = build_tau_vector(num_classes, counts_train, gamma=args.gamma, start_from_1=True)

    # 对 FCC 模型做 mis-area 诊断
    model_fcc = Network(cfg_fcc, mode="test", num_classes=num_classes)  # 先建 Network
    model_fcc.load_model(
        os.path.join(cfg_fcc.OUTPUT_DIR, cfg_fcc.NAME, "models", cfg_fcc.TEST.MODEL_FILE),
        tau_norm=bool(getattr(cfg_fcc.TEST.TAU_NORM, "USE_TAU_NORM", False)) if hasattr(cfg_fcc.TEST,
                                                                                        "TAU_NORM") else False,
        tau=float(getattr(cfg_fcc.TEST.TAU_NORM, "TAU", 1.0)) if hasattr(cfg_fcc.TEST, "TAU_NORM") else 1.0,
    )
    model_fcc = torch.nn.DataParallel(model_fcc).to(device)  # 再包 DataParallel
    model_fcc.eval()

    nmf_fcc, nof_fcc, ratio_fcc, per_rows_fcc = eval_misclassified_area(
        model_fcc, valid_set, device, tau_vec_fcc
    )

    # --- FCCR ---
    model_fccr = Network(cfg_fccr, mode="test", num_classes=num_classes)
    model_fccr.load_model(
        os.path.join(cfg_fccr.OUTPUT_DIR, cfg_fccr.NAME, "models", cfg_fccr.TEST.MODEL_FILE),
        tau_norm=bool(getattr(cfg_fccr.TEST.TAU_NORM, "USE_TAU_NORM", False)) if hasattr(cfg_fccr.TEST,
                                                                                         "TAU_NORM") else False,
        tau=float(getattr(cfg_fccr.TEST.TAU_NORM, "TAU", 1.0)) if hasattr(cfg_fccr.TEST, "TAU_NORM") else 1.0,
    )
    model_fccr = torch.nn.DataParallel(model_fccr).to(device)
    model_fccr.eval()

    nmf_r, nof_r, ratio_r, per_rows_r = eval_misclassified_area(
        model_fccr, valid_set, device, tau_vec_fccr
    )

    emit("===== Misclassified-Area (NOF/NMF) =====")
    emit(f"FCC : NMF={nmf_fcc}, NOF={nof_fcc}, Ratio={ratio_fcc:.4f}")
    emit(f"FCCR: NMF={nmf_r},   NOF={nof_r},   Ratio={ratio_r:.4f}")
    emit(f"ΔRatio (FCCR - FCC) = {ratio_r - ratio_fcc:+.4f}  -> 越负越好")
    emit("=========================================")

    # 导出 CSV
    save_csv(os.path.join(out_dir, f"mis_area_FCC{suffix}.csv"),
             ["class_id", "count", "nmf", "nof", "ratio"], per_rows_fcc)
    save_csv(os.path.join(out_dir, f"mis_area_FCCR{suffix}.csv"),
             ["class_id", "count", "nmf", "nof", "ratio"], per_rows_r)

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