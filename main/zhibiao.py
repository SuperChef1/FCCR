import os, argparse, numpy as np, torch



torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import  pandas
import openpyxl
import _init_paths
from config import cfg, update_config
from net import Network
from utils.repro import setup_repro, make_seeded_loader
from utils.utils import valid_logger, get_category_list

from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_balance import LAMDataset_balanced
from dataset.lam_dataset_LT import LAMDataset_LT

SEED = 111
setup_repro(SEED)

_to_allow = []
for attr in ("_core", "core"):
    try:
        _to_allow.append(getattr(np, attr).multiarray.scalar)
    except Exception:
        pass
if _to_allow:
    torch.serialization.add_safe_globals(_to_allow)

def build_dataset(split: str, cfg_node):
    return eval(cfg_node.DATASET.DATASET)(split, cfg_node)

def tertile_buckets_from_train(cfg_node):
    train_set = build_dataset("train", cfg_node)
    num_classes = train_set.get_num_classes()
    annotations = train_set.get_annotations()
    freq, _ = get_category_list(annotations, num_classes, cfg_node)
    freq = np.asarray(freq, dtype=int)
    order = np.argsort(freq)
    C = len(order)
    k = max(1, C // 3)
    tail = set(order[:k].tolist())
    head = set(order[-k:].tolist())
    mid  = set(order[k:-k].tolist()) if C - 2*k > 0 else set()
    return head, mid, tail, freq

# ---------- 核心度量工具 ----------

def _confusion_matrix_np(y_true, y_pred, C: int):
    cm = np.zeros((C, C), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm

def macro_avg_acc_and_f1(y_true, y_pred, C: int):
    """
    Macro-Avg Accuracy (Mean Class Acc) = mean per-class recall
    Macro-F1 = mean per-class F1 （只对有样本的类）
    """
    cm = _confusion_matrix_np(y_true, y_pred, C)
    tp       = np.diag(cm).astype(float)
    support  = cm.sum(axis=1).astype(float)  # 每类真样本数
    pred_pos = cm.sum(axis=0).astype(float)  # 每类预测数

    precision_c = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=(pred_pos > 0))
    recall_c    = np.divide(tp, support,  out=np.zeros_like(tp), where=(support  > 0))

    denom = precision_c + recall_c
    f1_c  = np.divide(2 * precision_c * recall_c, denom, out=np.zeros_like(tp), where=(denom > 0))

    valid = (support > 0)
    macro_acc = float(np.mean(recall_c[valid]))
    macro_f1  = float(np.mean(f1_c[valid]))
    return macro_acc, macro_f1

def tail_vs_rest_metrics(y_true, y_pred, tail_ids):
    """
    以 tail 集合为“正类”做二分类度量：Precision/Recall/F1/TNR。
    """
    tail = set(tail_ids)
    t_true = np.isin(y_true, list(tail))
    t_pred = np.isin(y_pred, list(tail))

    TP = np.logical_and(t_true,  t_pred).sum()
    TN = np.logical_and(~t_true, ~t_pred).sum()
    FP = np.logical_and(~t_true, t_pred).sum()
    FN = np.logical_and(t_true,  ~t_pred).sum()

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1        = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0.0
    tnr       = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return precision, recall, f1, tnr

# 若希望把 Tail Recall 换成“尾类细分类严格 Top-1 正确率”（而非 tail-vs-rest），解开此函数并在 infer 中替换：
def tail_strict_top1_recall(y_true, y_pred, tail_ids):
    idx = np.isin(y_true, list(tail_ids))
    if not np.any(idx): return float("nan")
    return float((y_true[idx] == y_pred[idx]).mean())

@torch.no_grad()
def infer_valid(cfg_node, cfg_path=None):
    logger, _ = valid_logger(cfg_node)

    # ===== Model =====
    device = torch.device("cuda")
    valid_set = build_dataset("valid", cfg_node)
    num_classes = valid_set.get_num_classes()

    model = Network(cfg_node, mode="test", num_classes=num_classes)
    model_file = os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=False, tau=1.0)
    model = torch.nn.DataParallel(model).to(device).eval()

    # 标识信息
    logger.info(f"[EXP] NAME={cfg_node.NAME}")
    logger.info(f"[CFG] {cfg_path if cfg_path else '(unknown)'}")
    logger.info(f"[WEIGHTS] {model_file}")
    print(f"\n[EXP] {cfg_node.NAME}\n[CFG] {cfg_path if cfg_path else '(unknown)'}\n[WEIGHTS] {os.path.basename(model_file)}\n")

    # ===== DataLoader =====
    loader = make_seeded_loader(
        valid_set,
        batch_size=cfg_node.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=getattr(cfg_node, "PIN_MEMORY", False),
        drop_last=False,
        seed=SEED,
    )

    # ===== Inference =====
    preds, labels = [], []
    for images, gt, _ in loader:
        images = images.to(device)
        logits = model(images)
        pred = logits.argmax(dim=1).cpu().numpy().astype(int)
        preds.append(pred)
        labels.append(gt.numpy().astype(int))
    y_pred = np.concatenate(preds, 0)
    y_true = np.concatenate(labels, 0)

    # ===== Metrics =====
    macro_acc, macro_f1 = macro_avg_acc_and_f1(y_true, y_pred, num_classes)
    head, mid, tail, _  = tertile_buckets_from_train(cfg_node)

    # tail-vs-rest 版本（Precision/Recall/F1/TNR）
    tail_prec, tail_rec, tail_f1, tail_tnr = tail_vs_rest_metrics(y_true, y_pred, tail)

    # 如需“严格 Top-1”版本的尾类召回，替换上一行 tail_rec：
    # tail_rec = tail_strict_top1_recall(y_true, y_pred, tail)

    # ===== 打印 =====
    print(f"Macro-Avg Accuracy (Mean Class Acc): {macro_acc*100:.2f}%")
    print(f"Macro-F1: {macro_f1*100:.2f}%")
    print(f"Tail class Precision: {tail_prec*100:.2f}%")
    print(f"Tail class Recall:    {tail_rec*100:.2f}%")
    print(f"Tail class F1:        {tail_f1*100:.2f}%")
    print(f"High-risk Specificity:       {tail_tnr*100:.2f}%")

    logger.info(f"Macro-Avg Accuracy (Mean Class Acc): {macro_acc*100:.2f}%")
    logger.info(f"Macro-F1: {macro_f1*100:.2f}%")
    logger.info(f"Tail class Precision: {tail_prec*100:.2f}%")
    logger.info(f"Tail class Recall:    {tail_rec*100:.2f}%")
    logger.info(f"Tail class F1:        {tail_f1*100:.2f}%")
    logger.info(f"High-risk Specificity:       {tail_tnr*100:.2f}%")

    # ===== 落盘到 Excel（若无 pandas → 回退 CSV）=====
    MODEL_NAME = "Alexnet_epoch"
    XLSX_EXT = ".xlsx"
    CSV_EXT = ".csv"

    XLSX_NAME = f"{MODEL_NAME}{XLSX_EXT}"
    CSV_NAME = f"{MODEL_NAME}{CSV_EXT}"
    out_dir = r"I:\2024223020007\FCC\eval_results"
    out_dir = os.path.join(out_dir,MODEL_NAME)
    os.makedirs(out_dir, exist_ok=True)
    xlsx_path = os.path.join(out_dir, XLSX_NAME)
    csv_path  = os.path.join(out_dir, CSV_NAME)

    row = {
        "exp_name": cfg_node.NAME,
        "cfg": cfg_path if cfg_path else "",
        "weights": os.path.basename(model_file),
        "Macro-Avg Accuracy": macro_acc,
        "Macro-F1": macro_f1,
        "Tail Precision": tail_prec,
        "Tail Recall": tail_rec,
        "Tail F1": tail_f1,
        "High-risk Specificity": tail_tnr,
        "tail_ids": ",".join(map(str, sorted(list(tail)))),
    }

    try:
        import pandas as pd
        import datetime as _dt
        ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row_with_time = {"timestamp": ts, **row}
        df = pd.DataFrame([row_with_time])
        if os.path.isfile(xlsx_path):
            # 追加到已有 xlsx（按 sheet=Sheet1 追加）
            with pd.ExcelWriter(xlsx_path, engine="openpyxl", mode="a", if_sheet_exists="overlay") as w:
                # 若需要竖向追加，可先读旧 df 再 concat；这里简单起见直接追加到下方：
                try:
                    old = pd.read_excel(xlsx_path)
                    out = pd.concat([old, df], ignore_index=True)
                except Exception:
                    out = df
                out.to_excel(w, index=False)
        else:
            df.to_excel(xlsx_path, index=False)
        logger.info(f"[SAVE] Excel -> {xlsx_path}")
    except Exception as e:
        # 回退 CSV
        import csv
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                w.writeheader()
            w.writerow(row)
        logger.info(f"[SAVE] CSV -> {csv_path} (pandas/openpyxl 不可用: {e})")

    return {
        "macro_avg_acc": macro_acc,
        "macro_f1": macro_f1,
        "tail_precision": tail_prec,
        "tail_recall": tail_rec,
        "tail_f1": tail_f1,
        "tail_tnr": tail_tnr,
        "tail_ids": sorted(list(tail)),
        "model_file": model_file,
        "cfg_path": cfg_path,
        "exp_name": cfg_node.NAME,
    }

def parse_args():
    ap = argparse.ArgumentParser("Eval overall/head/tail accuracy")
    ap.add_argument("--cfg",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\alexnet\LAM_fccr_1_start70_alexnet.yaml",
        type=str,)
    ap.add_argument("--gpus", default="0", type=str)


    # 重要：opts 给默认空列表，避免 merge_from_list(None)
    ap.add_argument("opts", nargs=argparse.REMAINDER, default=[])
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    update_config(cfg, args)
    out = infer_valid(cfg, cfg_path=args.cfg)