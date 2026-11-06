# eval_head_tail.py
import os, argparse, numpy as np, torch
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

    # ======= Metrics helpers =======

    def confusion_matrix_np(y_true, y_pred, C: int):
        cm = np.zeros((C, C), dtype=np.int64)
        # 速度更快的向量化聚合
        np.add.at(cm, (y_true, y_pred), 1)
        return cm

    def macro_avg_acc_and_f1(y_true, y_pred, C: int):
        """
        Macro-Avg Accuracy (Mean Class Acc) = 对每类“召回”取平均
        Macro-F1 = 对每类 F1 取平均（仅统计有样本的类别）
        """
        cm = confusion_matrix_np(y_true, y_pred, C)
        tp = np.diag(cm).astype(float)
        support = cm.sum(axis=1).astype(float)            # 每类真样本数
        pred_pos = cm.sum(axis=0).astype(float)           # 每类预测数

        # 按类 Precision / Recall
        precision_c = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=(pred_pos > 0))
        recall_c    = np.divide(tp, support, out=np.zeros_like(tp),  where=(support  > 0))

        # F1
        denom = precision_c + recall_c
        f1_c = np.divide(2 * precision_c * recall_c, denom, out=np.zeros_like(tp), where=(denom > 0))

        valid = (support > 0)  # 只对有样本的类别做平均
        macro_acc = float(np.mean(recall_c[valid]))  # mean class recall
        macro_f1  = float(np.mean(f1_c[valid]))
        return macro_acc, macro_f1

    def tail_recall_micro(y_true, y_pred, tail_set):
        """Tail Recall（micro）：真标签∈尾类的样本整体召回"""
        tail = set(tail_set)
        idx = np.isin(y_true, list(tail))
        if not idx.any():
            return float("nan")
        return float((y_true[idx] == y_pred[idx]).mean())

    # ======= Compute =======
    macro_acc, macro_f1 = macro_avg_acc_and_f1(y_true, y_pred, num_classes)
    head, mid, tail, _ = tertile_buckets_from_train(cfg_node)
    tail_rec = tail_recall_micro(y_true, y_pred, tail)
    # 若还想要 Tail Recall 的宏平均：取消下一行注释
    tail_rec_macro = float(np.mean([(y_pred[y_true==c]==c).mean() for c in tail if np.any(y_true==c)]))

    # ======= Log / Print =======
    logger.info(f"Macro-Avg Accuracy (Mean Class Acc): {macro_acc*100:.2f}%")
    logger.info(f"Macro-F1: {macro_f1*100:.2f}%")
    logger.info(f"Tail Recall (micro): {tail_rec*100:.2f}%")
    logger.info(f"Tail Recall (macro): {tail_rec_macro * 100:.2f}%")
    print(f"Macro-Avg Accuracy (Mean Class Acc): {macro_acc*100:.2f}%")
    print(f"Macro-F1: {macro_f1*100:.2f}%")
    print(f"Tail Recall: {tail_rec*100:.2f}%")
    print(f"Tail Recall(macro): {tail_rec_macro*100:.2f}%")

    return {
        "macro_avg_acc": macro_acc,
        "macro_f1": macro_f1,
        "tail_recall": tail_rec,          # micro
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
        default=r"I:\2024223020007\FCC\configs\LAM\alexnet\LAM_fccr_1_start1_alexnet.yaml",
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