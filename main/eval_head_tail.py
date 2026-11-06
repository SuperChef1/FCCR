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


def top1_overall(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((y_true == y_pred).mean())


def bucket_acc(y_true, y_pred, bucket):
    bucket = set(bucket)
    mask = np.isin(y_true, list(bucket))
    micro = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else float("nan")
    per_cls = []
    for c in bucket:
        m = (y_true == c)
        if m.any():
            per_cls.append(float((y_true[m] == y_pred[m]).mean()))
    macro = float(np.mean(per_cls)) if per_cls else float("nan")
    return micro, macro


@torch.no_grad()
def infer_valid(cfg_node, cfg_path=None):   # <<-- 新增 cfg_path 便于打印
    logger, _ = valid_logger(cfg_node)

    # ===== Model =====
    device = torch.device("cuda")
    num_classes = build_dataset("valid", cfg_node).get_num_classes()
    model = Network(cfg_node, mode="test", num_classes=num_classes)
    model_file = os.path.join(cfg_node.OUTPUT_DIR, cfg_node.NAME, "models", cfg_node.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=False, tau=1.0)
    model = torch.nn.DataParallel(model).to(device).eval()

    # 打印标识
    logger.info(f"[EXP] NAME={cfg_node.NAME}")
    logger.info(f"[CFG] {cfg_path if cfg_path else '(unknown)'}")
    logger.info(f"[WEIGHTS] {model_file}")
    print(f"\n[EXP] {cfg_node.NAME}\n[CFG] {cfg_path if cfg_path else '(unknown)'}\n[WEIGHTS] {os.path.basename(model_file)}\n")

    # ===== DataLoader =====
    valid_set = build_dataset("valid", cfg_node)
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
    overall = top1_overall(y_true, y_pred)
    head, mid, tail, _ = tertile_buckets_from_train(cfg_node)
    head_micro, head_macro = bucket_acc(y_true, y_pred, head)
    tail_micro, tail_macro = bucket_acc(y_true, y_pred, tail)

    logger.info(f"[VALID N={len(y_true)}] Top-1 overall = {overall*100:.2f}%")
    logger.info(f"Head classes: {sorted(list(head))} | Tail classes: {sorted(list(tail))}")
    logger.info(f"Head  acc - micro={head_micro*100:.2f}% | macro={head_macro*100:.2f}%")
    logger.info(f"Tail  acc - micro={tail_micro*100:.2f}% | macro={tail_macro*100:.2f}%")

    print(f"Overall Top-1 : {overall*100:.2f}%")
    print(f"Head   Top-1  : micro {head_micro*100:.2f}% | macro {head_macro*100:.2f}%")
    print(f"Tail   Top-1  : micro {tail_micro*100:.2f}% | macro {tail_macro*100:.2f}%")

    return {
        "overall": overall,
        "head_micro": head_micro,
        "tail_micro": tail_micro,
        "head_macro": head_macro,
        "tail_macro": tail_macro,
        "head_ids": sorted(list(head)),
        "tail_ids": sorted(list(tail)),
        "model_file": model_file,          # <<-- 返回权重路径
        "cfg_path": cfg_path,              # <<-- 返回 cfg 路径
        "exp_name": cfg_node.NAME,         # <<-- 返回实验名
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