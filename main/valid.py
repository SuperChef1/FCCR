import random

import _init_paths
from net import build_network
from config import cfg, update_config
from dataset.lam_dataset import LAMDataset
from dataset.lam_dataset_bbn import LAMDatasetBBN
from dataset.lam_dataset_balance import LAMDataset_balanced
from dataset.lam_dataset_LT import LAMDataset_LT
import numpy as np
import torch
torch.use_deterministic_algorithms(True, warn_only=True)
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # or ":16:8"
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from core.evaluate import FusionMatrix

import matplotlib.pyplot as plt
from utils.utils import valid_logger

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import sklearn.metrics as sm
from utils.repro import setup_repro, make_seeded_loader
SEED = 111
setup_repro(SEED)

_to_allow = []
for attr in ("_core", "core"):  # 兼容不同 NumPy 版本路径
    try:
        _to_allow.append(getattr(np, attr).multiarray.scalar)
    except Exception:
        pass
if _to_allow:
    torch.serialization.add_safe_globals(_to_allow)

def eval_matrix(labels, preds):
    precision, recall, f_score, _ = precision_recall_fscore_support(labels, preds, average='macro')

    cp = sm.classification_report(labels, preds)

    return precision, recall, f_score, cp

def parse_args():
    parser = argparse.ArgumentParser(description="tricks evaluation")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\resnet18\LAM_fccr_0.1_start70_resnet18.yaml",
        type=str,
    )
    parser.add_argument(
        "--gpus",
        help="decide which gpus to use",
        required=False,
        default='0',
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

def valid_model(dataLoader, model, cfg, device, num_classes):

    logger, log_file = valid_logger(cfg)

    result_list = []
    pbar = tqdm(total=len(dataLoader))
    model.eval()
    top1_count, top2_count, top3_count, index, fusion_matrix = (
        [],
        [],
        [],
        0,
        FusionMatrix(num_classes),
    )

    func = torch.nn.Sigmoid() \
        if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
        torch.nn.Softmax(dim=1)

    with torch.no_grad():
        preds = []
        out_labels = None
        for i, (image, image_labels, meta) in enumerate(dataLoader):
            image = image.to(device)
            output = model(image)
            result = func(output)
            _, top_k = result.topk(3, 1, True, True)
            score_result = result.cpu().numpy()

            fusion_matrix.update(score_result.argmax(axis=1), image_labels.numpy())
            topk_result = top_k.cpu().tolist()
            if not "image_id" in meta:
                meta["image_id"] = [0] * image.shape[0]
            image_ids = meta["image_id"]
            for i, image_id in enumerate(image_ids):
                result_list.append(
                    {
                        "image_id": image_id,
                        "image_label": int(image_labels[i]),
                        "top_3": topk_result[i],
                    }
                )
                top1_count += [topk_result[i][0] == image_labels[i]]
                top2_count += [image_labels[i] in topk_result[i][0:2]]
                top3_count += [image_labels[i] in topk_result[i][0:3]]
                index += 1
            now_acc = np.sum(top1_count) / index
            pbar.set_description("Now Top1:{:>5.2f}%".format(now_acc * 100))
            pbar.update(1)

            # cal eval matrix
            if len(preds) == 0:
                preds.append(output.detach().cpu().numpy())
                out_labels = image_labels.detach().cpu().numpy()

            else:
                preds[0] = np.append(preds[0], output.detach().cpu().numpy(), axis=0)
                out_labels = np.append(out_labels, image_labels.detach().cpu().numpy(), axis=0)

    preds = preds[0]
    preds = np.argmax(preds, axis=1)

    #print(out_labels)
    #print(preds)

    precision, recall, f1, cp = eval_matrix(out_labels, preds)
    logger.info("precision: {:.2f}%, recall: {:.2f}%, f1-score: {:.2f}%".format(precision*100, recall*100, f1*100))

    top1_acc = float(np.sum(top1_count) / len(top1_count))
    top2_acc = float(np.sum(top2_count) / len(top1_count))
    top3_acc = float(np.sum(top3_count) / len(top1_count))
    print(
        "Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )

    
    logger.info("Top1:{:>5.2f}%  Top2:{:>5.2f}%  Top3:{:>5.2f}%\n\n\n".format(
            top1_acc * 100, top2_acc * 100, top3_acc * 100
        )
    )

    logger.info("class report is \n{}".format(cp))
    

    pbar.close()
    fig = fusion_matrix.plot_confusion_matrix()
    
    save_path = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, "valid", "confusion_matrix")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'), dpi=1000)

    logger.info('Save confusion matrix to {}'.format(os.path.join(save_path, 'confusion_matrix.png')))
    
    #plt.savefig('confusion_matrix.png')


if __name__ == "__main__":
    args = parse_args()
    update_config(cfg, args)
    # SEED = 111
    # random.seed(SEED)
    # np.random.seed(SEED)
    # torch.manual_seed(SEED)
    # torch.cuda.manual_seed_all(SEED)
    #
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # # 关掉 TF32，减少数值路径差异
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    test_set = eval(cfg.DATASET.DATASET)("valid", cfg)
    num_classes = test_set.get_num_classes()
    device = torch.device("cuda")
    model = build_network(cfg, mode="test", num_classes=num_classes)

    model_file = os.path.join(cfg.OUTPUT_DIR, cfg.NAME, 'models', cfg.TEST.MODEL_FILE)
    model.load_model(model_file, tau_norm=False, tau=1.0)

    model = torch.nn.DataParallel(model).cuda()

    testLoader = make_seeded_loader(
        test_set,
        batch_size=cfg.TEST.BATCH_SIZE,
        num_workers=1,
        shuffle=False,
        pin_memory=getattr(cfg, "PIN_MEMORY", False),
        drop_last=False,
        seed=SEED
    )
    valid_model(testLoader, model, cfg, device, num_classes)
