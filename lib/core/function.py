import _init_paths
from core.evaluate import accuracy, AverageMeter, FusionMatrix
import csv
import torch.nn.functional as F
import numpy as np
import torch
import torch.distributed as dist
import time
from tqdm import tqdm
from torch.distributed.distributed_c10d import _coalescing_manager
try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    pass
import os

def _infer_num_classes(cfg, model, dataloader):
    # 优先从 model 取
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "num_classes") and m.num_classes is not None:
        return int(m.num_classes)
    # 再从 dataset 取
    ds = getattr(dataloader, "dataset", None)
    if ds is not None:
        for key in ["num_classes", "n_classes", "NUM_CLASSES"]:
            if hasattr(ds, key):
                return int(getattr(ds, key))
        if hasattr(ds, "cls_num_list"):
            return int(len(ds.cls_num_list))
    # 最后从 cfg 猜
    if hasattr(cfg, "DATASET") and hasattr(cfg.DATASET, "NUM_CLASSES"):
        return int(cfg.DATASET.NUM_CLASSES)
    raise ValueError("无法推断 num_classes：请在 model/dataset/cfg 中提供 NUM_CLASSES。")


def _append_loss_csv(cfg, epoch, split, overall_loss, class_loss_avg, class_count, rank=0):
    """
    追加写入：OUTPUT_DIR/NAME/loss_curves.csv
    每个 epoch 写两行：split=train/valid
    """
    if rank != 0:
        return

    out_dir = os.path.join(cfg.OUTPUT_DIR, cfg.NAME)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "loss_curves.csv")

    write_header = not os.path.exists(csv_path)
    num_classes = len(class_loss_avg)

    header = (["epoch", "split", "overall_loss"]
              + [f"class_{c}_loss" for c in range(num_classes)]
              + [f"class_{c}_count" for c in range(num_classes)])

    row = ([int(epoch), str(split), float(overall_loss)]
           + [float(x) for x in class_loss_avg]
           + [int(x) for x in class_count])

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(header)
        w.writerow(row)
def train_model(
    trainLoader, model, epoch, epoch_number, optimizer, combiner, criterion, cfg, logger, rank=0, use_apex=True, **kwargs
):
    if cfg.EVAL_MODE:
        model.eval()
    else:
        model.train()

    trainLoader.dataset.update(epoch)
    combiner.update(epoch)
    criterion.update(epoch)

    start_time = time.time()
    number_batch = len(trainLoader)

    all_loss = AverageMeter()
    acc = AverageMeter()

    # ===== 新增：每类 loss 统计容器（按“样本级 CE”统计，不影响反传）=====
    device = next(model.parameters()).device
    num_classes = _infer_num_classes(cfg, model, trainLoader)
    cls_loss_sum = torch.zeros(num_classes, device=device, dtype=torch.float64)
    cls_cnt = torch.zeros(num_classes, device=device, dtype=torch.float64)

    for i, (image, label, meta) in enumerate(trainLoader):
        ret = combiner.forward(model, criterion, image, label, meta, return_logits=True)

        if isinstance(ret, (list, tuple)) and len(ret) == 4:
            loss, now_acc, logits_detached, labels_detached = ret

            # ✅ 用“真实参与计算的样本数”计权
            cnt = int(labels_detached.shape[0])

            loss_vec = F.cross_entropy(logits_detached, labels_detached.long(), reduction="none").double()
            cls_loss_sum.scatter_add_(0, labels_detached.long(), loss_vec)
            cls_cnt.scatter_add_(0, labels_detached.long(), torch.ones_like(loss_vec, dtype=torch.float64))
        else:
            loss, now_acc = ret
            cnt = int(label.shape[0])  # 兜底（理论上你只跑 default/fcc/fccr 就不会进来）

        optimizer.zero_grad()
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        all_loss.update(loss.data.item(), cnt)
        acc.update(now_acc, cnt)

        if i % cfg.SHOW_STEP == 0 and rank == 0:
            pbar_str = "Epoch:{:>3d}  Batch:{:>3d}/{}  Batch_Loss:{:>5.3f}  Batch_Accuracy:{:>5.2f}%     ".format(
                epoch, i, number_batch, all_loss.val, acc.val * 100
            )
            logger.info(pbar_str)

    # ===== 分布式：把 cls_loss_sum / cls_cnt 做 all_reduce（可选）=====
    if getattr(cfg.TRAIN, "DISTRIBUTED", False) and dist.is_available() and dist.is_initialized():
        dist.all_reduce(cls_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(cls_cnt, op=dist.ReduceOp.SUM)

    # ===== 写 CSV（每类 loss = sum/count）=====
    cls_cnt_safe = torch.clamp(cls_cnt, min=1.0)
    cls_loss_avg = (cls_loss_sum / cls_cnt_safe).detach().cpu().numpy().tolist()
    cls_cnt_out = cls_cnt.detach().cpu().numpy().astype(int).tolist()
    _append_loss_csv(cfg, epoch, "train", all_loss.avg, cls_loss_avg, cls_cnt_out, rank=rank)

    end_time = time.time()
    pbar_str = "---Epoch:{:>3d}/{}   Avg_Loss:{:>5.3f}   Epoch_Accuracy:{:>5.2f}%   Epoch_Time:{:>5.2f}min---".format(
        epoch, epoch_number, all_loss.avg, acc.avg * 100, (end_time - start_time) / 60
    )
    if rank == 0:
        logger.info(pbar_str)

    return acc.avg, all_loss.avg

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= world_size
    return rt

def valid_model(
    dataLoader, epoch_number, model, cfg, criterion, logger, device, rank, distributed, **kwargs
):
    model.eval()

    if cfg.LOSS.LOSS_TYPE=="DiVEKLD":
        criterion = criterion.base_loss
    with torch.no_grad():
        all_loss = AverageMeter()
        acc_avg = AverageMeter()
        num_classes = _infer_num_classes(cfg, model, dataLoader)
        cls_loss_sum = torch.zeros(num_classes, device=device, dtype=torch.float64)
        cls_cnt = torch.zeros(num_classes, device=device, dtype=torch.float64)

        func = torch.nn.Sigmoid() \
            if cfg.LOSS.LOSS_TYPE in ['FocalLoss', 'ClassBalanceFocal'] else \
            torch.nn.Softmax(dim=1)

        for i, (image, label, meta) in enumerate(dataLoader):
            image, label = image.to(device), label.to(device)

            feature = model(image, feature_flag=True)

            output = model(feature, classifier_flag=True, label=label)
            loss = criterion(output, label, feature=feature)
            score_result = func(output)

            now_result = torch.argmax(score_result, 1)
            acc, cnt = accuracy(now_result.cpu().numpy(), label.cpu().numpy())

            if distributed:
                world_size = float(os.environ.get("WORLD_SIZE", 1))
                reduced_loss = reduce_tensor(loss.data, world_size)
                reduced_acc = reduce_tensor(torch.from_numpy(np.array([acc])).cuda(), world_size)
                loss = reduced_loss.cpu().data
                acc = reduced_acc.cpu().data
            loss_vec = F.cross_entropy(output, label.long(), reduction="none").double()
            cls_loss_sum.scatter_add_(0, label.long(), loss_vec)
            cls_cnt.scatter_add_(0, label.long(), torch.ones_like(loss_vec, dtype=torch.float64))
            all_loss.update(loss.data.item(), label.shape[0])
            if distributed:
                acc_avg.update(acc.data.item(), cnt*world_size)
            else:
                acc_avg.update(acc, cnt)

        pbar_str = "------- Valid: Epoch:{:>3d}  Valid_Loss:{:>5.3f}   Valid_Acc:{:>5.2f}%-------".format(
            epoch_number, all_loss.avg, acc_avg.avg * 100
        )
        if distributed and dist.is_available() and dist.is_initialized():
            dist.all_reduce(cls_loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(cls_cnt, op=dist.ReduceOp.SUM)

        cls_cnt_safe = torch.clamp(cls_cnt, min=1.0)
        cls_loss_avg = (cls_loss_sum / cls_cnt_safe).detach().cpu().numpy().tolist()
        cls_cnt_out = cls_cnt.detach().cpu().numpy().astype(int).tolist()
        _append_loss_csv(cfg, epoch_number, "valid", all_loss.avg, cls_loss_avg, cls_cnt_out, rank=rank)
        if rank == 0:
            logger.info(pbar_str)
        # === 在 Valid 日志下一行打印 Auto-F3C 的 gamma_shift（每类一个） ===

        if rank == 0 and getattr(cfg.TRAIN.COMBINER, "TYPE", "").lower() == "auto_f3c":
            # 兼容 DataParallel / 非 DP 取到 aligner
            aligner = getattr(model.module, "auto_f3c_aligner", None) if hasattr(model, "module") else getattr(model,"auto_f3c_aligner",None)
            if aligner is not None:
                with torch.no_grad():
                    # 调用 forward 以触发 clamp，拿到当前（已裁剪）的值
                    gamma_vals = aligner().detach().cpu().tolist()
          # 只打印前三类（你是 3 类），若将来类别数变化也不报错
                k = min(3, len(gamma_vals))
                msg = "GammaShift: " + " | ".join([f"class {i}: {gamma_vals[i]:.3f}" for i in range(k)])
                logger.info(msg)
    return acc_avg.avg, all_loss.avg
