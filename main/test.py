import _init_paths
from dataset.lam_dataset import LAMDataset
from config import cfg, update_config
import argparse
import ast
def parse_args():
    parser = argparse.ArgumentParser(description="codes for BagofTricks-LT")

    parser.add_argument(
        "--cfg",
        help="decide which cfg to use",
        required=False,
        default=r"I:\2024223020007\FCC\configs\LAM\Densenet\LAM_baseline_densenet.yaml",
        type=str,
    )
    parser.add_argument(
        "--ar",
        help="decide whether to use auto resume",
        type= ast.literal_eval,
        dest = 'auto_resume',
        required=False,
        default= False,
    )

    parser.add_argument(
        "--local_rank",
        help='local_rank for distributed training',
        type=int,
        default=0,
    )

    parser.add_argument(
        "opts",
        help="modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args

args = parse_args()
update_config(cfg, args)
cfg.defrost()
cfg.DATASET.DATASET = 'LAMDataset'
cfg.freeze()
dataset_instance = LAMDataset("train", cfg)
dataset_instance_2 = LAMDataset("valid", cfg)
print(dataset_instance)
print(dataset_instance_2)

dataset_instance_eval = eval(cfg.DATASET.DATASET)("train", cfg)
dataset_instance_valid = eval(cfg.DATASET.DATASET)("valid", cfg)
print(dataset_instance_valid)
print(dataset_instance_eval)