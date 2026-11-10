"""R1-5 quantitative feature-geometry analysis for Raw, FCC, and FCCR.

The experiment measures, in the original classifier-input feature space:

1. per-class intra-class scatter; and
2. the train-to-inference class-center gap before and after the configured
   training-time feature transform.

The raw train-to-inference gap is reported separately from the transformed
gap.  Their difference prevents the natural train/validation split shift from
being incorrectly attributed to FCC or FCCR.
"""

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np


EPS = 1e-12
SEED = 111


def _as_feature_matrix(name, values):
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] matrix")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinity")
    return array


def _as_labels(name, values, expected_size):
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1 or array.size != expected_size:
        raise ValueError(f"{name} must be a length-{expected_size} vector")
    if array.size == 0 or int(array.min()) < 0:
        raise ValueError(f"{name} must contain non-negative class IDs")
    return array


def _class_ids(train_labels, inference_labels):
    train_classes = np.unique(train_labels)
    inference_classes = np.unique(inference_labels)
    if not np.array_equal(train_classes, inference_classes):
        raise ValueError(
            "Train and inference splits must contain the same class IDs: "
            f"train={train_classes.tolist()}, inference={inference_classes.tolist()}"
        )
    return train_classes.astype(np.int64)


def _normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    zero_count = int((norms[:, 0] <= EPS).sum())
    normalized = features / np.maximum(norms, EPS)
    return normalized, zero_count


def _centers(features, labels, class_ids):
    centers = []
    counts = []
    for class_id in class_ids:
        class_features = features[labels == class_id]
        if class_features.shape[0] == 0:
            raise ValueError(f"Class {int(class_id)} has no samples")
        centers.append(class_features.mean(axis=0))
        counts.append(class_features.shape[0])
    return np.stack(centers, axis=0), np.asarray(counts, dtype=np.int64)


def _cosine_distance(first, second):
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= EPS:
        return float("nan")
    cosine = float(np.dot(first, second) / denominator)
    return 1.0 - float(np.clip(cosine, -1.0, 1.0))


def _mean_pairwise_center_distance(centers):
    distances = []
    for first in range(centers.shape[0]):
        for second in range(first + 1, centers.shape[0]):
            distances.append(float(np.linalg.norm(centers[first] - centers[second])))
    if not distances:
        raise ValueError("At least two classes are required for normalized center gaps")
    value = float(np.mean(distances))
    if value <= EPS:
        raise ValueError("Inference class centers have zero mean pairwise separation")
    return value


def compute_scatter_rows(features, labels, class_ids, class_names, representation):
    """Compute scatter after row-wise L2 normalization in the original D-space."""
    features = _as_feature_matrix("features", features)
    labels = _as_labels("labels", labels, features.shape[0])
    normalized, zero_count = _normalize_rows(features)
    rows = []
    for class_id in class_ids:
        class_mask = labels == class_id
        class_features = normalized[class_mask]
        center = class_features.mean(axis=0)
        squared_radii = np.sum((class_features - center) ** 2, axis=1)
        scatter = float(squared_radii.mean())
        rows.append(
            {
                "representation": representation,
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "sample_count": int(class_features.shape[0]),
                "feature_dim": int(features.shape[1]),
                "zero_norm_feature_count_total": zero_count,
                "normalized_intra_class_scatter": scatter,
                "normalized_rms_radius": math.sqrt(max(scatter, 0.0)),
                "normalized_center_norm": float(np.linalg.norm(center)),
            }
        )
    return rows


def compute_center_shift_rows(
    train_raw,
    train_transformed,
    train_labels,
    inference_raw,
    inference_labels,
    class_ids,
    class_names,
):
    """Decompose the center gap into split shift and transform-induced shift.

    Center distances use unnormalized classifier-input features so the actual
    FCC/FCCR operation remains visible.  Dimensionless gaps divide by the mean
    pairwise distance between inference class centers.
    """
    train_raw = _as_feature_matrix("train_raw", train_raw)
    train_transformed = _as_feature_matrix("train_transformed", train_transformed)
    inference_raw = _as_feature_matrix("inference_raw", inference_raw)
    if train_raw.shape != train_transformed.shape:
        raise ValueError("train_raw and train_transformed must have the same shape")
    if train_raw.shape[1] != inference_raw.shape[1]:
        raise ValueError("Train and inference features must have the same dimension")
    train_labels = _as_labels("train_labels", train_labels, train_raw.shape[0])
    inference_labels = _as_labels(
        "inference_labels", inference_labels, inference_raw.shape[0]
    )

    raw_centers, train_counts = _centers(train_raw, train_labels, class_ids)
    transformed_centers, _ = _centers(
        train_transformed, train_labels, class_ids
    )
    inference_centers, inference_counts = _centers(
        inference_raw, inference_labels, class_ids
    )
    reference_distance = _mean_pairwise_center_distance(inference_centers)

    rows = []
    for position, class_id in enumerate(class_ids):
        raw_center = raw_centers[position]
        transformed_center = transformed_centers[position]
        inference_center = inference_centers[position]
        baseline_l2 = float(np.linalg.norm(raw_center - inference_center))
        transformed_l2 = float(
            np.linalg.norm(transformed_center - inference_center)
        )
        alignment_l2 = float(np.linalg.norm(transformed_center - raw_center))
        rows.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "train_sample_count": int(train_counts[position]),
                "inference_sample_count": int(inference_counts[position]),
                "feature_dim": int(train_raw.shape[1]),
                "inference_mean_interclass_center_distance": reference_distance,
                "raw_train_to_inference_l2": baseline_l2,
                "transformed_train_to_inference_l2": transformed_l2,
                "transform_center_displacement_l2": alignment_l2,
                "raw_train_to_inference_normalized": baseline_l2
                / reference_distance,
                "transformed_train_to_inference_normalized": transformed_l2
                / reference_distance,
                "transform_center_displacement_normalized": alignment_l2
                / reference_distance,
                "transform_excess_gap_normalized": (
                    transformed_l2 - baseline_l2
                )
                / reference_distance,
                "raw_train_to_inference_cosine_distance": _cosine_distance(
                    raw_center, inference_center
                ),
                "transformed_train_to_inference_cosine_distance": _cosine_distance(
                    transformed_center, inference_center
                ),
            }
        )
    return rows


def analyze_feature_geometry(
    train_raw,
    train_transformed,
    train_labels,
    inference_raw,
    inference_labels,
    class_names,
):
    """Pure NumPy entry point used by the command-line path and unit smoke tests."""
    train_raw = _as_feature_matrix("train_raw", train_raw)
    train_transformed = _as_feature_matrix("train_transformed", train_transformed)
    inference_raw = _as_feature_matrix("inference_raw", inference_raw)
    train_labels = _as_labels("train_labels", train_labels, train_raw.shape[0])
    inference_labels = _as_labels(
        "inference_labels", inference_labels, inference_raw.shape[0]
    )
    classes = _class_ids(train_labels, inference_labels)
    expected_names = int(classes.max()) + 1
    if len(class_names) != expected_names:
        raise ValueError(
            f"Expected {expected_names} class names, got {len(class_names)}"
        )

    scatter_rows = []
    scatter_rows.extend(
        compute_scatter_rows(
            train_raw, train_labels, classes, class_names, "train_raw"
        )
    )
    scatter_rows.extend(
        compute_scatter_rows(
            train_transformed,
            train_labels,
            classes,
            class_names,
            "train_transformed",
        )
    )
    scatter_rows.extend(
        compute_scatter_rows(
            inference_raw,
            inference_labels,
            classes,
            class_names,
            "inference_raw",
        )
    )
    center_rows = compute_center_shift_rows(
        train_raw,
        train_transformed,
        train_labels,
        inference_raw,
        inference_labels,
        classes,
        class_names,
    )
    return {
        "class_ids": classes.tolist(),
        "scatter_rows": scatter_rows,
        "center_shift_rows": center_rows,
    }


def _load_feature_model(cfg_node, num_classes, device):
    import torch

    from net import build_network

    model = build_network(cfg_node, mode="test", num_classes=num_classes)
    checkpoint = Path(cfg_node.OUTPUT_DIR) / cfg_node.NAME / "models" / cfg_node.TEST.MODEL_FILE
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint.resolve()}")
    use_tau = bool(getattr(cfg_node.TEST.TAU_NORM, "USE_TAU_NORM", False))
    tau = float(getattr(cfg_node.TEST.TAU_NORM, "TAU", 1.0))
    model.load_model(str(checkpoint), tau_norm=use_tau, tau=tau)
    model = model.to(device)
    model.eval()
    return model, checkpoint.resolve()


def _extract_split_features(
    model,
    cfg_node,
    dataset,
    device,
    method,
    split,
    seed,
):
    import torch

    from utils.repro import make_seeded_loader

    if split == "train":
        batch_size = int(cfg_node.TRAIN.BATCH_SIZE)
    else:
        batch_size = int(cfg_node.TEST.BATCH_SIZE)
    loader = make_seeded_loader(
        dataset,
        batch_size=batch_size,
        num_workers=1,
        shuffle=False,
        pin_memory=bool(getattr(cfg_node, "PIN_MEMORY", False)),
        drop_last=False,
        seed=seed,
    )
    raw_parts = []
    transformed_parts = []
    label_parts = []
    now_epoch = int(cfg_node.TRAIN.MAX_EPOCH)

    model.eval()
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels_device = labels.to(device, non_blocking=True)
            raw = model(images, feature_flag=True)
            if split != "train" or method == "raw":
                transformed = raw
            elif method == "fcc":
                transformed = model(
                    images,
                    feature_fcc_flag=True,
                    label=labels_device,
                    now_epoch=now_epoch,
                )
            elif method == "fccr":
                transformed = model(
                    images,
                    feature_fccr_flag=True,
                    label=labels_device,
                    now_epoch=now_epoch,
                )
            else:
                raise ValueError(f"Unsupported method: {method}")
            raw_parts.append(raw.detach().cpu().numpy().astype(np.float32))
            transformed_parts.append(
                transformed.detach().cpu().numpy().astype(np.float32)
            )
            label_parts.append(labels.numpy().astype(np.int64))

    return (
        np.concatenate(raw_parts, axis=0),
        np.concatenate(transformed_parts, axis=0),
        np.concatenate(label_parts, axis=0),
    )


def _write_csv(path, rows):
    if not rows:
        raise ValueError(f"Refusing to write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _macro_mean(rows, key):
    values = np.asarray([float(row[key]) for row in rows], dtype=np.float64)
    return float(np.nanmean(values))


def _summary_row(method, result, large_class):
    inference_scatter = [
        row
        for row in result["scatter_rows"]
        if row["representation"] == "inference_raw"
    ]
    large_scatter = next(
        row for row in inference_scatter if row["class_id"] == large_class
    )
    center_rows = result["center_shift_rows"]
    large_center = next(row for row in center_rows if row["class_id"] == large_class)
    return {
        "method": method,
        "large_class_id": large_class,
        "large_class_name": large_scatter["class_name"],
        "large_inference_scatter": large_scatter[
            "normalized_intra_class_scatter"
        ],
        "macro_inference_scatter": _macro_mean(
            inference_scatter, "normalized_intra_class_scatter"
        ),
        "large_raw_train_to_inference_gap": large_center[
            "raw_train_to_inference_normalized"
        ],
        "macro_raw_train_to_inference_gap": _macro_mean(
            center_rows, "raw_train_to_inference_normalized"
        ),
        "large_transformed_train_to_inference_gap": large_center[
            "transformed_train_to_inference_normalized"
        ],
        "macro_transformed_train_to_inference_gap": _macro_mean(
            center_rows, "transformed_train_to_inference_normalized"
        ),
        "large_transform_excess_gap": large_center[
            "transform_excess_gap_normalized"
        ],
        "macro_transform_excess_gap": _macro_mean(
            center_rows, "transform_excess_gap_normalized"
        ),
        "large_transform_center_displacement": large_center[
            "transform_center_displacement_normalized"
        ],
        "macro_transform_center_displacement": _macro_mean(
            center_rows, "transform_center_displacement_normalized"
        ),
    }


def _load_cfg(yaml_path):
    """Clone and merge a config exactly as the existing comparison scripts do."""
    from config import cfg as default_cfg

    cfg_node = default_cfg.clone()
    cfg_node.defrost()
    cfg_node.merge_from_file(str(yaml_path))
    cfg_node.freeze()
    return cfg_node


def _build_dataset(split, cfg_node):
    """Use the same dataset set supported by the existing compare.py path."""
    from dataset.lam_dataset import LAMDataset
    from dataset.lam_dataset_LT import LAMDataset_LT
    from dataset.lam_dataset_balance import LAMDataset_balanced
    from dataset.lam_dataset_bbn import LAMDatasetBBN

    builders = {
        "LAMDataset": LAMDataset,
        "LAMDataset_LT": LAMDataset_LT,
        "LAMDataset_balanced": LAMDataset_balanced,
        "LAMDatasetBBN": LAMDatasetBBN,
    }
    dataset_name = str(cfg_node.DATASET.DATASET)
    if dataset_name not in builders:
        raise ValueError(
            f"Unsupported dataset {dataset_name!r}; supported: {sorted(builders)}"
        )
    return builders[dataset_name](split, cfg_node)


def _comparison_signature(cfg_node):
    """Settings that must match for a paired mechanism comparison."""
    return {
        "dataset": str(cfg_node.DATASET.DATASET),
        "input_size": tuple(cfg_node.INPUT_SIZE),
        "network": str(getattr(cfg_node.NETWORK, "TYPE", "standard")),
        "backbone": str(cfg_node.BACKBONE.TYPE),
        "module": str(cfg_node.MODULE.TYPE),
        "classifier": str(cfg_node.CLASSIFIER.TYPE),
        "classifier_bias": bool(cfg_node.CLASSIFIER.BIAS),
        "loss": str(cfg_node.LOSS.LOSS_TYPE),
        "sampler": str(cfg_node.TRAIN.SAMPLER.TYPE),
        "weighted_sampler": str(
            cfg_node.TRAIN.SAMPLER.WEIGHTED_SAMPLER.TYPE
        ),
        "bbn_sampler": str(cfg_node.TRAIN.SAMPLER.BBN_SAMPLER.TYPE),
        "train_transforms": tuple(cfg_node.TRANSFORMS.TRAIN_TRANSFORMS),
        "test_transforms": tuple(cfg_node.TRANSFORMS.TEST_TRANSFORMS),
        "batch_size": int(cfg_node.TRAIN.BATCH_SIZE),
        "max_epoch": int(cfg_node.TRAIN.MAX_EPOCH),
        "optimizer": str(cfg_node.TRAIN.OPTIMIZER.TYPE),
        "base_lr": float(cfg_node.TRAIN.OPTIMIZER.BASE_LR),
        "momentum": float(cfg_node.TRAIN.OPTIMIZER.MOMENTUM),
        "weight_decay": float(cfg_node.TRAIN.OPTIMIZER.WEIGHT_DECAY),
        "scheduler": str(cfg_node.TRAIN.LR_SCHEDULER.TYPE),
        "lr_steps": tuple(cfg_node.TRAIN.LR_SCHEDULER.LR_STEP),
        "lr_factor": float(cfg_node.TRAIN.LR_SCHEDULER.LR_FACTOR),
        "warm_epoch": int(cfg_node.TRAIN.LR_SCHEDULER.WARM_EPOCH),
    }


def _validate_comparison_configs(cfg_raw, cfg_fcc, cfg_fccr, allow_mismatch=False):
    signatures = {
        "raw": _comparison_signature(cfg_raw),
        "fcc": _comparison_signature(cfg_fcc),
        "fccr": _comparison_signature(cfg_fccr),
    }
    if not allow_mismatch:
        differences = []
        for method in ("fcc", "fccr"):
            for key, raw_value in signatures["raw"].items():
                method_value = signatures[method][key]
                if method_value != raw_value:
                    differences.append(
                        f"{method}.{key}={method_value!r} vs raw={raw_value!r}"
                    )
        if differences:
            raise ValueError(
                "Non-FCC training settings differ across configurations:\n  "
                + "\n  ".join(differences)
                + "\nUse --allow_config_mismatch only if this is intentional."
            )

    if bool(cfg_raw.FCC.WORK) or bool(cfg_raw.FCCR.WORK):
        raise ValueError("Raw config must disable both FCC.WORK and FCCR.WORK")
    if not bool(cfg_fcc.FCC.WORK) or bool(cfg_fcc.FCCR.WORK):
        raise ValueError("FCC config must enable FCC only")
    if not bool(cfg_fccr.FCCR.WORK) or bool(cfg_fccr.FCC.WORK):
        raise ValueError("FCCR config must enable FCCR only")

    combiner_types = {
        "raw": str(cfg_raw.TRAIN.COMBINER.TYPE).lower(),
        "fcc": str(cfg_fcc.TRAIN.COMBINER.TYPE).lower(),
        "fccr": str(cfg_fccr.TRAIN.COMBINER.TYPE).lower(),
    }
    expected_combiners = {"raw": "default", "fcc": "fcc", "fccr": "fccr"}
    for method, expected in expected_combiners.items():
        if combiner_types[method] != expected:
            raise ValueError(
                f"{method} config must use TRAIN.COMBINER.TYPE={expected!r}, "
                f"got {combiner_types[method]!r}"
            )

    fcc_setting = (float(cfg_fcc.FCC.GAMMA), int(cfg_fcc.FCC.START_EPOCH))
    fccr_setting = (
        float(cfg_fccr.FCCR.GAMMA),
        int(cfg_fccr.FCCR.START_EPOCH),
    )
    if fcc_setting != fccr_setting and not allow_mismatch:
        raise ValueError(
            "FCC and FCCR must use identical GAMMA and START_EPOCH: "
            f"FCC={fcc_setting}, FCCR={fccr_setting}."
        )
    return signatures


def _validate_paths_and_configs(args):
    paths = {
        "raw": Path(args.cfg_raw).resolve(),
        "fcc": Path(args.cfg_fcc).resolve(),
        "fccr": Path(args.cfg_fccr).resolve(),
    }
    for method, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{method} config does not exist: {path}")
    configs = {method: _load_cfg(path) for method, path in paths.items()}
    signatures = _validate_comparison_configs(
        configs["raw"],
        configs["fcc"],
        configs["fccr"],
        allow_mismatch=args.allow_config_mismatch,
    )
    for method in ("fcc", "fccr"):
        start_epoch = int(
            configs[method].FCC.START_EPOCH
            if method == "fcc"
            else configs[method].FCCR.START_EPOCH
        )
        if start_epoch > int(configs[method].TRAIN.MAX_EPOCH):
            raise ValueError(
                f"{method} START_EPOCH={start_epoch} exceeds MAX_EPOCH="
                f"{int(configs[method].TRAIN.MAX_EPOCH)}"
            )
    return paths, configs, signatures


def parse_args():
    parser = argparse.ArgumentParser(
        description="R1-5 high-dimensional feature scatter and center-shift audit"
    )
    parser.add_argument("--cfg_raw", required=True)
    parser.add_argument("--cfg_fcc", required=True)
    parser.add_argument("--cfg_fccr", required=True)
    parser.add_argument("--class_names", default="Negligible,Medium,Large")
    parser.add_argument("--large_class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--tag", default="")
    parser.add_argument(
        "--allow_config_mismatch",
        action="store_true",
        help="Allow non-FCC training settings to differ (not recommended for causal comparison)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import _init_paths  # noqa: F401 - adds lib/ to sys.path
    import torch

    from utils.repro import setup_repro

    if not torch.cuda.is_available():
        raise RuntimeError("Feature extraction requires an available CUDA device")
    setup_repro(args.seed)
    device = torch.device("cuda")

    config_paths, configs, signatures = _validate_paths_and_configs(args)
    class_names = [name.strip() for name in args.class_names.split(",")]
    if any(not name for name in class_names):
        raise ValueError("--class_names must contain non-empty comma-separated names")

    # A single Raw-config dataset instance is deliberately reused for all
    # checkpoints so sampling policies cannot change the evaluated corpus.
    train_set = _build_dataset("train", configs["raw"])
    inference_set = _build_dataset("valid", configs["raw"])
    num_classes = int(train_set.get_num_classes())
    if int(inference_set.get_num_classes()) != num_classes:
        raise ValueError("Train and inference datasets disagree on num_classes")
    if len(class_names) != num_classes:
        raise ValueError(
            f"Expected {num_classes} class names, got {len(class_names)}"
        )
    if args.large_class < 0 or args.large_class >= num_classes:
        raise ValueError(f"large_class must be in 0..{num_classes - 1}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        output_dir = (
            Path(configs["raw"].OUTPUT_DIR)
            / configs["raw"].NAME
            / "valid"
            / f"r1_5_feature_geometry{suffix}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scatter_rows = []
    all_center_rows = []
    summary_rows = []
    checkpoint_paths = {}

    for method in ("raw", "fcc", "fccr"):
        cfg_node = configs[method]
        model, checkpoint = _load_feature_model(cfg_node, num_classes, device)
        checkpoint_paths[method] = str(checkpoint)
        train_raw, train_transformed, train_labels = _extract_split_features(
            model,
            cfg_node,
            train_set,
            device,
            method,
            "train",
            args.seed,
        )
        inference_raw, _, inference_labels = _extract_split_features(
            model,
            cfg_node,
            inference_set,
            device,
            method,
            "valid",
            args.seed,
        )
        result = analyze_feature_geometry(
            train_raw,
            train_transformed,
            train_labels,
            inference_raw,
            inference_labels,
            class_names,
        )
        for row in result["scatter_rows"]:
            all_scatter_rows.append({"method": method, **row})
        for row in result["center_shift_rows"]:
            all_center_rows.append({"method": method, **row})
        summary_rows.append(_summary_row(method, result, args.large_class))

        del model, train_raw, train_transformed, inference_raw
        torch.cuda.empty_cache()

    _write_csv(output_dir / "intra_class_scatter.csv", all_scatter_rows)
    _write_csv(output_dir / "train_inference_center_shift.csv", all_center_rows)
    _write_csv(output_dir / "feature_geometry_summary.csv", summary_rows)

    manifest = {
        "experiment": "R1-5 original high-dimensional feature geometry",
        "loader_seed": args.seed,
        "lam_split_seed": SEED,
        "class_names": class_names,
        "large_class": args.large_class,
        "configs": {key: str(value) for key, value in config_paths.items()},
        "checkpoints": checkpoint_paths,
        "matched_training_signature": signatures["raw"],
        "train_samples": len(train_set),
        "inference_samples": len(inference_set),
        "definitions": {
            "primary_compactness": (
                "mean squared Euclidean radius of row-wise L2-normalized "
                "inference features around their class center"
            ),
            "center_gap": (
                "class-center L2 distance divided by the mean pairwise "
                "distance among inference class centers"
            ),
            "raw_train_to_inference": (
                "baseline train/validation split gap before the training-time transform"
            ),
            "transformed_train_to_inference": (
                "total gap after applying the configured FCC/FCCR transform"
            ),
            "transform_excess_gap": (
                "transformed gap minus raw gap; signed and normalized"
            ),
        },
        "limitations": [
            "Features are extracted from saved checkpoints in eval mode.",
            "The raw train-to-inference gap includes natural split shift; use the excess gap to isolate the transform contribution.",
            "FCCR in this repository uses batch class centers, not a persisted EMA center.",
            "A single checkpoint does not quantify training-seed uncertainty.",
        ],
    }
    with (output_dir / "experiment_manifest.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print("R1-5 feature-geometry audit completed")
    print(f"Results: {output_dir.resolve()}")
    for row in summary_rows:
        print(
            f"{row['method']:>4s} | Large scatter="
            f"{row['large_inference_scatter']:.6f} | "
            f"Large center gap(raw->transformed)="
            f"{row['large_raw_train_to_inference_gap']:.6f}->"
            f"{row['large_transformed_train_to_inference_gap']:.6f} | "
            f"excess={row['large_transform_excess_gap']:+.6f}"
        )


if __name__ == "__main__":
    main()
