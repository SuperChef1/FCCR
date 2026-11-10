"""R1-2 sample-transition audit for Raw, FCC, and FCCR.

The script reuses the existing three-configuration comparison inference path,
then quantifies which individual samples are broken or recovered. Particular
attention is given to false negatives of the safety-critical Large class.
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np


EVENT_DEFINITIONS = (
    (
        "raw_correct_to_fcc_wrong",
        "Raw is correct but FCC changes the sample to an error",
    ),
    (
        "fcc_wrong_to_fccr_correct",
        "FCC is wrong and FCCR recovers the correct label",
    ),
    (
        "fcc_correct_to_fccr_wrong",
        "FCC is correct but FCCR introduces an error (broad definition)",
    ),
    (
        "raw_and_fcc_correct_to_fccr_wrong",
        "Raw and FCC are both correct but FCCR introduces an error (strict definition)",
    ),
)


def _as_int_vector(name, values):
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    return array


def _pct(numerator, denominator):
    return 0.0 if denominator == 0 else 100.0 * numerator / denominator


def _validate_class_names(class_names, num_classes):
    if class_names is None:
        return [f"class_{index}" for index in range(num_classes)]
    names = [str(name).strip() for name in class_names]
    if len(names) != num_classes or any(not name for name in names):
        raise ValueError(
            f"Expected {num_classes} non-empty class names, got {class_names!r}"
        )
    return names


def analyze_error_transitions(
    labels,
    preds_raw,
    preds_fcc,
    preds_fccr,
    image_ids=None,
    large_cls=2,
    class_names=None,
):
    """Compute requested sample transitions without any model dependency."""
    labels = _as_int_vector("labels", labels)
    preds_raw = _as_int_vector("preds_raw", preds_raw)
    preds_fcc = _as_int_vector("preds_fcc", preds_fcc)
    preds_fccr = _as_int_vector("preds_fccr", preds_fccr)
    size = labels.size
    if not (preds_raw.size == preds_fcc.size == preds_fccr.size == size):
        raise ValueError("Labels and all prediction arrays must have the same length")
    if size == 0:
        raise ValueError("Cannot analyze an empty validation set")

    if image_ids is None:
        image_ids = np.arange(size, dtype=np.int64)
    else:
        image_ids = _as_int_vector("image_ids", image_ids)
        if image_ids.size != size:
            raise ValueError("image_ids must have the same length as labels")

    max_label = int(
        max(labels.max(), preds_raw.max(), preds_fcc.max(), preds_fccr.max())
    )
    num_classes = max_label + 1
    if large_cls < 0 or large_cls >= num_classes:
        raise ValueError(
            f"large_cls={large_cls} is outside the observed class range 0..{num_classes - 1}"
        )
    class_names = _validate_class_names(class_names, num_classes)

    raw_ok = preds_raw == labels
    fcc_ok = preds_fcc == labels
    fccr_ok = preds_fccr == labels
    large_mask = labels == large_cls

    events = {
        "raw_correct_to_fcc_wrong": raw_ok & ~fcc_ok,
        "fcc_wrong_to_fccr_correct": ~fcc_ok & fccr_ok,
        "fcc_correct_to_fccr_wrong": fcc_ok & ~fccr_ok,
        "raw_and_fcc_correct_to_fccr_wrong": raw_ok & fcc_ok & ~fccr_ok,
    }
    sources = {
        "raw_correct_to_fcc_wrong": raw_ok,
        "fcc_wrong_to_fccr_correct": ~fcc_ok,
        "fcc_correct_to_fccr_wrong": fcc_ok,
        "raw_and_fcc_correct_to_fccr_wrong": raw_ok & fcc_ok,
    }

    event_summary_rows = []
    for scope_name, scope_mask in (("all", np.ones(size, dtype=bool)), ("large", large_mask)):
        scope_count = int(scope_mask.sum())
        for event_name, _ in EVENT_DEFINITIONS:
            event_count = int((events[event_name] & scope_mask).sum())
            source_count = int((sources[event_name] & scope_mask).sum())
            event_summary_rows.append(
                [
                    scope_name,
                    event_name,
                    event_count,
                    source_count,
                    _pct(event_count, source_count),
                    scope_count,
                    _pct(event_count, scope_count),
                ]
            )

    model_predictions = {
        "raw": preds_raw,
        "fcc": preds_fcc,
        "fccr": preds_fccr,
    }
    large_count = int(large_mask.sum())
    large_model_rows = []
    large_destination_rows = []
    for model_name, predictions in model_predictions.items():
        correct_count = int(((predictions == labels) & large_mask).sum())
        false_negative_mask = large_mask & (predictions != large_cls)
        false_negative_count = int(false_negative_mask.sum())
        large_model_rows.append(
            [
                model_name,
                large_cls,
                class_names[large_cls],
                large_count,
                correct_count,
                _pct(correct_count, large_count),
                false_negative_count,
                _pct(false_negative_count, large_count),
            ]
        )
        for predicted_class in range(num_classes):
            if predicted_class == large_cls:
                continue
            count = int(
                (false_negative_mask & (predictions == predicted_class)).sum()
            )
            large_destination_rows.append(
                [
                    model_name,
                    large_cls,
                    class_names[large_cls],
                    predicted_class,
                    class_names[predicted_class],
                    count,
                    false_negative_count,
                    _pct(count, false_negative_count),
                    large_count,
                    _pct(count, large_count),
                ]
            )

    large_fn_raw = int((large_mask & ~raw_ok).sum())
    large_fn_fcc = int((large_mask & ~fcc_ok).sum())
    large_fn_fccr = int((large_mask & ~fccr_ok).sum())
    large_broken_by_fcc = int(
        (large_mask & events["raw_correct_to_fcc_wrong"]).sum()
    )
    large_recovered_by_fccr = int(
        (large_mask & events["fcc_wrong_to_fccr_correct"]).sum()
    )
    large_new_by_fccr = int(
        (large_mask & events["fcc_correct_to_fccr_wrong"]).sum()
    )
    large_strict_new_by_fccr = int(
        (large_mask & events["raw_and_fcc_correct_to_fccr_wrong"]).sum()
    )
    large_flow_rows = [
        ["raw_large_false_negatives", large_fn_raw],
        ["fcc_large_false_negatives", large_fn_fcc],
        ["fccr_large_false_negatives", large_fn_fccr],
        ["raw_correct_large_broken_by_fcc", large_broken_by_fcc],
        ["fcc_large_misses_recovered_by_fccr", large_recovered_by_fccr],
        ["fcc_correct_large_newly_missed_by_fccr", large_new_by_fccr],
        ["strict_large_newly_missed_by_fccr", large_strict_new_by_fccr],
        ["fcc_minus_raw_large_false_negatives", large_fn_fcc - large_fn_raw],
        ["fccr_minus_fcc_large_false_negatives", large_fn_fccr - large_fn_fcc],
        [
            "new_minus_recovered_large_misses_in_fccr",
            large_new_by_fccr - large_recovered_by_fccr,
        ],
    ]

    transition_rows = []
    scopes = [("all", -1, "all", np.ones(size, dtype=bool))]
    scopes.extend(
        (
            "class",
            class_id,
            class_names[class_id],
            labels == class_id,
        )
        for class_id in range(num_classes)
    )
    for scope, class_id, class_name, scope_mask in scopes:
        scope_count = int(scope_mask.sum())
        for raw_state in (0, 1):
            for fcc_state in (0, 1):
                for fccr_state in (0, 1):
                    mask = (
                        scope_mask
                        & (raw_ok == bool(raw_state))
                        & (fcc_ok == bool(fcc_state))
                        & (fccr_ok == bool(fccr_state))
                    )
                    count = int(mask.sum())
                    transition_rows.append(
                        [
                            scope,
                            class_id,
                            class_name,
                            raw_state,
                            fcc_state,
                            fccr_state,
                            count,
                            scope_count,
                            _pct(count, scope_count),
                        ]
                    )

    sample_rows = []
    event_sample_rows = []
    for index in range(size):
        flags = {name: int(mask[index]) for name, mask in events.items()}
        base_row = [
            index,
            int(image_ids[index]),
            int(labels[index]),
            class_names[int(labels[index])],
            int(preds_raw[index]),
            class_names[int(preds_raw[index])],
            int(raw_ok[index]),
            int(preds_fcc[index]),
            class_names[int(preds_fcc[index])],
            int(fcc_ok[index]),
            int(preds_fccr[index]),
            class_names[int(preds_fccr[index])],
            int(fccr_ok[index]),
            int(large_mask[index]),
            int(large_mask[index] and not raw_ok[index]),
            int(large_mask[index] and not fcc_ok[index]),
            int(large_mask[index] and not fccr_ok[index]),
        ]
        sample_rows.append(base_row + [flags[name] for name, _ in EVENT_DEFINITIONS])
        for event_name, _ in EVENT_DEFINITIONS:
            if flags[event_name]:
                event_sample_rows.append([event_name] + base_row)

    return {
        "num_samples": size,
        "num_classes": num_classes,
        "class_names": class_names,
        "large_cls": large_cls,
        "large_count": large_count,
        "event_summary_rows": event_summary_rows,
        "large_model_rows": large_model_rows,
        "large_destination_rows": large_destination_rows,
        "large_flow_rows": large_flow_rows,
        "transition_rows": transition_rows,
        "sample_rows": sample_rows,
        "event_sample_rows": event_sample_rows,
        "large_event_sample_rows": [row for row in event_sample_rows if row[14] == 1],
    }


def save_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def write_outputs(result, output_dir, config_info):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_csv(
        output_dir / "event_definitions.csv",
        ["event", "definition"],
        EVENT_DEFINITIONS,
    )
    save_csv(
        output_dir / "event_summary.csv",
        [
            "scope",
            "event",
            "count",
            "source_count",
            "source_rate_pct",
            "scope_count",
            "scope_rate_pct",
        ],
        result["event_summary_rows"],
    )
    save_csv(
        output_dir / "large_false_negative_summary.csv",
        [
            "model",
            "large_class_id",
            "large_class_name",
            "large_count",
            "correct_count",
            "large_recall_pct",
            "false_negative_count",
            "false_negative_rate_pct",
        ],
        result["large_model_rows"],
    )
    save_csv(
        output_dir / "large_false_negative_destinations.csv",
        [
            "model",
            "true_class_id",
            "true_class_name",
            "predicted_class_id",
            "predicted_class_name",
            "count",
            "model_false_negative_count",
            "pct_of_model_false_negatives",
            "large_count",
            "pct_of_large_samples",
        ],
        result["large_destination_rows"],
    )
    save_csv(
        output_dir / "large_miss_flow.csv",
        ["metric", "count_or_delta"],
        result["large_flow_rows"],
    )
    save_csv(
        output_dir / "correctness_transitions_by_class.csv",
        [
            "scope",
            "class_id",
            "class_name",
            "raw_correct",
            "fcc_correct",
            "fccr_correct",
            "count",
            "scope_count",
            "scope_pct",
        ],
        result["transition_rows"],
    )

    base_sample_header = [
        "dataset_index",
        "image_id",
        "true_label",
        "true_class",
        "pred_raw",
        "pred_raw_class",
        "raw_correct",
        "pred_fcc",
        "pred_fcc_class",
        "fcc_correct",
        "pred_fccr",
        "pred_fccr_class",
        "fccr_correct",
        "is_large",
        "raw_large_miss",
        "fcc_large_miss",
        "fccr_large_miss",
    ]
    save_csv(
        output_dir / "all_sample_predictions.csv",
        base_sample_header + [name for name, _ in EVENT_DEFINITIONS],
        result["sample_rows"],
    )
    save_csv(
        output_dir / "event_samples.csv",
        ["event"] + base_sample_header,
        result["event_sample_rows"],
    )
    save_csv(
        output_dir / "large_event_samples.csv",
        ["event"] + base_sample_header,
        result["large_event_sample_rows"],
    )

    event_lookup = {
        (row[0], row[1]): row for row in result["event_summary_rows"]
    }
    large_metrics = {row[0]: row for row in result["large_model_rows"]}
    flow = {row[0]: row[1] for row in result["large_flow_rows"]}
    with (output_dir / "summary.txt").open("w", encoding="utf-8") as stream:
        stream.write("R1-2 Raw/FCC/FCCR sample-transition audit\n")
        stream.write("=" * 58 + "\n")
        for key, value in config_info.items():
            stream.write(f"{key}: {value}\n")
        stream.write(
            f"N={result['num_samples']}, Large class={result['large_cls']} "
            f"({result['class_names'][result['large_cls']]}), "
            f"Large N={result['large_count']}\n\n"
        )

        stream.write("Requested correctness transitions\n")
        for event_name, definition in EVENT_DEFINITIONS:
            all_row = event_lookup[("all", event_name)]
            large_row = event_lookup[("large", event_name)]
            stream.write(f"- {event_name}: {definition}\n")
            stream.write(
                f"  all={all_row[2]}/{all_row[3]} ({all_row[4]:.2f}% of source), "
                f"Large={large_row[2]}/{large_row[3]} "
                f"({large_row[4]:.2f}% of Large source)\n"
            )

        stream.write("\nLarge-class false negatives\n")
        for model_name in ("raw", "fcc", "fccr"):
            row = large_metrics[model_name]
            stream.write(
                f"- {model_name}: recall={row[5]:.2f}%, "
                f"misses={row[6]}/{row[3]} ({row[7]:.2f}%)\n"
            )
        stream.write(
            "- FCCR vs FCC: recovered Large misses="
            f"{flow['fcc_large_misses_recovered_by_fccr']}, "
            "new Large misses="
            f"{flow['fcc_correct_large_newly_missed_by_fccr']}, "
            "net false-negative change="
            f"{flow['fccr_minus_fcc_large_false_negatives']}\n"
        )
        stream.write(
            "\nInterpretation note: these are paired, sample-level descriptive "
            "counts for the selected checkpoints and split; they do not by "
            "themselves establish statistical significance.\n"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        "R1-2 paired sample-transition audit for Raw/FCC/FCCR"
    )
    parser.add_argument("--cfg_raw", "--cfg_base", dest="cfg_raw", required=True)
    parser.add_argument("--cfg_fcc", required=True)
    parser.add_argument("--cfg_fccr", required=True)
    parser.add_argument("--gpus", default="0")
    parser.add_argument("--large_cls", type=int, default=2)
    parser.add_argument(
        "--class_names",
        default="Negligible,Medium,Large",
        help="Comma-separated class names in label-ID order",
    )
    parser.add_argument(
        "--tag",
        default="",
        help="Optional output suffix, e.g. gamma_0p5_start1_seed111",
    )
    parser.add_argument(
        "--output_dir",
        default="",
        help="Optional explicit output directory",
    )
    parser.add_argument(
        "--allow_config_mismatch",
        action="store_true",
        help="Allow non-FCC training settings to differ (not recommended for R1-2)",
    )
    return parser.parse_args()


def _comparison_signature(cfg):
    """Settings that must match for a paired mechanism comparison."""
    return {
        "dataset": str(cfg.DATASET.DATASET),
        "input_size": tuple(cfg.INPUT_SIZE),
        "network": str(getattr(cfg.NETWORK, "TYPE", "standard")),
        "backbone": str(cfg.BACKBONE.TYPE),
        "module": str(cfg.MODULE.TYPE),
        "classifier": str(cfg.CLASSIFIER.TYPE),
        "classifier_bias": bool(cfg.CLASSIFIER.BIAS),
        "loss": str(cfg.LOSS.LOSS_TYPE),
        "batch_size": int(cfg.TRAIN.BATCH_SIZE),
        "max_epoch": int(cfg.TRAIN.MAX_EPOCH),
        "optimizer": str(cfg.TRAIN.OPTIMIZER.TYPE),
        "base_lr": float(cfg.TRAIN.OPTIMIZER.BASE_LR),
        "momentum": float(cfg.TRAIN.OPTIMIZER.MOMENTUM),
        "weight_decay": float(cfg.TRAIN.OPTIMIZER.WEIGHT_DECAY),
        "scheduler": str(cfg.TRAIN.LR_SCHEDULER.TYPE),
        "lr_steps": tuple(cfg.TRAIN.LR_SCHEDULER.LR_STEP),
        "lr_factor": float(cfg.TRAIN.LR_SCHEDULER.LR_FACTOR),
        "warm_epoch": int(cfg.TRAIN.LR_SCHEDULER.WARM_EPOCH),
    }


def _validate_comparison_configs(cfg_raw, cfg_fcc, cfg_fccr, allow_mismatch=False):
    signatures = {
        "raw": _comparison_signature(cfg_raw),
        "fcc": _comparison_signature(cfg_fcc),
        "fccr": _comparison_signature(cfg_fccr),
    }
    if not allow_mismatch:
        reference = signatures["raw"]
        differences = []
        for model_name in ("fcc", "fccr"):
            for key, raw_value in reference.items():
                model_value = signatures[model_name][key]
                if model_value != raw_value:
                    differences.append(
                        f"{model_name}.{key}={model_value!r} vs raw={raw_value!r}"
                    )
        if differences:
            raise ValueError(
                "Non-FCC training settings differ across configurations:\n  "
                + "\n  ".join(differences)
                + "\nUse --allow_config_mismatch only if this is intentional."
            )

    if bool(cfg_raw.FCC.WORK) or bool(cfg_raw.FCCR.WORK):
        raise ValueError("Raw configuration must have both FCC.WORK and FCCR.WORK disabled")
    if not bool(cfg_fcc.FCC.WORK):
        raise ValueError("FCC configuration must set FCC.WORK=True")
    if not bool(cfg_fccr.FCCR.WORK):
        raise ValueError("FCCR configuration must set FCCR.WORK=True")

    fcc_setting = (float(cfg_fcc.FCC.GAMMA), int(cfg_fcc.FCC.START_EPOCH))
    fccr_setting = (float(cfg_fccr.FCCR.GAMMA), int(cfg_fccr.FCCR.START_EPOCH))
    if fcc_setting != fccr_setting and not allow_mismatch:
        raise ValueError(
            "FCC and FCCR must use the same GAMMA and START_EPOCH for the "
            f"realignment comparison: FCC={fcc_setting}, FCCR={fccr_setting}. "
            "Use --allow_config_mismatch only if this is intentional."
        )
    return signatures


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    import _init_paths  # noqa: F401 - adds lib/ to sys.path
    import torch

    from compare import build_dataset, infer_one_cfg, load_cfg

    if not torch.cuda.is_available():
        raise RuntimeError("This comparison path requires an available CUDA device")
    device = torch.device("cuda")

    cfg_paths = {
        "raw_cfg": os.path.abspath(args.cfg_raw),
        "fcc_cfg": os.path.abspath(args.cfg_fcc),
        "fccr_cfg": os.path.abspath(args.cfg_fccr),
    }
    for name, path in cfg_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")

    cfg_raw = load_cfg(cfg_paths["raw_cfg"])
    cfg_fcc = load_cfg(cfg_paths["fcc_cfg"])
    cfg_fccr = load_cfg(cfg_paths["fccr_cfg"])
    signatures = _validate_comparison_configs(
        cfg_raw,
        cfg_fcc,
        cfg_fccr,
        allow_mismatch=args.allow_config_mismatch,
    )

    checkpoint_paths = {
        "raw_checkpoint": str(
            Path(cfg_raw.OUTPUT_DIR)
            / cfg_raw.NAME
            / "models"
            / cfg_raw.TEST.MODEL_FILE
        ),
        "fcc_checkpoint": str(
            Path(cfg_fcc.OUTPUT_DIR)
            / cfg_fcc.NAME
            / "models"
            / cfg_fcc.TEST.MODEL_FILE
        ),
        "fccr_checkpoint": str(
            Path(cfg_fccr.OUTPUT_DIR)
            / cfg_fccr.NAME
            / "models"
            / cfg_fccr.TEST.MODEL_FILE
        ),
    }
    for name, path in checkpoint_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} does not exist: {os.path.abspath(path)}")

    valid_set = build_dataset("valid", cfg_raw)

    preds_raw, labels_raw, ids_raw = infer_one_cfg(cfg_raw, device, valid_set)
    preds_fcc, labels_fcc, ids_fcc = infer_one_cfg(cfg_fcc, device, valid_set)
    preds_fccr, labels_fccr, ids_fccr = infer_one_cfg(cfg_fccr, device, valid_set)

    if not (
        np.array_equal(labels_raw, labels_fcc)
        and np.array_equal(labels_raw, labels_fccr)
    ):
        raise RuntimeError("Raw/FCC/FCCR labels are not aligned sample by sample")
    if not (
        np.array_equal(ids_raw, ids_fcc)
        and np.array_equal(ids_raw, ids_fccr)
    ):
        raise RuntimeError("Raw/FCC/FCCR image IDs are not aligned sample by sample")

    class_names = [name.strip() for name in args.class_names.split(",")]
    result = analyze_error_transitions(
        labels_raw,
        preds_raw,
        preds_fcc,
        preds_fccr,
        image_ids=ids_raw,
        large_cls=args.large_cls,
        class_names=class_names,
    )

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        suffix = f"_{args.tag}" if args.tag else ""
        output_dir = (
            Path(cfg_raw.OUTPUT_DIR)
            / cfg_raw.NAME
            / "valid"
            / f"r1_2_error_transitions{suffix}"
        )

    config_info = {**cfg_paths, **checkpoint_paths}
    config_info["matched_training_signature"] = signatures["raw"]
    config_info["fcc_setting"] = (
        float(cfg_fcc.FCC.GAMMA),
        int(cfg_fcc.FCC.START_EPOCH),
    )
    config_info["fccr_setting"] = (
        float(cfg_fccr.FCCR.GAMMA),
        int(cfg_fccr.FCCR.START_EPOCH),
    )
    write_outputs(result, output_dir, config_info)

    print("R1-2 error-transition audit completed")
    print(f"Results: {output_dir.resolve()}")
    for row in result["large_model_rows"]:
        print(
            f"{row[0]:>4s} Large recall={row[5]:6.2f}% | "
            f"misses={row[6]}/{row[3]} ({row[7]:.2f}%)"
        )
    flow = {row[0]: row[1] for row in result["large_flow_rows"]}
    print(
        "FCCR vs FCC Large misses: "
        f"recovered={flow['fcc_large_misses_recovered_by_fccr']}, "
        f"new={flow['fcc_correct_large_newly_missed_by_fccr']}, "
        f"net_change={flow['fccr_minus_fcc_large_false_negatives']}"
    )


if __name__ == "__main__":
    main()
