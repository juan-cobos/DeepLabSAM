"""COCO keypoint scoring + CSV reporting shared by the pose evals.

``eval_pose_coco.py`` (SAM 3 boxes/masks + DLC pose head) and ``eval_dlc.py``
(stock SuperAnimal detector + pose) each produce COCO-format keypoint results and
then hand them here, so the two runs are scored by literally the same code and
their CSVs line up column for column.
"""

import numpy as np
import pandas as pd
from faster_coco_eval import COCOeval_faster


def n_labeled_gt(ann) -> int:
    return int(np.count_nonzero(np.asarray(ann["keypoints"])[2::3] > 0))


def score_and_report(
    coco_gt,
    results: list[dict],
    image_ids: list[int],
    sigmas: np.ndarray,
    gt_keypoints: list[str],
    csv_prefix,
    params: dict,
) -> None:
    """Score ``results`` with faster-coco-eval, print the summary, write the CSVs.

    Args:
        coco_gt: the loaded ground truth (``faster_coco_eval.COCO``).
        results: COCO keypoint results — one dict per detection.
        image_ids: the images that were actually run (scoring is restricted to them).
        sigmas: per-keypoint OKS sigmas, in the ground truth's keypoint order.
        gt_keypoints: bodypart names, for the per-bodypart breakdown.
        csv_prefix: ``<prefix>_metrics.csv`` and friends are written next to it.
        params: run settings (model, thresholds, ...) appended to the metrics CSV so
            a CSV is self-describing later on.
    """
    # Keypoints only: for a keypoint result file loadRes rewrites each bbox from the
    # keypoint extent, so a bbox eval here would score that, not the detector's boxes.
    # Use scripts/eval_coco.py for detection mAP.
    coco_dt = coco_gt.loadRes(results)
    # extra_calc writes the matched dt<->gt pairs (and each pair's OKS) back onto the
    # annotations, which is what the OKS reporting below reads.
    coco_eval = COCOeval_faster(
        coco_gt,
        coco_dt,
        "keypoints",
        kpt_oks_sigmas=sigmas,
        extra_calc=True,
        print_function=print,
    )
    coco_eval.params.imgIds = image_ids
    coco_eval.run()

    matched = [ann for ann in coco_dt.anns.values() if ann.get("tp")]
    n_gt = len(coco_gt.getAnnIds(imgIds=image_ids))
    if not matched:
        raise SystemExit("no detection matched a ground truth; no OKS to report")

    # Per-keypoint OKS terms for the matched pairs, same formula COCO sums over:
    #   oks_k = exp(-d_k^2 / (2 * (2*sigma_k)^2 * area))
    # kept unsummed so the mean per bodypart shows *which* joints the head misses.
    img_info = {i: coco_gt.loadImgs(i)[0] for i in image_ids}
    per_kpt_oks = np.full((len(matched), len(sigmas)), np.nan)
    keypoint_rows = []
    for i, dt in enumerate(matched):
        gt = coco_gt.anns[dt["gt_id"]]
        g = np.asarray(gt["keypoints"], dtype=float).reshape(-1, 3)
        d = np.asarray(dt["keypoints"], dtype=float).reshape(-1, 3)
        labeled = g[:, 2] > 0  # v == -1 is gradient-masked in this dataset
        dist2 = ((d[:, 0] - g[:, 0]) ** 2 + (d[:, 1] - g[:, 1]) ** 2)[labeled]
        e = dist2 / (2 * sigmas[labeled] ** 2) / 4 / (gt["area"] + np.spacing(1))
        per_kpt_oks[i, labeled] = np.exp(-e)
        for k in np.flatnonzero(labeled):
            keypoint_rows.append(
                {
                    "image_id": dt["image_id"],
                    "file_name": img_info[dt["image_id"]]["file_name"],
                    "dt_ann_id": dt["id"],
                    "gt_ann_id": gt["id"],
                    "bodypart": gt_keypoints[k],
                    "oks": round(float(per_kpt_oks[i, k]), 6),
                    "pixel_error": round(
                        float(np.hypot(d[k, 0] - g[k, 0], d[k, 1] - g[k, 1])), 3
                    ),
                    "gt_x": round(float(g[k, 0]), 3),
                    "gt_y": round(float(g[k, 1]), 3),
                    "gt_visibility": int(g[k, 2]),
                    "pred_x": round(float(d[k, 0]), 3),
                    "pred_y": round(float(d[k, 1]), 3),
                    "pred_confidence": round(float(d[k, 2]), 6),
                }
            )

    oks = np.array([dt["iou"] for dt in matched])
    print(f"\nOKS over {len(matched)} matched pairs ({n_gt} ground truths):")
    print(f"  mean   {oks.mean():.3f}   (mIoU {coco_eval.compute_mIoU():.3f})")
    print(f"  median {np.median(oks):.3f}   min {oks.min():.3f}   max {oks.max():.3f}")
    for thr in (0.5, 0.75, 0.9):
        print(f"  OKS >= {thr:.2f}: {(oks >= thr).mean():.1%}")

    # Bodyparts nobody labeled (n=0) get a NaN mean rather than a 0 that reads as a
    # failure; argsort puts those last, so the worst real joints come first.
    counts = np.count_nonzero(~np.isnan(per_kpt_oks), axis=0)
    means = np.where(
        counts > 0, np.nansum(per_kpt_oks, axis=0) / np.maximum(counts, 1), np.nan
    )
    print("\nMean OKS per bodypart, worst first (n = labeled ground truths):")
    for k in np.argsort(means):
        score = "    - " if counts[k] == 0 else f"{means[k]:.3f}"
        print(f"  {gt_keypoints[k]:<16} {score}  (n={counts[k]})")

    # --- CSVs ---------------------------------------------------------------
    # One row per detection *and* per missed ground truth, so the misses (whole
    # images where the detector found nothing) are in the table rather than implied
    # by absent rows.
    false_positives = [ann for ann in coco_dt.anns.values() if ann.get("fp")]
    kept = set(image_ids)
    missed = [
        ann
        for ann in coco_gt.anns.values()
        if ann.get("fn") and ann["image_id"] in kept  # fn is flagged across all images
    ]

    match_rows = [
        {
            "image_id": dt["image_id"],
            "file_name": img_info[dt["image_id"]]["file_name"],
            "match": "tp",
            "dt_ann_id": dt["id"],
            "gt_ann_id": dt["gt_id"],
            "detection_score": round(float(dt["score"]), 6),
            "oks": round(float(dt["iou"]), 6),
            "gt_area": round(float(coco_gt.anns[dt["gt_id"]]["area"]), 3),
            "n_labeled_keypoints": n_labeled_gt(coco_gt.anns[dt["gt_id"]]),
        }
        for dt in matched
    ]
    match_rows += [
        {
            "image_id": dt["image_id"],
            "file_name": img_info[dt["image_id"]]["file_name"],
            "match": "fp",
            "dt_ann_id": dt["id"],
            "gt_ann_id": "",
            "detection_score": round(float(dt["score"]), 6),
            "oks": "",
            "gt_area": "",
            "n_labeled_keypoints": "",
        }
        for dt in false_positives
    ]
    match_rows += [
        {
            "image_id": gt["image_id"],
            "file_name": img_info[gt["image_id"]]["file_name"],
            "match": "fn",
            "dt_ann_id": "",
            "gt_ann_id": gt["id"],
            "detection_score": "",
            "oks": "",
            "gt_area": round(float(gt["area"]), 3),
            "n_labeled_keypoints": n_labeled_gt(gt),
        }
        for gt in missed
    ]
    match_rows.sort(key=lambda r: (r["image_id"], r["match"]))

    bodypart_rows = [
        {
            "bodypart": gt_keypoints[k],
            "mean_oks": "" if counts[k] == 0 else round(float(means[k]), 6),
            "n_labeled": int(counts[k]),
        }
        for k in np.argsort(means)
    ]

    metric_rows = [
        {"metric": k, "value": round(float(v), 6)}
        for k, v in coco_eval.stats_as_dict.items()
    ]
    metric_rows += [
        {"metric": k, "value": v}
        for k, v in {
            "oks_mean": round(float(oks.mean()), 6),
            "oks_median": round(float(np.median(oks)), 6),
            "oks_min": round(float(oks.min()), 6),
            "oks_max": round(float(oks.max()), 6),
            "oks_frac_ge_0.50": round(float((oks >= 0.5).mean()), 6),
            "oks_frac_ge_0.75": round(float((oks >= 0.75).mean()), 6),
            "oks_frac_ge_0.90": round(float((oks >= 0.9).mean()), 6),
            "n_images": len(image_ids),
            "n_ground_truths": n_gt,
            "n_detections": len(results),
            "n_true_positives": len(matched),
            "n_false_positives": len(false_positives),
            "n_missed": len(missed),
            **params,
        }.items()
    ]

    def write_csv(suffix: str, rows: list[dict]):
        path = csv_prefix.with_name(f"{csv_prefix.name}_{suffix}.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    print()
    for suffix, rows in (
        ("metrics", metric_rows),
        ("matches", match_rows),
        ("bodyparts", bodypart_rows),
        ("keypoints", keypoint_rows),
    ):
        print(f"wrote {len(rows)} rows -> {write_csv(suffix, rows)}")
