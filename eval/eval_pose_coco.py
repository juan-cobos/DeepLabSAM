import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from faster_coco_eval import COCO, COCOeval_faster
from PIL import Image
from tqdm import tqdm
from transformers import Sam3VideoConfig

from deeplabsam.pose.backends.dlc import DLCPoseHead
from deeplabsam.segment.sam3video import Sam3VideoWrapper

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "dataset_dir", type=Path, help="root holding images/ and annotations/"
)
parser.add_argument("--split", default="test", help="annotations/<split>.json")
parser.add_argument("--text", default="mouse", help="text prompt for SAM 3")
parser.add_argument("--limit", type=int, default=None, help="cap the number of images")
parser.add_argument("--image-size", type=int, default=1008, help="SAM 3 input size")
parser.add_argument("--score-threshold", type=float, default=0.5)
parser.add_argument("--nms-threshold", type=float, default=0.5)
parser.add_argument("--super-animal", default="superanimal_topviewmouse")
parser.add_argument(
    "--checkpoint-path",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "checkpoints" / "sam3",
)
parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="write the COCO-format predictions here (default outputs/<split>_pose_preds.json)",
)
parser.add_argument(
    "--csv-prefix",
    type=Path,
    default=None,
    help="prefix for the result CSVs (default outputs/<split>_pose)",
)
args = parser.parse_args()

images_dir = args.dataset_dir / "images"
ann_path = args.dataset_dir / "annotations" / f"{args.split}.json"
out_path = args.out or Path("eval/outputs") / f"{args.split}_pose_preds.json"
csv_prefix = args.csv_prefix or Path("eval/outputs") / f"{args.split}_pose"

coco_gt = COCO(str(ann_path))
category_id = coco_gt.getCatIds()[0]
gt_keypoints = coco_gt.loadCats(category_id)[0]["keypoints"]
sigmas = np.array(
    json.loads((args.dataset_dir / "supertopview_dataset.json").read_text())[
        "dataset_info"
    ]["sigmas"]
)

config = Sam3VideoConfig.from_pretrained(str(args.checkpoint_path))
config.image_size = args.image_size
config.score_threshold_detection = args.score_threshold
predictor = Sam3VideoWrapper(checkpoint_path=args.checkpoint_path, config=config)
pose_head = DLCPoseHead(super_animal=args.super_animal)

# The pose head and the annotations must name the same 27 bodyparts in the same
# order, or the OKS pairing is silently comparing different joints.
if pose_head.bodyparts != gt_keypoints:
    raise SystemExit(
        "keypoint layout mismatch between pose head and annotations:\n"
        f"  head: {pipe.pose_head.bodyparts}\n"
        f"  gt:   {gt_keypoints}"
    )

image_ids = sorted(coco_gt.getImgIds())
if args.limit:
    # Evenly spaced rather than the first N: the ids are grouped by source dataset,
    # so a prefix is one recording's quirks (aspect ratio, lighting), not a sample.
    image_ids = image_ids[:: max(1, len(image_ids) // args.limit)][: args.limit]
results = []
for image_id in tqdm(image_ids):
    info = coco_gt.loadImgs(image_id)[0]
    image = Image.open(images_dir / info["file_name"]).convert("RGB")
    # HWC uint8 RGB on the GPU — the same layout the video decoder hands the pipeline.
    frame = torch.from_numpy(np.asarray(image)).to(predictor.device)

    # Fresh session per image: these are unrelated stills, not a tracked sequence.
    det = predictor.reset(args.text)(frame).to_detections()
    if len(det) == 0:
        continue
    det = det.with_nms(threshold=args.nms_threshold)
    kpts = pose_head.predict_tensor(frame, det.xyxy, det.mask)

    for keypoints, score in zip(kpts, det.confidence):
        results.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "keypoints": keypoints.reshape(-1).astype(float).tolist(),
                "score": float(score),
            }
        )

out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(results))
print(f"{len(results)} detections over {len(image_ids)} images -> {out_path}")
if not results:
    raise SystemExit("no detections; nothing to score")

# Keypoints only: for a keypoint result file loadRes rewrites each bbox from the
# keypoint extent, so a bbox eval here would score that, not SAM 3's boxes.
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

# --- CSVs -------------------------------------------------------------------
# One row per detection *and* per missed ground truth, so the misses (whole images
# where SAM 3 found nothing) are in the table rather than implied by absent rows.
false_positives = [ann for ann in coco_dt.anns.values() if ann.get("fp")]
kept = set(image_ids)
missed = [
    ann
    for ann in coco_gt.anns.values()
    if ann.get("fn") and ann["image_id"] in kept  # fn is flagged across all images
]


def n_labeled_gt(ann) -> int:
    return int(np.count_nonzero(np.asarray(ann["keypoints"])[2::3] > 0))


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
        # Recorded alongside the scores so a CSV is self-describing later on.
        "split": args.split,
        "text_prompt": args.text,
        "image_size": args.image_size,
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "super_animal": args.super_animal,
    }.items()
]


def write_csv(suffix: str, rows: list[dict]) -> Path:
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
