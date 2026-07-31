"""Score the stock SuperAnimal DeepLabCut pipeline on a COCO keypoint split.

The baseline eval_pose_coco.py is compared against: DLC's own top-down modelzoo
runners (a Faster R-CNN detector, then the SuperAnimal pose model on each box)
instead of SAM 3 boxes/masks + the DLC pose head. Scoring goes through the shared
``pose_metrics.score_and_report``, so the CSV pairs
``<split>_dlc_*.csv`` / ``<split>_pose_*.csv`` are directly comparable.

Needs the optional ``dlc`` extra (the deeplabcut package):

    uv run --extra dlc eval/eval_dlc.py ~/Downloads/TopViewMouse5K_20240913
"""

import argparse
import json
from pathlib import Path

import numpy as np
from deeplabcut.pose_estimation_pytorch.modelzoo.inference_helpers import (
    create_superanimal_inference_runners,
)
from faster_coco_eval import COCO
from PIL import Image
from pose_metrics import score_and_report
from tqdm import tqdm

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "dataset_dir", type=Path, help="root holding images/ and annotations/"
)
parser.add_argument("--split", default="test", help="annotations/<split>.json")
parser.add_argument("--limit", type=int, default=None, help="cap the number of images")
parser.add_argument("--super-animal", default="superanimal_topviewmouse")
parser.add_argument("--pose-model", default="hrnet_w32")
parser.add_argument("--detector", default="fasterrcnn_resnet50_fpn_v2")
parser.add_argument(
    "--max-individuals", type=int, default=10, help="boxes kept per image"
)
parser.add_argument(
    "--score-threshold",
    type=float,
    default=0.0,
    help="drop boxes below this detector score (0 keeps everything DLC returns)",
)
parser.add_argument("--device", default="auto")
parser.add_argument(
    "--out",
    type=Path,
    default=None,
    help="write the COCO-format predictions here (default outputs/<split>_dlc_preds.json)",
)
parser.add_argument(
    "--csv-prefix",
    type=Path,
    default=None,
    help="prefix for the result CSVs (default outputs/<split>_dlc)",
)
args = parser.parse_args()

images_dir = args.dataset_dir / "images"
ann_path = args.dataset_dir / "annotations" / f"{args.split}.json"
out_path = args.out or Path("eval/outputs") / f"{args.split}_dlc_preds.json"
csv_prefix = args.csv_prefix or Path("eval/outputs") / f"{args.split}_dlc"

coco_gt = COCO(str(ann_path))
category_id = coco_gt.getCatIds()[0]
gt_keypoints = coco_gt.loadCats(category_id)[0]["keypoints"]
sigmas = np.array(
    json.loads((args.dataset_dir / "supertopview_dataset.json").read_text())[
        "dataset_info"
    ]["sigmas"]
)


def dlc_bodyparts(model_cfg) -> list[str]:
    """The pose model's keypoint names, in the order it outputs them."""
    meta = model_cfg.get("metadata") or {}
    parts = meta.get("bodyparts") or model_cfg.get("bodyparts")
    if not parts:
        raise KeyError(f"no bodyparts in model_cfg (keys: {list(model_cfg)})")
    return list(parts)


def keypoint_order(bodyparts: list[str], gt_keypoints: list[str]) -> list[int]:
    """Indices that pull ``bodyparts`` into the ground truth's keypoint order.

    The model and the annotations both come from SuperAnimal-TopViewMouse, so this
    is usually the identity — but the pairing is what OKS is computed on, so match
    by name and fail loudly rather than trust the order.
    """
    index = {name: i for i, name in enumerate(bodyparts)}
    missing = [name for name in gt_keypoints if name not in index]
    if missing:
        raise SystemExit(
            "keypoint layout mismatch between pose model and annotations:\n"
            f"  model: {bodyparts}\n"
            f"  gt:    {gt_keypoints}\n"
            f"  the model predicts no {missing}"
        )
    return [index[name] for name in gt_keypoints]


def reorder_keypoints(keypoints: np.ndarray, order: list[int]) -> np.ndarray:
    """(N, K_model, 3) predictions -> (N, K_gt, 3) in the ground truth's order."""
    return keypoints[:, order, :]


def predict_frame(det_runner, pose_runner, rgb: np.ndarray) -> dict:
    """Detect and pose one RGB frame -> boxes/scores/keypoints, or {} if empty."""
    # Top-down: detect boxes, then predict a pose inside each box.
    det_pred = det_runner.inference([rgb])[0]
    if len(np.asarray(det_pred.get("bboxes", [])).reshape(-1, 4)) == 0:
        return {}

    # The pose prediction carries its own boxes/scores aligned 1:1 with the
    # bodyparts, all padded to max_individuals; padding rows have score -1. Read
    # them here (not the detector's output) so counts always line up.
    pose_pred = pose_runner.inference([(rgb, det_pred)])[0]
    boxes = np.asarray(pose_pred["bboxes"], dtype=float).reshape(-1, 4)  # xywh
    scores = np.asarray(pose_pred["bbox_scores"], dtype=float).reshape(-1)
    bodyparts = np.asarray(pose_pred["bodyparts"], dtype=float).reshape(
        len(scores), -1, 3
    )

    keep = scores > max(0.0, args.score_threshold)
    if not keep.any():
        return {}
    boxes, scores, bodyparts = boxes[keep], scores[keep], bodyparts[keep]
    xyxy = boxes.copy()
    xyxy[:, 2:] = boxes[:, :2] + boxes[:, 2:]  # xywh -> xyxy
    return {"boxes": xyxy, "scores": scores, "keypoints": bodyparts}


pose_runner, det_runner, model_cfg = create_superanimal_inference_runners(
    superanimal_name=args.super_animal,
    model_name=args.pose_model,
    detector_name=args.detector,
    max_individuals=args.max_individuals,
    device=args.device,
)
if det_runner is None:
    raise RuntimeError(
        f"{args.super_animal}/{args.pose_model} has no detector; top-down eval needs one"
    )
order = keypoint_order(dlc_bodyparts(model_cfg), gt_keypoints)

image_ids = sorted(coco_gt.getImgIds())
if args.limit:
    # Evenly spaced rather than the first N: the ids are grouped by source dataset,
    # so a prefix is one recording's quirks (aspect ratio, lighting), not a sample.
    image_ids = image_ids[:: max(1, len(image_ids) // args.limit)][: args.limit]
results = []
for image_id in tqdm(image_ids):
    info = coco_gt.loadImgs(image_id)[0]
    rgb = np.asarray(Image.open(images_dir / info["file_name"]).convert("RGB"))

    pred = predict_frame(det_runner, pose_runner, rgb)
    if not pred:
        continue

    for keypoints, score in zip(
        reorder_keypoints(pred["keypoints"], order), pred["scores"]
    ):
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

score_and_report(
    coco_gt,
    results,
    image_ids,
    sigmas,
    gt_keypoints,
    csv_prefix,
    params={
        # Recorded alongside the scores so a CSV is self-describing later on.
        "split": args.split,
        "super_animal": args.super_animal,
        "pose_model": args.pose_model,
        "detector": args.detector,
        "max_individuals": args.max_individuals,
        "score_threshold": args.score_threshold,
    },
)
