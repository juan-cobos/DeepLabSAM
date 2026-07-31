import argparse
import json
from pathlib import Path

import numpy as np
import torch
from faster_coco_eval import COCO
from PIL import Image
from pose_metrics import score_and_report
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
parser.add_argument("--score-threshold", type=float, default=0.05)
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
        f"  head: {pose_head.bodyparts}\n"
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
        "text_prompt": args.text,
        "image_size": args.image_size,
        "score_threshold": args.score_threshold,
        "super_animal": args.super_animal,
    },
)
