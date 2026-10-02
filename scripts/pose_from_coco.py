"""Add DeepLabCut keypoints to an existing COCO segmentation file.

The counterpart to ``examples/sam3video_to_coco.py``: instead of re-running SAM 3,
this reads the annotations it already wrote (``images`` + per-instance ``bbox`` and
``segmentation``) and only fits pose. Each frame is loaded from ``file_name``, each
instance's segmentation is decoded back to a binary mask, and box + mask go
straight into :class:`~deeplabsam.pose.backends.dlc.DLCPoseHead` -- the mask zeroes
the crop's background so the pose stays on the intended animal.

Keypoints are written back in COCO person-keypoints form:
  * ``categories[*]["keypoints"]`` -- the SuperAnimal bodypart names,
  * ``annotations[*]["keypoints"]`` -- flat ``[x, y, v] * K`` (``v = 2`` when the
    model's confidence clears ``--threshold``, else 0),
  * ``annotations[*]["keypoint_scores"]`` -- the raw per-keypoint confidences,
  * ``annotations[*]["num_keypoints"]`` -- how many cleared the threshold.
Everything already in the file (ids, scores, track ids, masks) is carried through
untouched, so the frames on disk still line up.

With ``--video`` (on by default) an annotated mp4 is written alongside the output
so the masks + keypoints can be eyeballed before the file is used for training.

Example:
    uv run scripts/pose_from_coco.py \
        --in /home/juan/Pictures/sam_masks/cleaned/barnes_maze2/annotations.json \
        --out /home/juan/Pictures/sam_masks/cleaned/barnes_maze2/annotations_pose.json

"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from tqdm import tqdm

from deeplabsam.pose.backends.dlc import DLCPoseHead
from deeplabsam.pose.from_coco import decode_segmentation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        required=True,
        help="Input COCO annotations.json with segmentations (from sam3video_to_coco.py).",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        type=Path,
        default=None,
        help="Where to write the keypoint-augmented COCO file "
        "(default: <in>_pose.json next to the input).",
    )
    p.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Directory the image `file_name` entries are relative to "
        "(default: the input file's directory).",
    )
    p.add_argument(
        "--super-animal",
        default="superanimal_topviewmouse",
        help="SuperAnimal pose model to fit.",
    )
    p.add_argument(
        "--model-name",
        default=None,
        help="Backbone override (default: hrnet_w32, or resnet_50 for bird).",
    )
    p.add_argument(
        "--input-size",
        type=int,
        default=256,
        help="Letterbox size fed to the pose model; must divide by the backbone's "
        "pad divisor (32 for HRNet).",
    )
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto).")
    p.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Keypoint confidence at or above which a keypoint counts as visible "
        "(COCO v=2) and is drawn in the video.",
    )
    p.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Stop after this many images (debug).",
    )
    p.add_argument(
        "--video",
        type=Path,
        default=None,
        help="Annotated mp4 to write for visual QC "
        "(default: <out>_check.mp4; use --no-video to skip).",
    )
    p.add_argument("--no-video", action="store_true", help="Skip the QC video.")
    p.add_argument("--fps", type=float, default=30.0, help="QC video frame rate.")
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for torch/numpy so repeated runs are byte-identical.",
    )
    return p.parse_args()


def to_coco_keypoints(
    pose: np.ndarray,
    threshold: float,
) -> tuple[list[float], list[float], int]:
    """(K, 3) ``(x, y, conf)`` -> COCO ``keypoints`` flat list, scores, count.

    Sub-threshold keypoints keep their coordinates but get ``v = 0``: the pose is
    still there for anything that wants to re-threshold later, while COCO-native
    tooling treats them as not labelled.
    """
    visible = pose[:, 2] >= threshold
    flat: list[float] = []
    for (x, y, _), v in zip(pose.tolist(), visible.tolist()):
        flat += [round(x, 2), round(y, 2), 2 if v else 0]
    scores = [round(c, 5) for c in pose[:, 2].tolist()]
    return flat, scores, int(visible.sum())


def main() -> None:
    args = parse_args()
    torch.set_float32_matmul_precision("high")
    # Pose inference is already deterministic (eval mode, no sampling), but seeding
    # + pinning cuDNN's algorithm choice makes that a guarantee rather than a
    # property that a future backend change could quietly break.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    in_path = args.in_path
    out_path = args.out_path or in_path.with_name(f"{in_path.stem}_pose.json")
    image_root = args.image_root or in_path.parent

    with open(in_path) as f:
        coco = json.load(f)

    name_by_id = {c["id"]: c["name"] for c in coco["categories"]}

    # COCO allows several instances per image; group so each frame is decoded once
    # and all of its instances are posed in a single batched forward.
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    # Frame order for the video (and for cache-friendly reads); files written by
    # sam3video_to_coco.py carry `frame_index`, plain COCO files fall back to id.
    images = sorted(coco["images"], key=lambda im: im.get("frame_index", im["id"]))
    if args.max_images is not None:
        images = images[: args.max_images]

    print(f"Loading {args.super_animal} pose head ...")
    head = DLCPoseHead(
        super_animal=args.super_animal,
        model_name=args.model_name,
        device=args.device,
        input_size=args.input_size,
    )
    print(f"  {head.num_bodyparts} bodyparts ({head.model_name}) on {head.device}")

    # Same bodypart list for every category: one pose model fits them all here.
    for category in coco["categories"]:
        category["keypoints"] = list(head.bodyparts)

    writer = None
    if not args.no_video:
        video_path = args.video or out_path.with_name(f"{out_path.stem}_check.mp4")
        mask_annot = sv.MaskAnnotator(color_lookup=sv.ColorLookup.TRACK)
        box_annot = sv.BoxAnnotator(color_lookup=sv.ColorLookup.TRACK)
        label_annot = sv.LabelAnnotator(color_lookup=sv.ColorLookup.TRACK)
        # VertexAnnotator takes a single color (no ColorLookup), so keypoints are
        # drawn one instance at a time with that instance's track color -- the same
        # ColorPalette.DEFAULT.by_idx the mask/box annotators use for TRACK, so a
        # mouse's dots match its mask. Annotators are cached per track id.
        vertex_annots: dict[int, sv.VertexAnnotator] = {}

    def vertex_annotator_for(track_id: int) -> sv.VertexAnnotator:
        if track_id not in vertex_annots:
            vertex_annots[track_id] = sv.VertexAnnotator(
                color=sv.ColorPalette.DEFAULT.by_idx(track_id),
                radius=3,
            )
        return vertex_annots[track_id]

    n_posed = 0
    empty_masks = 0
    missing: list[str] = []
    pbar = tqdm(images, desc="pose")
    for image in pbar:
        anns = by_image.get(image["id"], [])
        path = image_root / image["file_name"]
        bgr = cv2.imread(str(path))
        if bgr is None:
            missing.append(image["file_name"])
            continue
        height, width = bgr.shape[:2]

        if anns:
            masks = np.stack(
                [decode_segmentation(a["segmentation"], height, width) for a in anns],
            )
            # COCO bbox is xywh; the pose head crops from xyxy.
            xywh = np.array([a["bbox"] for a in anns], dtype=np.float32)
            xyxy = np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], axis=1)
            # An instance whose segmentation didn't survive export would otherwise
            # be posed on an all-zero crop; fall back to its bbox rectangle, which
            # is what an unmasked pose head would see anyway.
            for i in np.flatnonzero(~masks.any(axis=(1, 2))):
                x1, y1, x2, y2 = xyxy[i].round().astype(int)
                masks[i, max(0, y1) : y2, max(0, x1) : x2] = True
                empty_masks += 1
            frame = torch.from_numpy(np.ascontiguousarray(bgr[..., ::-1]))  # BGR->RGB
            poses = head.predict_tensor(frame, xyxy, masks)  # (N, K, 3)
            for ann, pose in zip(anns, poses):
                flat, scores, n_visible = to_coco_keypoints(pose, args.threshold)
                ann["keypoints"] = flat
                ann["keypoint_scores"] = scores
                ann["num_keypoints"] = n_visible
            n_posed += len(anns)

        if writer is None and not args.no_video:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                args.fps,
                (width, height),
            )
        if writer is not None:
            scene = bgr.copy()
            if anns:
                # tracker_id keys the mask/box colors, so an instance keeps its
                # color across frames; class_id is 0-based, category_id is 1-based.
                det = sv.Detections(
                    xyxy=xyxy,
                    mask=masks,
                    class_id=np.array([a["category_id"] - 1 for a in anns], dtype=int),
                    tracker_id=np.array(
                        [a.get("track_id", i) for i, a in enumerate(anns)],
                        dtype=int,
                    ),
                )
                labels = [
                    f"{name_by_id[a['category_id']]} #{t}" for a, t in zip(anns, det.tracker_id)
                ]
                scene = mask_annot.annotate(scene, det)
                scene = box_annot.annotate(scene, det)
                scene = label_annot.annotate(scene, det, labels=labels)
                for pose, track_id in zip(poses, det.tracker_id):
                    conf = pose[None, :, 2]
                    scene = vertex_annotator_for(int(track_id)).annotate(
                        scene,
                        sv.KeyPoints(
                            xy=pose[None, :, :2],
                            keypoint_confidence=conf,
                            visible=conf >= args.threshold,
                        ),
                    )
            writer.write(scene)
        pbar.set_postfix(posed=n_posed)

    if writer is not None:
        writer.release()

    with open(out_path, "w") as f:
        json.dump(coco, f)

    print(f"\nDone. {n_posed} instances posed over {len(images)} images -> {out_path}")
    if writer is not None:
        print(f"QC video -> {video_path}")
    if empty_masks:
        print(f"NOTE: {empty_masks} instances had no segmentation; posed on bbox.")
    if missing:
        print(f"WARNING: {len(missing)} images not found on disk, e.g. {missing[:3]}")


if __name__ == "__main__":
    main()
