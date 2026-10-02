"""COCO -> COCO+keypoints: fit DeepLabCut pose onto an existing SAM 3 COCO export."""

import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from pycocotools import mask as mask_utils
from tqdm import tqdm

from deeplabsam.pose.backends.dlc import DLCPoseHead


def decode_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    """COCO ``segmentation`` (polygons, uncompressed RLE, or RLE) -> HxW bool mask."""
    if isinstance(segmentation, list):  # polygon(s)
        mask = np.zeros((height, width), dtype=np.uint8)
        polygons = [
            np.asarray(p, dtype=np.float32).reshape(-1, 2).round().astype(np.int32)
            for p in segmentation
            if len(p) >= 6  # a polygon needs >= 3 points
        ]
        if polygons:
            cv2.fillPoly(mask, polygons, 1)
        return mask.astype(bool)
    if isinstance(segmentation["counts"], list):  # uncompressed RLE
        rle = mask_utils.frPyObjects(segmentation, height, width)
    else:  # compressed RLE
        rle = segmentation
    return mask_utils.decode(rle).astype(bool)


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


def pose_from_coco(
    in_path: str | Path,
    out_path: str | Path | None = None,
    image_root: str | Path | None = None,
    super_animal: str = "superanimal_topviewmouse",
    model_name: str | None = None,
    input_size: int = 256,
    device: str | None = None,
    threshold: float = 0.3,
    max_images: int | None = None,
    video: str | Path | None = None,
    save_video: bool = True,
    fps: float = 30.0,
) -> Path:
    """Fit ``super_animal`` pose onto every instance in a SAM 3 COCO export.

    ``in_path``'s ``images`` + per-instance ``bbox``/``segmentation`` are used to
    crop and mask each animal; no re-detection happens here. With ``save_video``
    (on by default) an annotated mp4 is written alongside the output so the masks
    + keypoints can be eyeballed before the file is used for training.

    Returns:
        Path to the keypoint-augmented COCO file.

    """
    torch.set_float32_matmul_precision("high")

    in_path = Path(in_path)
    out_path = Path(out_path) if out_path else in_path.with_name(f"{in_path.stem}_pose.json")
    image_root = Path(image_root) if image_root else in_path.parent

    with open(in_path) as f:
        coco = json.load(f)

    name_by_id = {c["id"]: c["name"] for c in coco["categories"]}

    # COCO allows several instances per image; group so each frame is decoded once
    # and all of its instances are posed in a single batched forward.
    by_image: dict[int, list[dict]] = defaultdict(list)
    for ann in coco["annotations"]:
        by_image[ann["image_id"]].append(ann)

    # Frame order for the video (and for cache-friendly reads); files written by
    # segment_to_coco carry `frame_index`, plain COCO files fall back to id.
    images = sorted(coco["images"], key=lambda im: im.get("frame_index", im["id"]))
    if max_images is not None:
        images = images[:max_images]

    print(f"Loading {super_animal} pose head ...")
    head = DLCPoseHead(
        super_animal=super_animal,
        model_name=model_name,
        device=device,
        input_size=input_size,
    )
    print(f"  {head.num_bodyparts} bodyparts ({head.model_name}) on {head.device}")

    # Same bodypart list for every category: one pose model fits them all here.
    for category in coco["categories"]:
        category["keypoints"] = list(head.bodyparts)

    writer = None
    video_path = None
    if save_video:
        video_path = Path(video) if video else out_path.with_name(f"{out_path.stem}_check.mp4")
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
                flat, scores, n_visible = to_coco_keypoints(pose, threshold)
                ann["keypoints"] = flat
                ann["keypoint_scores"] = scores
                ann["num_keypoints"] = n_visible
            n_posed += len(anns)

        if writer is None and save_video:
            writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
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
                            visible=conf >= threshold,
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
    return out_path
